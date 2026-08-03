"""Independently audit structured-mask power estimates against exact SVDs.

The production structured runner deliberately keeps its implementation hash
independent of this file.  This audit consumes completed shards after the run;
it never imports ``experiment_common`` or ``run_structured_experiments``.

Raw transition payloads are optional.  When they are absent, geometric masks
can be reconstructed from ``mask_gamma``.  For other masks, the audit verifies
the exact-SVD fields persisted in the metric row and labels that evidence as
archival rather than independent.  Summary statistics such as ``a_mean`` and
operator singular values are not sufficient to reconstruct a transition mask.
Exit status 2 means numerical checks passed but independent coverage is
incomplete; only a full independent audit exits zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


AUDIT_SCHEMA_VERSION = 1
DEFAULT_MODELS = ("mamba2", "retnet", "based")
VALIDATION_RTOL = 5e-4
VALIDATION_ATOL = 2e-5
SHORT_LENGTH_LIMIT = 512
SUBSET_LENGTH = 1024
SUBSET_MODULUS = 10
SCALAR_VARIANTS = {
    "mamba2": {"native", "all_one", "identity", "constant_decay", "transition_shuffled"},
    "retnet": {"native", "all_one", "identity", "decay_0.5", "decay_0.9", "decay_0.99"},
    "based": {"native_raw", "identity"},
}
IDENTITY_COLUMNS = (
    "checkpoint",
    "document_id",
    "length",
    "layer",
    "head_or_channel",
    "operator_variant",
    "observation_id",
)
INLINE_TRANSITION_COLUMNS = ("mask_transitions", "transition_values", "transitions")
TRANSITION_REFERENCE_COLUMNS = ("mask_transition_ref", "transition_ref", "transitions_ref")
TRANSITION_ARCHIVE_COLUMNS = ("mask_transition_archive", "transition_archive")


class AuditInputError(RuntimeError):
    """Raised when shard contents cannot support a trustworthy audit."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_safe(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(canonical_json(json_safe(row)) + "\n" for row in rows))


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def row_value(row: pd.Series, name: str) -> Any:
    value = row.get(name)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def model_name(row: pd.Series) -> str:
    value = row_value(row, "checkpoint") or row_value(row, "model")
    return "" if value is None else str(value)


def is_scalar_mask(row: pd.Series) -> bool:
    definition = str(row_value(row, "mask_definition") or "")
    if definition == "normalized_operator":
        return False
    if definition in {"scalar_semiseparable_transitions", "geometric_scalar_transitions"}:
        return True
    # This audit targets the production estimator. Nonnegative normalized
    # operators have a gain but no power-iteration metadata and no scalar a_t.
    if row_value(row, "mask_kappa_power_iterations") is not None:
        return True
    if row_value(row, "mask_kappa_exact") is not None:
        return True
    return str(row_value(row, "operator_variant") or "") in SCALAR_VARIANTS.get(model_name(row), set())


def subset_identity(row: pd.Series) -> dict[str, Any]:
    required = ("document_id", "layer", "head_or_channel")
    missing = [name for name in required if row_value(row, name) is None]
    if missing:
        raise AuditInputError(f"Cannot apply the 1024 subset rule; missing columns: {missing}")
    checkpoint = model_name(row)
    if not checkpoint:
        raise AuditInputError("Cannot apply the 1024 subset rule without checkpoint/model.")
    return {
        "checkpoint": checkpoint,
        "document_id": str(row_value(row, "document_id")),
        "layer": int(row_value(row, "layer")),
        "head_or_channel": int(row_value(row, "head_or_channel")),
    }


def subset_digest(row: pd.Series) -> str:
    return hashlib.sha256(canonical_json(subset_identity(row)).encode("ascii")).hexdigest()


def selected_for_exact_audit(row: pd.Series) -> tuple[bool, str | None, str | None]:
    length_value = finite_number(row_value(row, "length"))
    if length_value is None or int(length_value) != length_value or length_value < 1:
        raise AuditInputError(f"Invalid mask length in row {row_value(row, 'observation_id')!r}: {length_value}")
    length = int(length_value)
    if length <= SHORT_LENGTH_LIMIT:
        return True, "all_masks_length_at_most_512", None
    if length == SUBSET_LENGTH:
        digest = subset_digest(row)
        selected = int(digest[:8], 16) % SUBSET_MODULUS == 0
        return selected, "deterministic_10_percent_base_subset_at_1024" if selected else None, digest
    return False, None, None


def normalize_transitions(values: Any, expected_length: int) -> np.ndarray:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError as exc:
            raise AuditInputError(f"Transition JSON is not parseable: {exc}") from exc
    transitions = np.asarray(values, dtype=np.float64).reshape(-1)
    if transitions.size != expected_length:
        raise AuditInputError(
            f"Transition payload length {transitions.size} does not match metric length {expected_length}."
        )
    if not np.isfinite(transitions).all():
        raise AuditInputError("Transition payload contains a non-finite value.")
    tolerance = 32.0 * np.finfo(np.float32).eps
    if np.any(transitions < -tolerance) or np.any(transitions > 1.0 + tolerance):
        raise AuditInputError("Transition payload contains values outside [0, 1].")
    return np.clip(transitions, 0.0, 1.0)


def explicit_semiseparable_mask(transitions: np.ndarray) -> np.ndarray:
    """Form L[i,j] = prod(transitions[j+1:i+1]) with zero-safe indexing."""
    transitions = normalize_transitions(transitions, len(transitions))
    length = transitions.size
    mask = np.zeros((length, length), dtype=np.float64)
    mask[0, 0] = 1.0
    for index in range(1, length):
        mask[index, index] = 1.0
        mask[index, :index] = transitions[index] * mask[index - 1, :index]
    return mask


def memory_sums(transitions: np.ndarray) -> tuple[float, float]:
    transitions = normalize_transitions(transitions, len(transitions))
    length = transitions.size
    rows = np.ones(length, dtype=np.float64)
    columns = np.ones(length, dtype=np.float64)
    for index in range(1, length):
        rows[index] += transitions[index] * rows[index - 1]
    for index in range(length - 2, -1, -1):
        columns[index] += transitions[index + 1] * columns[index + 1]
    return float(rows.max()), float(columns.max())


def _smallest_bidiagonal_gram_eigenvalue(transitions: np.ndarray) -> float:
    """Return lambda_min(B^T B) for B = L^{-1} via Sturm bisection.

    B has ones on its diagonal and ``-a[i]`` on its first subdiagonal.  This
    reduces an exact singular-value problem for dense L to a symmetric
    tridiagonal eigenproblem without approximating L by power iteration.
    """
    transitions = normalize_transitions(transitions, len(transitions))
    length = transitions.size
    if length == 1:
        return 1.0
    off_diagonal = -transitions[1:]
    diagonal = np.ones(length, dtype=np.float64)
    diagonal[:-1] += transitions[1:] ** 2

    pivot_floor = np.finfo(np.float64).tiny

    def eigenvalues_at_or_below(value: float) -> int:
        pivot = diagonal[0] - value
        if abs(pivot) < pivot_floor:
            pivot = -pivot_floor
        count = int(pivot < 0.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for index in range(1, length):
                pivot = diagonal[index] - value - off_diagonal[index - 1] ** 2 / pivot
                if not math.isfinite(float(pivot)):
                    pivot = math.copysign(np.finfo(np.float64).max, float(pivot))
                elif abs(pivot) < pivot_floor:
                    pivot = -pivot_floor
                count += int(pivot < 0.0)
        return count

    lower = 0.0
    upper = float(diagonal.min())
    for _ in range(192):
        midpoint = lower + (upper - lower) / 2.0
        if eigenvalues_at_or_below(midpoint) >= 1:
            upper = midpoint
        else:
            lower = midpoint
        if upper - lower <= max(8.0 * np.finfo(np.float64).eps * upper, np.finfo(np.float64).tiny):
            break
    eigenvalue = lower + (upper - lower) / 2.0
    if not math.isfinite(eigenvalue) or eigenvalue <= 0.0:
        raise ArithmeticError(f"Invalid smallest Gram eigenvalue: {eigenvalue}")
    return eigenvalue


def exact_mask_gain(transitions: np.ndarray, *, dense: bool = False) -> float:
    """Compute the leading singular value of L without power iteration."""
    transitions = normalize_transitions(transitions, len(transitions))
    if dense:
        return float(np.linalg.svd(explicit_semiseparable_mask(transitions), compute_uv=False)[0])
    return float(1.0 / math.sqrt(_smallest_bidiagonal_gram_eigenvalue(transitions)))


def transition_digest(transitions: np.ndarray) -> str:
    canonical = np.asarray(transitions, dtype="<f8").reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(canonical.size).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TransitionPayload:
    values: np.ndarray
    source: str
    reference: str | None = None


class TransitionResolver:
    """Resolve explicit transition payloads without inspecting spectra arrays."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir.resolve()
        self._archives: dict[Path, dict[str, np.ndarray]] = {}
        self._key_index: dict[tuple[str, str], list[Path]] = defaultdict(list)
        patterns = (
            "structured_shards/*/transitions/*.npz",
            "structured_shards/*/transition_arrays/*.npz",
            "structured_shards/*/masks/*.npz",
        )
        for pattern in patterns:
            for path in sorted(self.results_dir.glob(pattern)):
                model = path.parent.parent.name
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        keys = tuple(archive.files)
                except Exception as exc:
                    raise AuditInputError(f"Unreadable transition archive {path}: {exc}") from exc
                for key in keys:
                    self._key_index[(model, key)].append(path)

    def _load_archive(self, path: Path) -> dict[str, np.ndarray]:
        path = path.resolve()
        if path not in self._archives:
            try:
                with np.load(path, allow_pickle=False) as archive:
                    self._archives[path] = {key: np.asarray(archive[key]) for key in archive.files}
            except Exception as exc:
                raise AuditInputError(f"Unreadable transition archive {path}: {exc}") from exc
        return self._archives[path]

    def _resolve_archive_path(self, value: str, shard_path: Path) -> Path:
        candidate = Path(value)
        possibilities = [candidate] if candidate.is_absolute() else [
            self.results_dir / candidate,
            shard_path.parent / candidate,
        ]
        for possibility in possibilities:
            if possibility.exists():
                return possibility.resolve()
        raise AuditInputError(f"Referenced transition archive does not exist: {value}")

    def resolve(self, row: pd.Series, shard_path: Path) -> TransitionPayload | None:
        length = int(row_value(row, "length"))
        for column in INLINE_TRANSITION_COLUMNS:
            value = row_value(row, column)
            if value is not None:
                return TransitionPayload(normalize_transitions(value, length), f"inline_column:{column}")

        reference = next(
            (str(row_value(row, column)) for column in TRANSITION_REFERENCE_COLUMNS if row_value(row, column) is not None),
            None,
        )
        archive_value = next(
            (str(row_value(row, column)) for column in TRANSITION_ARCHIVE_COLUMNS if row_value(row, column) is not None),
            None,
        )
        if archive_value is not None:
            archive_path = self._resolve_archive_path(archive_value, shard_path)
            archive = self._load_archive(archive_path)
            candidate_keys = [reference] if reference else self._candidate_keys(row)
            matches = [key for key in candidate_keys if key in archive]
            if len(matches) != 1:
                raise AuditInputError(
                    f"Expected one transition key in {archive_path}, found {matches} for row "
                    f"{row_value(row, 'observation_id')!r}."
                )
            key = matches[0]
            return TransitionPayload(
                normalize_transitions(archive[key], length),
                "referenced_transition_archive",
                f"{archive_path}:{key}",
            )

        model = model_name(row)
        candidate_keys = [reference] if reference else self._candidate_keys(row)
        matches: list[tuple[Path, str]] = []
        for key in candidate_keys:
            matches.extend((path, key) for path in self._key_index.get((model, key), ()))
        unique_matches = sorted(set(matches), key=lambda item: (str(item[0]), item[1]))
        if len(unique_matches) > 1:
            raise AuditInputError(
                f"Ambiguous transition payload for {row_value(row, 'observation_id')!r}: {unique_matches}"
            )
        if unique_matches:
            archive_path, key = unique_matches[0]
            archive = self._load_archive(archive_path)
            return TransitionPayload(
                normalize_transitions(archive[key], length),
                "indexed_transition_archive",
                f"{archive_path}:{key}",
            )

        gamma = finite_number(row_value(row, "mask_gamma"))
        definition = str(row_value(row, "mask_definition") or "")
        family = str(row_value(row, "mask_family") or "")
        if gamma is not None and (definition == "geometric_scalar_transitions" or family == "geometric"):
            if not 0.0 <= gamma <= 1.0:
                raise AuditInputError(f"Invalid stored geometric decay: {gamma}")
            # The production runner creates geometric transitions in float32.
            values = np.full(length, np.float32(gamma), dtype=np.float32).astype(np.float64)
            return TransitionPayload(values, "reconstructed_from_geometric_metadata", f"mask_gamma={gamma:.17g}")
        return None

    @staticmethod
    def _candidate_keys(row: pd.Series) -> list[str]:
        observation = row_value(row, "observation_id")
        if observation is None:
            return []
        observation = str(observation)
        return [
            observation,
            f"transition-{observation}",
            f"transitions-{observation}",
            f"mask-transition-{observation}",
            f"mask-transitions-{observation}",
        ]


def _validate_completed_shard(path: Path, frame: pd.DataFrame) -> None:
    spectrum_path = path.parent.parent / "spectra" / f"{path.stem}.npz"
    if not spectrum_path.exists():
        raise AuditInputError(f"Metric shard has no paired spectrum archive: {path}")
    if "spectrum_id" not in frame:
        raise AuditInputError(f"Metric shard has no spectrum_id column: {path}")
    try:
        with np.load(spectrum_path, allow_pickle=False) as archive:
            spectrum_keys = set(archive.files)
    except Exception as exc:
        raise AuditInputError(f"Unreadable spectrum archive {spectrum_path}: {exc}") from exc
    metric_keys = {str(value) for value in frame["spectrum_id"].dropna()}
    if metric_keys != spectrum_keys:
        raise AuditInputError(
            f"Metric/spectrum key mismatch in completed shard {path}: "
            f"metrics={len(metric_keys)}, archive={len(spectrum_keys)}"
        )


def load_metric_rows(results_dir: Path, models: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    limitations: list[str] = []
    for model in models:
        shard_root = results_dir / "structured_shards" / model / "metrics"
        shard_paths = sorted(shard_root.glob("*.parquet"))
        if shard_paths:
            for path in shard_paths:
                frame = pd.read_parquet(path)
                _validate_completed_shard(path, frame)
                if "checkpoint" in frame:
                    checkpoints = {str(value) for value in frame["checkpoint"].dropna()}
                    if checkpoints and checkpoints != {model}:
                        raise AuditInputError(
                            f"Metric shard {path} is under model {model!r} but contains checkpoints {checkpoints}."
                        )
                frame["_metric_shard"] = str(path.resolve())
                frames.append(frame)
            continue
        consolidated = results_dir / f"{model}_metrics.parquet"
        if not consolidated.exists():
            raise AuditInputError(f"No completed structured metrics found for requested model {model!r}.")
        frame = pd.read_parquet(consolidated)
        frame["_metric_shard"] = str(consolidated.resolve())
        frames.append(frame)
        limitations.append(
            f"{model}: per-work-unit metric shards were unavailable; the audit used consolidated metrics and "
            "could not verify metric/spectrum pairing per completed shard."
        )
    if not frames:
        raise AuditInputError("No structured metric rows were found.")
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    if "implementation_sha256" in metrics:
        for model, group in metrics.groupby(metrics.apply(model_name, axis=1), sort=True):
            fingerprints = {str(value) for value in group["implementation_sha256"].dropna()}
            if len(fingerprints) != 1:
                raise AuditInputError(
                    f"Structured rows for {model!r} do not have one implementation fingerprint: "
                    f"{sorted(fingerprints)}"
                )
    if "observation_id" not in metrics:
        raise AuditInputError("Structured metrics have no observation_id column.")
    duplicate = metrics["observation_id"].astype(str).duplicated(keep=False)
    if bool(duplicate.any()):
        examples = sorted(set(metrics.loc[duplicate, "observation_id"].astype(str)))[:5]
        raise AuditInputError(f"Duplicate observations across structured shards: {examples}")
    return metrics, limitations


def within_tolerance(measured: float, reference: float) -> bool:
    return abs(measured - reference) <= VALIDATION_ATOL + VALIDATION_RTOL * abs(reference)


def relative_error(measured: float, reference: float) -> float:
    return abs(measured - reference) / max(abs(reference), 1e-30)


def archived_exact_values(row: pd.Series) -> tuple[float, float, float] | None:
    values = tuple(finite_number(row_value(row, name)) for name in ("mask_R_exact", "mask_C_exact", "mask_kappa_exact"))
    if all(value is not None for value in values):
        return values  # type: ignore[return-value]
    if any(value is not None for value in values):
        raise AuditInputError(
            f"Partially populated archived exact fields for observation {row_value(row, 'observation_id')!r}."
        )
    return None


def exact_for_payload(
    payload: TransitionPayload,
    cache: dict[str, tuple[float, float, float, float | None]],
    *,
    bidiagonal_crosscheck_max_length: int,
) -> tuple[tuple[float, float, float, float | None], bool]:
    digest = transition_digest(payload.values)
    if digest in cache:
        return cache[digest], True
    exact_r, exact_c = memory_sums(payload.values)
    exact_gain_value = exact_mask_gain(payload.values, dense=True)
    bidiagonal_relative_error = None
    if payload.values.size <= bidiagonal_crosscheck_max_length:
        bidiagonal_gain = exact_mask_gain(payload.values)
        bidiagonal_relative_error = relative_error(bidiagonal_gain, exact_gain_value)
        if not within_tolerance(exact_gain_value, bidiagonal_gain):
            raise ArithmeticError(
                f"Dense SVD disagrees with the bidiagonal exact solver: "
                f"{exact_gain_value} versus {bidiagonal_gain}"
            )
    cache[digest] = (exact_r, exact_c, exact_gain_value, bidiagonal_relative_error)
    return cache[digest], False


def audit_row(
    row: pd.Series,
    resolver: TransitionResolver,
    cache: dict[str, tuple[float, float, float, float | None]],
    *,
    bidiagonal_crosscheck_max_length: int,
) -> dict[str, Any]:
    selected, selection_rule, selection_hash = selected_for_exact_audit(row)
    if not selected:
        raise ValueError("audit_row called for an unselected row")
    detail = {name: json_safe(row_value(row, name)) for name in IDENTITY_COLUMNS}
    detail.update(
        {
            "selection_rule": selection_rule,
            "selection_sha256": selection_hash,
            "metric_shard": str(row_value(row, "_metric_shard")),
            "validation_rtol": VALIDATION_RTOL,
            "validation_atol": VALIDATION_ATOL,
            "issues": [],
        }
    )
    issues: list[str] = detail["issues"]
    estimated_gain = finite_number(row_value(row, "mask_kappa"))
    stored_r = finite_number(row_value(row, "mask_R"))
    stored_c = finite_number(row_value(row, "mask_C"))
    iterations = finite_number(row_value(row, "mask_kappa_power_iterations"))
    residual = finite_number(row_value(row, "mask_kappa_power_residual"))
    detail.update(
        {
            "stored_mask_R": stored_r,
            "stored_mask_C": stored_c,
            "stored_power_gain": estimated_gain,
            "stored_power_iterations": iterations,
            "stored_power_residual": residual,
        }
    )
    if estimated_gain is None or stored_r is None or stored_c is None:
        issues.append("missing_or_nonfinite_stored_memory_metric")
    if iterations is None or iterations < 1 or int(iterations) != iterations:
        issues.append("missing_or_invalid_power_iteration_count")
    if residual is None or residual < 0.0:
        issues.append("missing_or_invalid_power_residual")

    payload = resolver.resolve(row, Path(str(row_value(row, "_metric_shard"))))
    archived = archived_exact_values(row)
    if payload is not None:
        exact, cache_hit = exact_for_payload(
            payload,
            cache,
            bidiagonal_crosscheck_max_length=bidiagonal_crosscheck_max_length,
        )
        exact_r, exact_c, exact_gain_value, bidiagonal_relative_error = exact
        evidence = "independent_exact_recomputation"
        detail.update(
            {
                "transition_source": payload.source,
                "transition_reference": payload.reference,
                "transition_sha256": transition_digest(payload.values),
                "exact_result_cache_hit": cache_hit,
                "bidiagonal_crosscheck_relative_error": bidiagonal_relative_error,
            }
        )
    elif archived is not None:
        exact_r, exact_c, exact_gain_value = archived
        evidence = "archived_runner_exact_fields_only"
        detail.update(
            {
                "transition_source": None,
                "transition_reference": None,
                "transition_sha256": None,
                "exact_result_cache_hit": False,
                "bidiagonal_crosscheck_relative_error": None,
            }
        )
    else:
        exact_r = exact_c = exact_gain_value = None
        evidence = "missing_exact_evidence"
        detail.update(
            {
                "transition_source": None,
                "transition_reference": None,
                "transition_sha256": None,
                "exact_result_cache_hit": False,
                "bidiagonal_crosscheck_relative_error": None,
            }
        )
        issues.append("no_transition_payload_geometric_reconstruction_or_archived_exact_fields")

    detail["evidence"] = evidence
    detail["reference_mask_R"] = exact_r
    detail["reference_mask_C"] = exact_c
    detail["reference_exact_gain"] = exact_gain_value
    if exact_gain_value is not None and estimated_gain is not None:
        detail["gain_absolute_error"] = abs(estimated_gain - exact_gain_value)
        detail["gain_relative_error"] = relative_error(estimated_gain, exact_gain_value)
        detail["gain_within_tolerance"] = within_tolerance(estimated_gain, exact_gain_value)
        if not detail["gain_within_tolerance"]:
            issues.append("stored_power_gain_disagrees_with_exact_svd")
    else:
        detail["gain_absolute_error"] = None
        detail["gain_relative_error"] = None
        detail["gain_within_tolerance"] = False

    for label, stored, exact in (("row", stored_r, exact_r), ("column", stored_c, exact_c)):
        passed = stored is not None and exact is not None and within_tolerance(stored, exact)
        detail[f"{label}_sum_within_tolerance"] = passed
        detail[f"{label}_sum_relative_error"] = (
            relative_error(stored, exact) if stored is not None and exact is not None else None
        )
        if exact is not None and not passed:
            issues.append(f"stored_{label}_sum_disagrees_with_exact_recurrence")

    if archived is not None and payload is not None:
        archived_r, archived_c, archived_gain = archived
        for label, archived_value, recomputed in (
            ("mask_R", archived_r, exact_r),
            ("mask_C", archived_c, exact_c),
            ("mask_kappa", archived_gain, exact_gain_value),
        ):
            passed = within_tolerance(archived_value, recomputed)
            detail[f"archived_{label}_matches_recomputation"] = passed
            if not passed:
                issues.append(f"archived_{label}_disagrees_with_independent_recomputation")

    archived_passed = row_value(row, "mask_exact_validation_passed")
    if archived_passed is not None:
        detail["archived_validation_passed"] = bool(archived_passed)
        if not bool(archived_passed):
            issues.append("production_runner_marked_exact_validation_failed")
    else:
        detail["archived_validation_passed"] = None
    if evidence == "archived_runner_exact_fields_only" and archived_passed is None:
        issues.append("archived_exact_fields_lack_validation_pass_flag")
    archived_relative = finite_number(row_value(row, "mask_kappa_validation_relative_error"))
    detail["archived_gain_relative_error"] = archived_relative
    if archived_relative is not None and archived is not None and estimated_gain is not None:
        calculated = relative_error(estimated_gain, archived[2])
        if not math.isclose(archived_relative, calculated, rel_tol=1e-7, abs_tol=1e-12):
            issues.append("archived_gain_relative_error_field_is_inconsistent")

    missing_issue = "no_transition_payload_geometric_reconstruction_or_archived_exact_fields"
    detail["status"] = (
        "pass"
        if not issues
        else "missing"
        if evidence == "missing_exact_evidence" and set(issues) == {missing_issue}
        else "fail"
    )
    return detail


def grouped_counts(details: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in details).items()))


def grouped_model_evidence(details: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in details:
        result[str(row.get("checkpoint"))][str(row.get("evidence"))] += 1
    return {model: dict(sorted(counts.items())) for model, counts in sorted(result.items())}


def run_audit(
    results_dir: Path,
    models: tuple[str, ...] = DEFAULT_MODELS,
    *,
    bidiagonal_crosscheck_max_length: int = 1024,
    require_independent: bool = False,
    require_full_length_scope: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    results_dir = results_dir.resolve()
    metrics, limitations = load_metric_rows(results_dir, models)
    resolver = TransitionResolver(results_dir)
    scalar = metrics[metrics.apply(is_scalar_mask, axis=1)].copy()
    if scalar.empty:
        raise AuditInputError("Structured metrics contain no scalar semiseparable mask rows.")
    short_mask_rows_present = int((pd.to_numeric(scalar["length"], errors="coerce") <= SHORT_LENGTH_LIMIT).sum())
    long_mask_rows_present = int((pd.to_numeric(scalar["length"], errors="coerce") == SUBSET_LENGTH).sum())
    if require_full_length_scope and short_mask_rows_present == 0:
        raise AuditInputError("No scalar mask rows at or below length 512 are present; required audit scope is absent.")
    if require_full_length_scope and long_mask_rows_present == 0:
        raise AuditInputError("No scalar mask rows at length 1024 are present; required subset audit scope is absent.")
    selected_indices: list[int] = []
    selection_counts: Counter[str] = Counter()
    eligible_1024_bases: set[tuple[Any, ...]] = set()
    selected_1024_bases: set[tuple[Any, ...]] = set()
    for index, row in scalar.iterrows():
        selected, reason, _ = selected_for_exact_audit(row)
        length = int(row_value(row, "length"))
        if length == SUBSET_LENGTH:
            base = (
                model_name(row),
                row_value(row, "document_id"),
                int(row_value(row, "layer")),
                int(row_value(row, "head_or_channel")),
            )
            eligible_1024_bases.add(base)
            if selected:
                selected_1024_bases.add(base)
        if selected:
            selected_indices.append(index)
            selection_counts[str(reason)] += 1
    if not selected_indices:
        raise AuditInputError("No scalar masks satisfied the exact-SVD audit selection rule.")

    cache: dict[str, tuple[float, float, float, float | None]] = {}
    details: list[dict[str, Any]] = []
    for index in selected_indices:
        details.append(
            audit_row(
                scalar.loc[index],
                resolver,
                cache,
                bidiagonal_crosscheck_max_length=bidiagonal_crosscheck_max_length,
            )
        )

    evidence_counts = Counter(str(row["evidence"]) for row in details)
    status_counts = Counter(str(row["status"]) for row in details)
    archival = evidence_counts["archived_runner_exact_fields_only"]
    missing = evidence_counts["missing_exact_evidence"]
    failures = status_counts["fail"]
    if archival:
        limitations.append(
            f"{archival} selected masks have no reconstructible transition payload. Their power estimates were "
            "checked against exact fields stored by the production runner, not against an independently rerun SVD."
        )
    mamba_archival = sum(
        row["checkpoint"] == "mamba2" and row["evidence"] == "archived_runner_exact_fields_only"
        for row in details
    )
    if mamba_archival:
        limitations.append(
            f"{mamba_archival} selected Mamba native or shuffled masks are archival-only because current structured "
            "shards store operator spectra and transition summaries, but not the full input-dependent a_t arrays. "
            "Those summaries cannot be inverted to recover L."
        )
    if missing:
        limitations.append(f"{missing} selected masks have neither reconstructible transitions nor complete archived exact fields.")

    if failures:
        overall_status = "FAIL"
        exit_code = 1
    elif missing:
        overall_status = "INCOMPLETE"
        exit_code = 2
    elif require_independent and archival:
        overall_status = "INCOMPLETE_INDEPENDENT_COVERAGE"
        exit_code = 2
    elif archival:
        overall_status = "PASS_WITH_ARCHIVAL_LIMITATIONS"
        exit_code = 2
    else:
        overall_status = "PASS"
        exit_code = 0

    source_path = Path(__file__)
    report = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source": str(source_path.resolve()),
        "audit_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "results_dir": str(results_dir),
        "requested_models": list(models),
        "status": overall_status,
        "passed_numerical_checks": failures == 0,
        "complete_exact_evidence": missing == 0,
        "complete_independent_recomputation": archival == 0 and missing == 0,
        "require_independent": require_independent,
        "require_full_length_scope": require_full_length_scope,
        "selection_protocol": {
            "all_masks_at_or_below_length": SHORT_LENGTH_LIMIT,
            "subset_length": SUBSET_LENGTH,
            "subset_rule": (
                "sha256(canonical JSON of checkpoint, document_id, layer, head_or_channel)[:8] modulo 10 == 0; "
                "all scalar variants sharing a selected base are audited"
            ),
            "excluded_length": 768,
        },
        "numerical_protocol": {
            "gain_reference": "largest value from numpy.linalg.svd on the explicit float64 mask L",
            "solver": "dense float64 SVD (no power iteration)",
            "secondary_crosscheck": "sigma_max(L) = 1 / sigma_min(L^{-1}); L^{-1} is lower bidiagonal",
            "secondary_solver": "float64 symmetric-tridiagonal Sturm bisection",
            "bidiagonal_crosscheck_max_length": bidiagonal_crosscheck_max_length,
            "rtol": VALIDATION_RTOL,
            "atol": VALIDATION_ATOL,
            "geometric_transition_dtype": "float32, matching production torch.full, then promoted to float64",
        },
        "coverage": {
            "all_structured_metric_rows": len(metrics),
            "scalar_mask_rows": len(scalar),
            "scalar_mask_rows_at_or_below_512": short_mask_rows_present,
            "scalar_mask_rows_at_1024": long_mask_rows_present,
            "selected_mask_rows": len(details),
            "selection_counts": dict(sorted(selection_counts.items())),
            "eligible_1024_bases": len(eligible_1024_bases),
            "selected_1024_bases": len(selected_1024_bases),
            "selected_1024_base_fraction": (
                len(selected_1024_bases) / len(eligible_1024_bases) if eligible_1024_bases else None
            ),
            "evidence_counts": dict(sorted(evidence_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "by_model_and_evidence": grouped_model_evidence(details),
            "by_length": grouped_counts(details, "length"),
            "unique_exact_transition_problems_solved": len(cache),
        },
        "limitations": sorted(set(limitations)),
        "failure_examples": [row for row in details if row["status"] == "fail"][:10],
        "missing_evidence_examples": [row for row in details if row["status"] == "missing"][:10],
    }
    return report, details, exit_code


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not models:
        raise argparse.ArgumentTypeError("At least one model is required.")
    unknown = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown structured models: {unknown}")
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments_2027"))
    parser.add_argument("--models", type=parse_models, default=DEFAULT_MODELS)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--details", type=Path)
    parser.add_argument(
        "--bidiagonal-crosscheck-max-length",
        type=int,
        default=1024,
        help="Also compare dense SVD with the independent bidiagonal solver through this length.",
    )
    parser.add_argument(
        "--require-independent",
        action="store_true",
        help="Label archival-only coverage as INCOMPLETE_INDEPENDENT_COVERAGE (all incomplete coverage exits 2).",
    )
    args = parser.parse_args(argv)
    if args.bidiagonal_crosscheck_max_length < 0:
        parser.error("--bidiagonal-crosscheck-max-length must be nonnegative")
    report_path = args.report or args.results_dir / "logs" / "structured_memory_svd_audit.json"
    details_path = args.details or args.results_dir / "logs" / "structured_memory_svd_audit_rows.jsonl"
    try:
        report, details, exit_code = run_audit(
            args.results_dir,
            args.models,
            bidiagonal_crosscheck_max_length=args.bidiagonal_crosscheck_max_length,
            require_independent=args.require_independent,
            require_full_length_scope=True,
        )
    except Exception as exc:
        error_report = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "results_dir": str(args.results_dir.resolve()),
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_json(report_path, error_report)
        print(f"Structured memory SVD audit ERROR: {exc}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        return 1
    atomic_json(report_path, report)
    atomic_jsonl(details_path, details)
    print(
        f"Structured memory SVD audit {report['status']}: "
        f"selected={report['coverage']['selected_mask_rows']}, "
        f"evidence={report['coverage']['evidence_counts']}, "
        f"statuses={report['coverage']['status_counts']}"
    )
    print(f"Report: {report_path}")
    print(f"Row details: {details_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
