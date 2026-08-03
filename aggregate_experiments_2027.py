#!/usr/bin/env python3
"""Strict aggregation, bootstrap, plotting, and reporting for ICLR 2027.

The experiment runners deliberately preserve head-resolved observations.  This
program is the only place where inferential aggregation happens: heads are
first reduced by a median within each document, and documents are then treated
as the independent units in every confidence interval and slope fit.
The paper scope is limited to trained GPT-2, Based, Mamba-2, and RetNet runs.

Production usage::

    python aggregate_experiments_2027.py validate \
        --results-dir experiments_2027
    python aggregate_experiments_2027.py analyze \
        --results-dir experiments_2027

The production defaults are frozen to 32 documents, 2,000 document bootstrap
resamples, seed 20270802, and log--log slope fits over lengths >=256.  The
``self-test`` command uses small synthetic arrays and does not relax production
analysis settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


BASE_SEED = 20270802
BOOTSTRAP_RESAMPLES = 2000
SLOPE_MIN_LENGTH = 256
EXPECTED_DOCUMENTS = 32
PRIMARY_LENGTHS = (64, 128, 256, 384, 512, 768, 1024)
PAPER_MODELS = ("gpt2", "based", "mamba2", "retnet")
RANK_METRICS = ("srank", "erank_1", "erank_2")
RANK_LABELS = {
    "srank": "stable rank",
    "erank_1": r"$\mathrm{erank}_1$",
    "erank_2": r"$\mathrm{erank}_2$",
}
LAYERS = {
    "gpt2": (0, 5, 11),
    "based": (1, 11, 21),
    "mamba2": (0, 31, 63),
    "retnet": (0, 11, 23),
}
ALL_GPT2_LAYERS = tuple(range(12))
GPT2_INTENSIVE_HEADS = (0, 3, 6, 9)
HEADS = {
    "gpt2": tuple(range(12)),
    "based": tuple(range(16)),
    "mamba2": (0, 16, 32, 48, 64, 80, 96, 112),
    "retnet": tuple(range(8)),
}
REVISIONS = {
    "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "mamba2": "4f086c08c1e0f07bdc50ca25125dbbf7475d21da",
    "retnet": "7fddefc4d5e196a8d1f076bb7612d54321b3effe",
    "based": "4a63ec61ddb6043a88c198463a7ae75fa90d4038",
    "wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
}
STRUCTURED_VARIANTS = {
    "based": ("native_raw", "native_normalized", "identity"),
    "mamba2": ("native", "all_one", "identity", "constant_decay", "transition_shuffled"),
    "retnet": ("native", "all_one", "identity", "decay_0.5", "decay_0.9", "decay_0.99", "causal_softmax"),
}
NATIVE_VARIANT = {
    "gpt2": "native_softmax",
    "based": "native_normalized",
    "mamba2": "native",
    "retnet": "native",
}
OBSERVATION_KEY = (
    "checkpoint",
    "corpus",
    "document_id",
    "input_condition",
    "length",
    "layer",
    "head_or_channel",
    "operator_variant",
)
PRIMARY_REQUIRED_COLUMNS = (
    "checkpoint_revision",
    "corpus",
    "document_id",
    "sample_offset",
    "input_condition",
    "length",
    "layer",
    "head_or_channel",
    "operator_variant",
    "srank",
    "erank_1",
    "erank_2",
    "operator_norm",
    "frobenius_norm",
    "nuclear_norm",
    "best_rank1_error",
    "sink_S",
    "sink_D",
    "sink_D_over_S",
    "sink_rho2",
    "sink_rho1",
    "score_margin_summary",
    "collision_summary",
    "mask_R",
    "mask_C",
    "mask_kappa",
    "q_rank_metrics",
    "k_rank_metrics",
    "score_rank_metrics",
)
JSON_COLUMNS = {
    "q_rank_metrics",
    "k_rank_metrics",
    "score_rank_metrics",
    "score_margin_summary",
    "collision_summary",
    "block_products",
    "effective_half_life_summary",
    "canonical_correlations_by_rank",
    "token_alignment_by_rank",
    "projection_overlap_by_rank",
    "scale_diagnostic",
    "geometry_by_rank",
    "scale_diagnostics",
    "low_rank_reconstruction",
    "token_alignment",
    "projection_overlap",
}


class AnalysisError(RuntimeError):
    """Raised when an artifact or statistical contract is incomplete."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AnalysisError(f"Expected an object in {path}:{line_number}")
            rows.append(value)
    return rows


def parse_json(value: Any, *, context: str = "JSON field") -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid {context}: {value[:160]!r}") from exc


def json_safe(value: Any) -> Any:
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


@dataclass
class Audit:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, *, required: bool = True) -> None:
        self.checks.append(
            {"check": name, "passed": bool(passed), "required": bool(required), "detail": str(detail)}
        )

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [row for row in self.checks if row["required"] and not row["passed"]]

    def require(self) -> None:
        if self.failures:
            preview = "\n".join(f"- {row['check']}: {row['detail']}" for row in self.failures[:20])
            remainder = len(self.failures) - min(20, len(self.failures))
            if remainder:
                preview += f"\n- ... and {remainder} more failures"
            raise AnalysisError(f"Production analysis validation failed:\n{preview}")


def _source_candidates(results_dir: Path, model: str) -> list[Path]:
    preferred = [
        results_dir / f"{model}_metrics.parquet",
        results_dir / f"{model}_metrics.jsonl",
    ]
    for path in preferred:
        if path.exists():
            return [path]
    candidates: list[Path] = []
    for pattern in (
        f"shards/{model}/**/*.parquet",
        f"shards/{model}/**/*.jsonl",
        f"{model}_shards/**/*.parquet",
        f"{model}_shards/**/*.jsonl",
    ):
        candidates.extend(results_dir.glob(pattern))
    return sorted(set(candidates))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.DataFrame(read_jsonl(path))
    raise AnalysisError(f"Unsupported metrics table: {path}")


def _normalize_model_table(frame: pd.DataFrame, model: str, sources: Sequence[Path]) -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty:
        raise AnalysisError(f"Metrics source for {model} is empty: {', '.join(map(str, sources))}")
    if "checkpoint" not in frame:
        frame["checkpoint"] = model
    if "model" not in frame:
        frame["model"] = model
    frame["model"] = model
    if "head" in frame and "head_or_channel" not in frame:
        frame["head_or_channel"] = frame["head"]
    if "operator" in frame and "operator_variant" not in frame:
        frame["operator_variant"] = frame["operator"]
    if "q_rank_metrics" not in frame and "q_factor_metrics" in frame:
        frame["q_rank_metrics"] = frame["q_factor_metrics"]
    if "k_rank_metrics" not in frame and "k_factor_metrics" in frame:
        frame["k_rank_metrics"] = frame["k_factor_metrics"]
    if "score_margin_summary" not in frame:
        margin_fields = ("margin_median", "margin_p05", "margin_log_i_slope", "margin_log_i_intercept")
        if any(name in frame for name in margin_fields):
            frame["score_margin_summary"] = frame.apply(
                lambda row: canonical_json(
                    {name: json_safe(row.get(name)) for name in margin_fields if json_safe(row.get(name)) is not None}
                ),
                axis=1,
            )
    if "collision_summary" not in frame:
        collision_fields = (
            "collision_scaled_median",
            "collision_scaled_p95",
            "conditional_collision_median",
            "conditional_collision_p95",
        )
        if any(name in frame for name in collision_fields):
            frame["collision_summary"] = frame.apply(
                lambda row: canonical_json(
                    {name: json_safe(row.get(name)) for name in collision_fields if json_safe(row.get(name)) is not None}
                ),
                axis=1,
            )
    for name in ("length", "layer", "head_or_channel", "sample_offset"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("Int64")
    for name in JSON_COLUMNS & set(frame.columns):
        frame[name] = frame[name].map(
            lambda value: canonical_json(json_safe(value))
            if isinstance(value, (dict, list, tuple, np.ndarray))
            else value
        )
    frame["source_artifact"] = ",".join(str(path) for path in sources)
    return frame


def load_metrics(results_dir: Path, audit: Audit) -> tuple[pd.DataFrame, dict[str, list[Path]]]:
    frames: list[pd.DataFrame] = []
    sources_by_model: dict[str, list[Path]] = {}
    for model in PAPER_MODELS:
        sources = _source_candidates(results_dir, model)
        sources_by_model[model] = sources
        audit.add(f"metrics source: {model}", bool(sources), ", ".join(map(str, sources)) or "missing")
        if not sources:
            continue
        parts = [_read_table(path) for path in sources]
        frame = pd.concat(parts, ignore_index=True, sort=False)
        frames.append(_normalize_model_table(frame, model, sources))
    if not frames:
        return pd.DataFrame(), sources_by_model
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in OBSERVATION_KEY:
        audit.add(f"canonical column: {column}", column in combined, "present" if column in combined else "missing")
    if not all(column in combined for column in OBSERVATION_KEY):
        return combined, sources_by_model
    missing_key = combined[list(OBSERVATION_KEY)].isna().any(axis=1)
    audit.add("non-null observation keys", not missing_key.any(), f"{int(missing_key.sum())} rows have null keys")
    duplicate = combined.duplicated(list(OBSERVATION_KEY), keep=False)
    duplicate_examples = combined.loc[duplicate, list(OBSERVATION_KEY)].head(5).to_dict("records")
    audit.add("unique observations", not duplicate.any(), f"duplicates={int(duplicate.sum())}; examples={duplicate_examples}")
    if not duplicate.any() and not missing_key.any():
        combined["observation_id"] = combined.apply(
            lambda row: hashlib.sha256(
                canonical_json([json_safe(row[name]) for name in OBSERVATION_KEY]).encode("ascii")
            ).hexdigest()[:24],
            axis=1,
        )
    return combined, sources_by_model


def _grid_audit(
    audit: Audit,
    name: str,
    frame: pd.DataFrame,
    expected: Mapping[str, Sequence[Any]],
    *,
    nonnull: Sequence[str] = (),
) -> None:
    columns = list(expected)
    absent = [column for column in columns if column not in frame]
    if absent:
        audit.add(name, False, f"missing grid columns {absent}")
        return
    selected = frame.copy()
    for column, values in expected.items():
        selected = selected[selected[column].isin(values)]
    expected_index = pd.MultiIndex.from_product([list(expected[column]) for column in columns], names=columns)
    observed_index = pd.MultiIndex.from_frame(selected[columns].drop_duplicates())
    missing = expected_index.difference(observed_index)
    # Rows outside the requested product were filtered above.  Within the
    # product, any repeated coordinate is a schema error rather than an extra
    # replicate: heads and documents already appear explicitly in the key.
    duplicate_rows = int(selected.duplicated(columns, keep=False).sum())
    finite_failures: dict[str, int] = {}
    for column in nonnull:
        if column not in selected:
            finite_failures[column] = len(expected_index)
            continue
        numeric = pd.to_numeric(selected[column], errors="coerce")
        finite_failures[column] = int((~np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))).sum())
    passed = len(missing) == 0 and duplicate_rows == 0 and not any(finite_failures.values())
    examples = [tuple(json_safe(item) for item in entry) for entry in list(missing[:3])]
    audit.add(
        name,
        passed,
        f"expected={len(expected_index)}, observed_unique={len(observed_index)}, missing={len(missing)} "
        f"duplicate_rows={duplicate_rows}, nonfinite={finite_failures}, missing_examples={examples}",
    )


def _json_contract_audit(
    audit: Audit,
    name: str,
    frame: pd.DataFrame,
    column: str,
    validator: Callable[[Any], bool],
) -> None:
    if column not in frame:
        audit.add(name, False, f"column {column} is missing")
        return
    invalid: list[int] = []
    for index, value in frame[column].items():
        try:
            parsed = parse_json(value, context=f"{column} row {index}")
            if not validator(parsed):
                invalid.append(int(index))
        except AnalysisError:
            invalid.append(int(index))
    audit.add(name, not invalid, f"rows={len(frame)}, invalid={len(invalid)}, examples={invalid[:5]}")


def _json_candidate_contract_audit(
    audit: Audit,
    name: str,
    frame: pd.DataFrame,
    candidates: Sequence[str],
    validator: Callable[[Any, str], bool],
) -> None:
    """Validate the first populated candidate in each row of a union schema."""
    invalid: list[int] = []
    selected_counts: dict[str, int] = {}
    for index, row in frame.iterrows():
        selected: tuple[str, Any] | None = None
        for column in candidates:
            if column not in frame:
                continue
            value = row[column]
            if value is None or value is pd.NA:
                continue
            try:
                if bool(pd.isna(value)):
                    continue
            except (TypeError, ValueError):
                pass
            selected = (column, value)
            break
        if selected is None:
            invalid.append(int(index))
            continue
        column, value = selected
        selected_counts[column] = selected_counts.get(column, 0) + 1
        try:
            parsed = parse_json(value, context=f"{column} row {index}")
            if not validator(parsed, column):
                invalid.append(int(index))
        except AnalysisError:
            invalid.append(int(index))
    audit.add(
        name,
        not invalid,
        f"rows={len(frame)}, invalid={len(invalid)}, selected={selected_counts}, examples={invalid[:5]}",
    )


def _low_rank_contract(value: Any, factor_dimension: int | None) -> bool:
    parsed = parse_json(value)
    if not isinstance(parsed, list) or not parsed:
        return False
    rows = {int(row["truncation_rank"]): row for row in parsed if isinstance(row, dict) and "truncation_rank" in row}
    if not rows:
        return False
    dimension = int(factor_dimension or max(rows))
    required = {rank for rank in (1, 2, 4, 8, 16, 32) if rank <= dimension}
    required.add(dimension)
    if not required.issubset(rows):
        return False
    for rank in required:
        row = rows[rank]
        required_any = (
            ("relative_mixer_frobenius_error", "mixer_relative_frobenius_error"),
            ("q_squared_energy_retained", "q_retained_energy"),
            ("k_squared_energy_retained", "k_retained_energy"),
            ("relative_srank_error", "srank_relative_error"),
            ("relative_erank_1_error", "erank_1_relative_error"),
            ("relative_erank_2_error", "erank_2_relative_error"),
        )
        if any(not any(name in row and row[name] is not None for name in alternatives) for alternatives in required_any):
            return False
    return True


def _validate_structured_geometry(audit: Audit, model: str, native: pd.DataFrame) -> None:
    for column in ("q_rank_metrics", "k_rank_metrics", "score_rank_metrics"):
        _json_contract_audit(
            audit,
            f"WP3 {model} {column}",
            native,
            column,
            lambda value: isinstance(value, dict)
            and all(key in value for key in ("srank", "erank_1", "erank_2", "r90", "r95", "r99", "dimension")),
        )
    for label, candidates, aspect in (
        ("canonical correlations", ("canonical_correlations_by_rank", "geometry_by_rank"), "canonical"),
        ("token alignment", ("token_alignment_by_rank", "token_alignment", "geometry_by_rank"), "token"),
        ("projection overlap and QQ", ("projection_overlap_by_rank", "projection_overlap", "geometry_by_rank"), "projection"),
        ("scale diagnostic", ("scale_diagnostic", "scale_diagnostics"), "scale"),
    ):
        _json_candidate_contract_audit(
            audit,
            f"WP3 {model} {label}",
            native,
            candidates,
            lambda value, selected, selected_aspect=aspect: _structured_geometry_contract(
                value, selected, selected_aspect
            ),
        )
    if "low_rank_reconstruction" not in native:
        audit.add(f"WP3 {model} truncation fidelity", False, "column low_rank_reconstruction is missing")
        return
    invalid = []
    for index, row in native.iterrows():
        q = parse_json(row.get("q_rank_metrics"))
        dimension = int(q["dimension"]) if isinstance(q, dict) and q.get("dimension") else None
        try:
            if not _low_rank_contract(row["low_rank_reconstruction"], dimension):
                invalid.append(int(index))
        except (AnalysisError, TypeError, ValueError):
            invalid.append(int(index))
    audit.add(
        f"WP3 {model} truncation fidelity",
        not invalid,
        f"rows={len(native)}, invalid={len(invalid)}, examples={invalid[:5]}",
    )


def _structured_geometry_contract(value: Any, column: str, aspect: str) -> bool:
    """Validate either the current combined geometry payload or legacy fields."""

    if column != "geometry_by_rank":
        if aspect == "scale" and column == "scale_diagnostics":
            return (
                isinstance(value, list)
                and bool(value)
                and all(
                    isinstance(item, dict)
                    and item.get("q_direction") is not None
                    and item.get("k_direction") is not None
                    and item.get("normalized_product") is not None
                    for item in value
                )
            )
        return isinstance(value, (dict, list)) and bool(value)
    if not isinstance(value, dict) or not value:
        return False
    entries: list[tuple[int, dict[str, Any]]] = []
    try:
        for rank_text, item in value.items():
            rank = int(str(rank_text).lstrip("r"))
            if rank < 1 or not isinstance(item, dict) or int(item.get("rank", rank)) != rank:
                return False
            entries.append((rank, item))
    except (TypeError, ValueError):
        return False
    if aspect == "canonical":
        return all(
            isinstance(item.get("token_canonical_correlations"), list)
            and len(item["token_canonical_correlations"]) == rank
            for rank, item in entries
        )
    if aspect == "token":
        return all(
            item.get("token_alignment_score") is not None
            and item.get("token_procrustes_distance") is not None
            for _, item in entries
        )
    if aspect == "projection":
        required_arrays = (
            "projection_scaled_entries",
            "projection_overlap_singular_values",
            "projection_normal_qq_probabilities",
            "projection_normal_qq_theoretical",
            "projection_normal_qq_empirical",
        )
        return all(
            all(isinstance(item.get(name), list) and bool(item[name]) for name in required_arrays)
            and len(item["projection_scaled_entries"]) == rank * rank
            and len(item["projection_overlap_singular_values"]) == rank
            for rank, item in entries
        )
    return False


def _manifest_documents(results_dir: Path, audit: Audit) -> list[str]:
    path = results_dir / "sample_manifest.jsonl"
    audit.add("sample manifest exists", path.exists(), str(path))
    if not path.exists():
        return []
    rows = read_jsonl(path)
    documents = sorted({str(row.get("document_id")) for row in rows if row.get("document_id")})
    audit.add(
        "sample manifest document count",
        len(documents) == EXPECTED_DOCUMENTS,
        f"expected={EXPECTED_DOCUMENTS}, observed={len(documents)}",
    )
    key = ("checkpoint", "document_id", "input_condition", "length")
    keys = [tuple(row.get(name) for name in key) for row in rows]
    audit.add("sample manifest unique records", len(keys) == len(set(keys)), f"rows={len(keys)}, unique={len(set(keys))}")
    required_fields = {
        "checkpoint",
        "document_id",
        "article_index",
        "sample_offset",
        "source_row_start",
        "source_row_end",
        "text_sha256",
        "input_condition",
        "length",
        "token_ids",
        "decoded_text",
    }
    missing_fields = sorted(required_fields - {name for row in rows for name in row})
    audit.add("sample manifest provenance fields", not missing_fields, f"missing={missing_fields}")
    prefix_failures: list[str] = []
    for group_key, group_rows in _group_records(rows, ("checkpoint", "document_id", "input_condition")).items():
        ordered = sorted(group_rows, key=lambda row: int(row["length"]))
        if not ordered:
            continue
        longest = ordered[-1].get("token_ids")
        if not isinstance(longest, list):
            prefix_failures.append(f"{group_key}: longest token_ids is not a list")
            continue
        offsets = {row.get("sample_offset") for row in ordered}
        articles = {row.get("article_index") for row in ordered}
        for row in ordered:
            tokens = row.get("token_ids")
            length = int(row["length"])
            if not isinstance(tokens, list) or len(tokens) != length or tokens != longest[:length]:
                prefix_failures.append(f"{group_key}: nonnested or wrong-length prefix at ell={length}")
                break
        if len(offsets) != 1 or len(articles) != 1:
            prefix_failures.append(f"{group_key}: offsets={offsets}, articles={articles}")
    audit.add(
        "sample manifest exact nested prefixes",
        not prefix_failures,
        f"groups={len(_group_records(rows, ('checkpoint', 'document_id', 'input_condition')))}, "
        f"failures={len(prefix_failures)}, examples={prefix_failures[:5]}",
    )
    for model in PAPER_MODELS:
        conditions = ("native-prefix", "explicit-boundary") if model == "gpt2" else ("native-prefix",)
        observed = {
            (row.get("document_id"), row.get("input_condition"), row.get("length"))
            for row in rows
            if row.get("checkpoint") == model
        }
        expected = set((doc, condition, length) for doc in documents for condition in conditions for length in PRIMARY_LENGTHS)
        missing = expected - observed
        audit.add(
            f"manifest grid: {model}",
            not missing,
            f"expected={len(expected)}, observed={len(observed)}, missing={len(missing)}",
        )
    return documents


def _group_records(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    return grouped


def validate_reconstruction(results_dir: Path, audit: Audit) -> dict[str, Any]:
    path = results_dir / "reconstruction_checks.json"
    audit.add("WP0 reconstruction artifact", path.exists(), str(path))
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    acceptance = value.get("acceptance", {}) if isinstance(value, dict) else {}
    for model in PAPER_MODELS:
        row = acceptance.get(model, {})
        audit.add(
            f"WP0 passed: {model}",
            bool(row.get("passed")),
            canonical_json(json_safe(row)) if row else "missing acceptance record",
        )
    return value


def validate_checkpoint_manifest(results_dir: Path, audit: Audit) -> None:
    path = results_dir / "checkpoint_manifest.json"
    audit.add("checkpoint manifest exists", path.exists(), str(path))
    if not path.exists():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    serialized = canonical_json(value)
    for model, revision in REVISIONS.items():
        audit.add(f"frozen revision: {model}", revision in serialized, revision)


def validate_run_provenance(results_dir: Path, audit: Audit) -> None:
    for name in ("environment.txt", "git_revision.txt"):
        path = results_dir / name
        present = path.exists() and path.is_file() and bool(path.read_text(encoding="utf-8").strip())
        audit.add(f"run provenance: {name}", present, str(path))


def write_run_provenance(results_dir: Path) -> None:
    """Capture analysis environment and the repository revision/dirty state."""

    import platform
    import subprocess

    import matplotlib
    import pyarrow

    environment = "\n".join(
        (
            f"python={platform.python_version()}",
            f"platform={platform.platform()}",
            f"numpy={np.__version__}",
            f"pandas={pd.__version__}",
            f"pyarrow={pyarrow.__version__}",
            f"matplotlib={matplotlib.__version__}",
        )
    )
    atomic_text(results_dir / "environment.txt", environment + "\n")
    repository = results_dir.parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout
        text = f"git_revision={revision}\nworktree_dirty={bool(dirty.strip())}\n"
        if dirty.strip():
            text += "dirty_paths:\n" + dirty
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError(f"Unable to record git provenance from {repository}: {exc}") from exc
    atomic_text(results_dir / "git_revision.txt", text)


def _validate_spectrum_references(results_dir: Path, metrics: pd.DataFrame, audit: Audit) -> None:
    reference_columns = [
        name
        for name in (
            "spectrum_id",
            "spectrum_ref",
            "q_spectrum_ref",
            "k_spectrum_ref",
            "score_thin_spectrum_ref",
            "sink_residual_spectrum_ref",
        )
        if name in metrics
    ]
    if not reference_columns:
        audit.add("spectrum references", False, "no spectrum reference columns found")
        return
    cache: dict[Path, set[str] | None] = {}
    missing_archive: list[str] = []
    missing_key: list[str] = []
    duplicate_structured_keys: list[str] = []
    structured_scan_failures: list[str] = []
    structured_archives: dict[tuple[str, str], Path] = {}
    for model in ("based", "mamba2", "retnet"):
        archive_root = results_dir / "structured_shards" / model / "spectra"
        for archive_path in sorted(archive_root.glob("*.npz")):
            try:
                with np.load(archive_path, allow_pickle=False) as bundle:
                    keys = set(bundle.files)
                cache[archive_path] = keys
            except Exception as exc:
                cache[archive_path] = None
                structured_scan_failures.append(f"{archive_path}: {exc}")
                continue
            for key in keys:
                identity = (model, key)
                previous = structured_archives.get(identity)
                if previous is not None and previous != archive_path:
                    duplicate_structured_keys.append(f"{model}:{key}")
                else:
                    structured_archives[identity] = archive_path
    checked = 0
    for _, row in metrics.iterrows():
        refs = [(column, row.get(column)) for column in reference_columns]
        refs = [(column, str(value)) for column, value in refs if pd.notna(value) and str(value)]
        if not refs:
            continue
        archive_value = row.get("spectrum_archive")
        if archive_value is None or archive_value is pd.NA or pd.isna(archive_value):
            archive_value = row.get("array_archive")
        archive: Path | None = None
        if archive_value is not None and archive_value is not pd.NA and pd.notna(archive_value) and str(archive_value):
            archive = Path(str(archive_value))
            if not archive.is_absolute():
                archive = results_dir / archive
        if archive is None and row.get("model") in ("based", "mamba2", "retnet"):
            for _column, reference in refs:
                archive = structured_archives.get((str(row.get("model")), reference))
                if archive is not None:
                    break
        if archive is None and row.get("model") in ("based", "mamba2", "retnet"):
            candidate = results_dir / "spectra" / f"{row['model']}.npz"
            if candidate.exists():
                archive = candidate
        if archive is None and row.get("model") == "gpt2":
            candidate = results_dir / "spectra" / "gpt2.npz"
            if candidate.exists():
                archive = candidate
        if archive is None and pd.notna(row.get("source_artifact")):
            for source_text in str(row.get("source_artifact")).split(","):
                source = Path(source_text)
                if not source.is_absolute():
                    source = results_dir / source
                candidates = (
                    source.with_suffix(".npz"),
                    source.parent / "spectra" / f"{source.stem}.npz",
                    source.parent.parent / "spectra" / f"{source.stem}.npz",
                )
                archive = next((candidate for candidate in candidates if candidate.exists()), None)
                if archive is not None:
                    break
        if archive is None or not archive.exists():
            missing_archive.append(str(archive or archive_value))
            continue
        if archive not in cache:
            try:
                with np.load(archive, allow_pickle=False) as bundle:
                    cache[archive] = set(bundle.files)
            except Exception:
                cache[archive] = None
        keys = cache[archive]
        if keys is None:
            missing_archive.append(f"unreadable: {archive}")
            continue
        for column, reference in refs:
            checked += 1
            if reference not in keys:
                missing_key.append(f"{archive}:{column}={reference}")
    audit.add(
        "referenced NPZ spectra",
        not missing_archive
        and not missing_key
        and not duplicate_structured_keys
        and not structured_scan_failures
        and checked > 0,
        f"checked={checked}, missing_archives={len(missing_archive)}, missing_keys={len(missing_key)}, "
        f"duplicate_structured_keys={len(duplicate_structured_keys)}, "
        f"structured_scan_failures={len(structured_scan_failures)}, "
        f"examples={(missing_archive + missing_key + duplicate_structured_keys + structured_scan_failures)[:5]}",
    )


def _validate_retnet_positive_control(metrics: pd.DataFrame, audit: Audit) -> None:
    native = metrics[(metrics["model"] == "retnet") & (metrics["operator_variant"] == "native")].copy()
    required = {"length", "head_or_channel", "mask_R", "mask_C", "mask_kappa"}
    if native.empty or not required.issubset(native.columns):
        audit.add("WP4 RetNet geometric positive control", False, f"missing columns={sorted(required - set(native))}")
        return
    head = pd.to_numeric(native["head_or_channel"], errors="coerce").to_numpy(float)
    length = pd.to_numeric(native["length"], errors="coerce").to_numpy(float)
    gamma = 1.0 - np.power(2.0, -5.0 - head)
    finite_sum = (1.0 - np.power(gamma, length)) / (1.0 - gamma)
    envelope = 1.0 / (1.0 - gamma)
    row_error = np.abs(pd.to_numeric(native["mask_R"], errors="coerce").to_numpy(float) - finite_sum) / finite_sum
    column_error = np.abs(pd.to_numeric(native["mask_C"], errors="coerce").to_numpy(float) - finite_sum) / finite_sum
    gain = pd.to_numeric(native["mask_kappa"], errors="coerce").to_numpy(float)
    passed = (
        np.isfinite(row_error).all()
        and np.isfinite(column_error).all()
        and np.isfinite(gain).all()
        and float(np.max(row_error)) <= 1e-4
        and float(np.max(column_error)) <= 1e-4
        and bool(np.all(gain <= envelope * (1.0 + 1e-5)))
        and bool(np.all(gain <= np.sqrt(pd.to_numeric(native["mask_R"], errors="coerce") * pd.to_numeric(native["mask_C"], errors="coerce")) * (1.0 + 1e-5)))
    )
    audit.add(
        "WP4 RetNet geometric positive control",
        passed,
        f"max_R_error={np.nanmax(row_error):.3g}, max_C_error={np.nanmax(column_error):.3g}, "
        f"max_gain/envelope={np.nanmax(gain / envelope):.6g}",
    )


def validate_metrics_design(results_dir: Path, metrics: pd.DataFrame, documents: Sequence[str], audit: Audit) -> None:
    if metrics.empty or not documents or not all(name in metrics for name in OBSERVATION_KEY):
        return
    rank_columns = (*RANK_METRICS, "operator_norm", "frobenius_norm", "nuclear_norm", "best_rank1_error")
    for model in ("based", "mamba2", "retnet"):
        model_rows = metrics[metrics["model"] == model]
        revisions = set(model_rows.get("checkpoint_revision", pd.Series(dtype=str)).dropna().astype(str))
        audit.add(f"metrics revision: {model}", revisions == {REVISIONS[model]}, f"observed={sorted(revisions)}")
        _grid_audit(
            audit,
            f"full structured grid: {model}",
            model_rows,
            {
                "document_id": documents,
                "input_condition": ("native-prefix",),
                "length": PRIMARY_LENGTHS,
                "layer": LAYERS[model],
                "head_or_channel": HEADS[model],
                "operator_variant": STRUCTURED_VARIANTS[model],
            },
            nonnull=rank_columns,
        )
        native_names = ("native_raw", "native_normalized") if model == "based" else ("native",)
        native = model_rows[model_rows["operator_variant"].isin(native_names)]
        _validate_structured_geometry(audit, model, native)
    gpt = metrics[metrics["model"] == "gpt2"]
    revisions = set(gpt.get("checkpoint_revision", pd.Series(dtype=str)).dropna().astype(str))
    audit.add("metrics revision: gpt2", revisions == {REVISIONS["gpt2"]}, f"observed={sorted(revisions)}")
    _grid_audit(
        audit,
        "GPT-2 all-layer sink grid",
        gpt[gpt["operator_variant"] == "native_softmax"],
        {
            "document_id": documents,
            "input_condition": ("native-prefix", "explicit-boundary"),
            "length": PRIMARY_LENGTHS,
            "layer": ALL_GPT2_LAYERS,
            "head_or_channel": HEADS["gpt2"],
            "operator_variant": ("native_softmax",),
        },
        nonnull=("sink_S", "sink_D", "sink_D_over_S", "sink_rho2", "margin_median", "accessible_oscillation_B"),
    )
    _grid_audit(
        audit,
        "GPT-2 selected-layer full spectra",
        gpt[gpt["operator_variant"] == "native_softmax"],
        {
            "document_id": documents,
            "input_condition": ("native-prefix", "explicit-boundary"),
            "length": PRIMARY_LENGTHS,
            "layer": LAYERS["gpt2"],
            "head_or_channel": HEADS["gpt2"],
            "operator_variant": ("native_softmax",),
        },
        nonnull=rank_columns,
    )
    _grid_audit(
        audit,
        "GPT-2 final-layer counterfactual grid",
        gpt[gpt["operator_variant"].isin(("neutralize_first", "late_spike"))],
        {
            "document_id": documents,
            "input_condition": ("native-prefix", "explicit-boundary"),
            "length": PRIMARY_LENGTHS,
            "layer": (LAYERS["gpt2"][-1],),
            "head_or_channel": GPT2_INTENSIVE_HEADS,
            "operator_variant": ("neutralize_first", "late_spike"),
        },
        nonnull=rank_columns,
    )
    _grid_audit(
        audit,
        "GPT-2 preregistered-head depth heatmap rank grid",
        gpt[gpt["operator_variant"] == "native_softmax"],
        {
            "document_id": documents,
            "input_condition": ("native-prefix", "explicit-boundary"),
            "length": (512,),
            "layer": ALL_GPT2_LAYERS,
            "head_or_channel": GPT2_INTENSIVE_HEADS,
            "operator_variant": ("native_softmax",),
        },
        nonnull=RANK_METRICS,
    )
    native_gpt_selected = gpt[
        (gpt["operator_variant"] == "native_softmax")
        & gpt["layer"].isin(LAYERS["gpt2"])
    ]
    for column in ("q_rank_metrics", "k_rank_metrics", "score_rank_metrics"):
        _json_contract_audit(
            audit,
            f"WP3 GPT-2 {column}",
            native_gpt_selected,
            column,
            lambda value: isinstance(value, dict)
            and all(key in value for key in ("srank", "erank_1", "erank_2", "r90", "r95", "r99")),
        )
    for column in ("token_alignment", "projection_overlap"):
        _json_contract_audit(
            audit,
            f"WP3 GPT-2 {column}",
            native_gpt_selected,
            column,
            lambda value: isinstance(value, list) and bool(value),
        )
    audit.add(
        "WP3 GPT-2 scale diagnostic",
        "qk_scale_products_ref" in native_gpt_selected,
        "qk_scale_products_ref present" if "qk_scale_products_ref" in native_gpt_selected else "missing",
    )
    gpt_trunc = gpt[gpt["operator_variant"].astype(str).str.startswith("softmax_trunc_r", na=False)]
    _grid_audit(
        audit,
        "GPT-2 r16 truncation grid",
        gpt_trunc[gpt_trunc["operator_variant"] == "softmax_trunc_r16"],
        {
            "document_id": documents,
            "input_condition": ("native-prefix", "explicit-boundary"),
            "length": PRIMARY_LENGTHS,
            "layer": LAYERS["gpt2"],
            "head_or_channel": GPT2_INTENSIVE_HEADS,
            "operator_variant": ("softmax_trunc_r16",),
        },
        nonnull=(
            "q_squared_energy_retained",
            "k_squared_energy_retained",
            "mixer_relative_frobenius_error",
            "srank_relative_error",
            "erank_1_relative_error",
            "erank_2_relative_error",
        ),
    )
    for model in ("mamba2", "retnet"):
        subset = metrics[metrics["model"] == model]
        _grid_audit(
            audit,
            f"WP4/WP5 variant-specific mask metrics: {model}",
            subset,
            {
                "document_id": documents,
                "input_condition": ("native-prefix",),
                "length": PRIMARY_LENGTHS,
                "layer": LAYERS[model],
                "head_or_channel": HEADS[model],
                "operator_variant": STRUCTURED_VARIANTS[model],
            },
            nonnull=("mask_R", "mask_C", "mask_kappa"),
        )
    _validate_retnet_positive_control(metrics, audit)
    mamba_native = metrics[(metrics["model"] == "mamba2") & (metrics["operator_variant"] == "native")]
    _json_contract_audit(
        audit,
        "WP4 Mamba block products",
        mamba_native,
        "block_products",
        lambda value: isinstance(value, dict)
        and any(str(block) in value for block in (4, 8, 16, 32, 64)),
    )
    _json_contract_audit(
        audit,
        "WP4 Mamba effective half-life",
        mamba_native,
        "effective_half_life_summary",
        lambda value: isinstance(value, dict) and bool(value),
    )
    _validate_spectrum_references(results_dir, metrics, audit)


class BootstrapEngine:
    """Deterministic document bootstrap with one shared draw matrix per n."""

    def __init__(self, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BASE_SEED):
        if resamples < 1:
            raise ValueError("resamples must be positive")
        self.resamples = int(resamples)
        self.seed = int(seed)
        self._cache: dict[int, np.ndarray] = {}

    def indices(self, n_documents: int) -> np.ndarray:
        if n_documents < 1:
            raise AnalysisError("Cannot bootstrap zero documents")
        if n_documents not in self._cache:
            rng = np.random.default_rng(self.seed)
            self._cache[n_documents] = rng.integers(
                0, n_documents, size=(self.resamples, n_documents), endpoint=False
            )
        return self._cache[n_documents]

    def median(self, values: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise AnalysisError("Bootstrap median requires a nonempty finite vector")
        replicates = np.median(values[self.indices(values.size)], axis=1)
        return (
            float(np.median(values)),
            float(np.quantile(replicates, 0.025)),
            float(np.quantile(replicates, 0.975)),
            replicates,
        )

    def slope(self, matrix: np.ndarray, lengths: Sequence[int]) -> tuple[float, float, float, np.ndarray]:
        matrix = np.asarray(matrix, dtype=np.float64)
        lengths_array = np.asarray(lengths, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != lengths_array.size or matrix.shape[0] < 1:
            raise AnalysisError("Slope bootstrap expects documents x lengths")
        if lengths_array.size < 2 or np.any(matrix <= 0.0) or not np.isfinite(matrix).all():
            raise AnalysisError("Log--log slopes require at least two positive finite lengths and values")
        log_x = np.log(lengths_array)

        def fit(curves: np.ndarray) -> np.ndarray:
            centered_x = log_x - log_x.mean()
            denominator = float(np.square(centered_x).sum())
            centered_y = np.log(curves) - np.log(curves).mean(axis=-1, keepdims=True)
            return (centered_y * centered_x).sum(axis=-1) / denominator

        point_curve = np.median(matrix, axis=0, keepdims=True)
        point = float(fit(point_curve)[0])
        sampled = matrix[self.indices(matrix.shape[0])]
        curves = np.median(sampled, axis=1)
        replicates = fit(curves)
        return point, float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975)), replicates


def _doc_head_medians(frame: pd.DataFrame, metric: str, group_columns: Sequence[str]) -> pd.DataFrame:
    required = [*group_columns, "document_id", "head_or_channel", metric]
    missing = [name for name in required if name not in frame]
    if missing:
        raise AnalysisError(f"Cannot aggregate {metric}; missing columns {missing}")
    selected = frame[required].copy()
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected[np.isfinite(selected[metric])]
    if selected.empty:
        return pd.DataFrame(columns=[*group_columns, "document_id", metric, "n_heads"])
    groups = [*group_columns, "document_id"]
    result = selected.groupby(groups, observed=True, dropna=False).agg(
        **{metric: (metric, "median")}, n_heads=("head_or_channel", "nunique")
    ).reset_index()
    return result


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    model: str
    variant: str
    input_condition: str = "native-prefix"
    checkpoint: str | None = None


PRIMARY_SERIES = (
    SeriesSpec("GPT-2", "gpt2", "native_softmax"),
    SeriesSpec("Based raw", "based", "native_raw"),
    SeriesSpec("Based normalized", "based", "native_normalized"),
    SeriesSpec("Mamba-2", "mamba2", "native"),
    SeriesSpec("RetNet", "retnet", "native"),
)


def select_series(metrics: pd.DataFrame, spec: SeriesSpec) -> pd.DataFrame:
    selected = metrics[
        (metrics["model"] == spec.model)
        & (metrics["operator_variant"] == spec.variant)
        & (metrics["input_condition"] == spec.input_condition)
    ]
    if spec.checkpoint is not None:
        selected = selected[selected["checkpoint"] == spec.checkpoint]
    return selected


def curve_and_slope_tables(
    metrics: pd.DataFrame,
    bootstrap: BootstrapEngine,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    per_document_rows: list[dict[str, Any]] = []
    for spec in PRIMARY_SERIES:
        selected = select_series(metrics, spec)
        for layer in LAYERS[spec.model]:
            layer_rows = selected[selected["layer"] == layer]
            for metric in RANK_METRICS:
                doc = _doc_head_medians(layer_rows, metric, ("length",))
                if doc.empty:
                    continue
                for length, group in doc.groupby("length", observed=True):
                    point, low, high, _ = bootstrap.median(group[metric].to_numpy(float))
                    curve_rows.append(
                        {
                            "series": spec.name,
                            "model": spec.model,
                            "checkpoint": spec.checkpoint or spec.model,
                            "input_condition": spec.input_condition,
                            "operator_variant": spec.variant,
                            "layer": int(layer),
                            "layer_role": ("early", "middle", "final")[LAYERS[spec.model].index(layer)],
                            "length": int(length),
                            "metric": metric,
                            "estimate": point,
                            "ci_low": low,
                            "ci_high": high,
                            "n_documents": int(group["document_id"].nunique()),
                            "n_heads_min": int(group["n_heads"].min()),
                            "n_heads_max": int(group["n_heads"].max()),
                            "n_resamples": bootstrap.resamples,
                            "seed": bootstrap.seed,
                        }
                    )
                slope_doc = doc[doc["length"] >= SLOPE_MIN_LENGTH]
                lengths = sorted(int(value) for value in slope_doc["length"].unique())
                pivot = slope_doc.pivot(index="document_id", columns="length", values=metric)
                if len(lengths) < 2 or pivot.empty or pivot[lengths].isna().any().any():
                    raise AnalysisError(f"Incomplete slope grid for {spec.name}, layer {layer}, {metric}")
                point, low, high, replicates = bootstrap.slope(pivot[lengths].to_numpy(float), lengths)
                slope_rows.append(
                    {
                        "series": spec.name,
                        "model": spec.model,
                        "checkpoint": spec.checkpoint or spec.model,
                        "input_condition": spec.input_condition,
                        "operator_variant": spec.variant,
                        "layer": int(layer),
                        "layer_role": ("early", "middle", "final")[LAYERS[spec.model].index(layer)],
                        "metric": metric,
                        "slope": point,
                        "ci_low": low,
                        "ci_high": high,
                        "n_documents": int(pivot.shape[0]),
                        "n_resamples": bootstrap.resamples,
                        "seed": bootstrap.seed,
                        "min_length": int(min(lengths)),
                        "max_length": int(max(lengths)),
                    }
                )
                replicate_rows.extend(
                    {
                        "series": spec.name,
                        "layer": int(layer),
                        "metric": metric,
                        "replicate": index,
                        "slope": float(value),
                    }
                    for index, value in enumerate(replicates)
                )
                log_x = np.log(np.asarray(lengths, dtype=float))
                for document_id, values in pivot[lengths].iterrows():
                    per_document_rows.append(
                        {
                            "series": spec.name,
                            "model": spec.model,
                            "layer": int(layer),
                            "layer_role": ("early", "middle", "final")[LAYERS[spec.model].index(layer)],
                            "metric": metric,
                            "document_id": document_id,
                            "slope": float(np.polyfit(log_x, np.log(values.to_numpy(float)), 1)[0]),
                        }
                    )
    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(slope_rows),
        pd.DataFrame(replicate_rows),
        pd.DataFrame(per_document_rows),
    )


def paired_effect_table(metrics: pd.DataFrame, bootstrap: BootstrapEngine) -> pd.DataFrame:
    comparisons: list[tuple[str, str, str]] = [
        ("gpt2", "native_softmax", "neutralize_first"),
        ("gpt2", "native_softmax", "late_spike"),
        ("gpt2", "native_softmax", "causal_raw_all_one"),
        ("gpt2", "native_softmax", "causal_raw_decay_0.5"),
        ("gpt2", "native_softmax", "causal_raw_decay_0.9"),
        ("gpt2", "native_softmax", "causal_raw_decay_0.99"),
        ("based", "native_raw", "native_normalized"),
        ("based", "native_raw", "identity"),
        ("mamba2", "native", "all_one"),
        ("mamba2", "native", "identity"),
        ("mamba2", "native", "constant_decay"),
        ("mamba2", "native", "transition_shuffled"),
        ("retnet", "native", "all_one"),
        ("retnet", "native", "identity"),
        ("retnet", "native", "decay_0.5"),
        ("retnet", "native", "decay_0.9"),
        ("retnet", "native", "decay_0.99"),
        ("retnet", "native", "causal_softmax"),
    ]
    keys = ["model", "checkpoint", "document_id", "input_condition", "length", "layer", "head_or_channel"]
    rows: list[dict[str, Any]] = []
    for model, reference, intervention in comparisons:
        subset = metrics[metrics["model"] == model]
        left = subset[subset["operator_variant"] == reference]
        right = subset[subset["operator_variant"] == intervention]
        if left.empty or right.empty:
            continue
        for metric in RANK_METRICS:
            available_keys = [key for key in keys if key in subset]
            merged = left[available_keys + [metric]].merge(
                right[available_keys + [metric]], on=available_keys, how="inner", suffixes=("_reference", "_intervention"), validate="one_to_one"
            )
            merged["difference"] = pd.to_numeric(merged[f"{metric}_intervention"], errors="coerce") - pd.to_numeric(
                merged[f"{metric}_reference"], errors="coerce"
            )
            merged = merged[np.isfinite(merged["difference"])]
            for group_values, group in merged.groupby(
                ["input_condition", "length", "layer"], observed=True, dropna=False
            ):
                condition, length, layer = group_values
                doc = group.groupby("document_id", observed=True).agg(
                    difference=("difference", "median"), n_heads=("head_or_channel", "nunique")
                )
                point, low, high, _ = bootstrap.median(doc["difference"].to_numpy(float))
                rows.append(
                    {
                        "model": model,
                        "reference_variant": reference,
                        "intervention_variant": intervention,
                        "input_condition": condition,
                        "length": int(length),
                        "layer": int(layer),
                        "metric": metric,
                        "effect_intervention_minus_reference": point,
                        "ci_low": low,
                        "ci_high": high,
                        "n_documents": int(doc.shape[0]),
                        "n_heads_min": int(doc["n_heads"].min()),
                        "n_heads_max": int(doc["n_heads"].max()),
                        "n_resamples": bootstrap.resamples,
                        "seed": bootstrap.seed,
                    }
                )
    return pd.DataFrame(rows)


def gpt2_sink_paired_table(metrics: pd.DataFrame, bootstrap: BootstrapEngine) -> pd.DataFrame:
    """Paired depth and boundary-condition effects for the confirmatory sink."""

    gpt = metrics[
        (metrics["model"] == "gpt2")
        & (metrics["operator_variant"] == "native_softmax")
        & metrics["sink_D_over_S"].notna()
    ].copy()
    gpt["log_sink_D_over_S"] = np.log(
        pd.to_numeric(gpt["sink_D_over_S"], errors="coerce").clip(lower=1e-15)
    )
    rows: list[dict[str, Any]] = []

    early = gpt[gpt["layer"] == LAYERS["gpt2"][0]]
    final = gpt[gpt["layer"] == LAYERS["gpt2"][-1]]
    depth_keys = ["document_id", "input_condition", "length", "head_or_channel"]
    depth = early[depth_keys + ["log_sink_D_over_S"]].merge(
        final[depth_keys + ["log_sink_D_over_S"]],
        on=depth_keys,
        how="inner",
        suffixes=("_early", "_final"),
        validate="one_to_one",
    )
    depth["effect"] = depth["log_sink_D_over_S_final"] - depth["log_sink_D_over_S_early"]
    for (condition, length), group in depth.groupby(["input_condition", "length"], observed=True):
        doc = group.groupby("document_id", observed=True).agg(
            effect=("effect", "median"), n_heads=("head_or_channel", "nunique")
        )
        point, low, high, _ = bootstrap.median(doc["effect"].to_numpy(float))
        rows.append(
            {
                "effect_type": "depth_final_minus_early_log_D_over_S",
                "input_condition": condition,
                "length": int(length),
                "layer": pd.NA,
                "estimate": point,
                "ci_low": low,
                "ci_high": high,
                "interpretation": "negative means D/S falls with depth in the same heads",
                "n_documents": int(doc.shape[0]),
                "n_heads_min": int(doc["n_heads"].min()),
                "n_heads_max": int(doc["n_heads"].max()),
                "n_resamples": bootstrap.resamples,
                "seed": bootstrap.seed,
            }
        )

    native = gpt[gpt["input_condition"] == "native-prefix"]
    boundary = gpt[gpt["input_condition"] == "explicit-boundary"]
    condition_keys = ["document_id", "length", "layer", "head_or_channel"]
    condition = native[condition_keys + ["log_sink_D_over_S"]].merge(
        boundary[condition_keys + ["log_sink_D_over_S"]],
        on=condition_keys,
        how="inner",
        suffixes=("_native", "_boundary"),
        validate="one_to_one",
    )
    condition["effect"] = condition["log_sink_D_over_S_boundary"] - condition["log_sink_D_over_S_native"]
    for (layer, length), group in condition.groupby(["layer", "length"], observed=True):
        doc = group.groupby("document_id", observed=True).agg(
            effect=("effect", "median"), n_heads=("head_or_channel", "nunique")
        )
        point, low, high, _ = bootstrap.median(doc["effect"].to_numpy(float))
        rows.append(
            {
                "effect_type": "boundary_minus_native_log_D_over_S",
                "input_condition": "paired explicit-boundary minus native-prefix",
                "length": int(length),
                "layer": int(layer),
                "estimate": point,
                "ci_low": low,
                "ci_high": high,
                "interpretation": (
                    "CI excluding zero indicates boundary-token identity modulates the first-position effect; "
                    "CI containing zero supports a generic first-position interpretation"
                ),
                "n_documents": int(doc.shape[0]),
                "n_heads_min": int(doc["n_heads"].min()),
                "n_heads_max": int(doc["n_heads"].max()),
                "n_resamples": bootstrap.resamples,
                "seed": bootstrap.seed,
            }
        )
    return pd.DataFrame(rows)


def _rank_geometry(value: Any) -> dict[str, Any] | None:
    parsed = parse_json(value)
    return parsed if isinstance(parsed, dict) else None


def _factor_dimension(row: Mapping[str, Any], *metrics: Mapping[str, Any]) -> int:
    dimensions = [int(value["dimension"]) for value in metrics if value.get("dimension") is not None]
    if dimensions:
        return min(dimensions)
    head_dim = json_safe(row.get("head_dim"))
    if head_dim is None:
        raise AnalysisError("Factor-rank metrics omit dimension and the observation has no head_dim fallback")
    return int(min(int(row["length"]), int(head_dim)))


def _alignment_at_rank(row: pd.Series, target: int = 16) -> tuple[float, int] | tuple[float, None]:
    for column in ("token_alignment_by_rank", "token_alignment", "geometry_by_rank"):
        if column not in row or pd.isna(row[column]):
            continue
        parsed = parse_json(row[column])
        if isinstance(parsed, dict):
            entries = []
            for rank, value in parsed.items():
                if isinstance(value, dict):
                    value = value.get(
                        "alignment_score",
                        value.get("token_alignment_score", value.get("token_alignment")),
                    )
                if value is not None:
                    entries.append((int(str(rank).lstrip("r")), float(value)))
        elif isinstance(parsed, list):
            entries = [
                (int(item.get("rank", item.get("alignment_rank"))), float(item.get("alignment_score", item.get("token_alignment"))))
                for item in parsed
                if isinstance(item, dict)
                and item.get("rank", item.get("alignment_rank")) is not None
                and item.get("alignment_score", item.get("token_alignment")) is not None
            ]
        else:
            entries = []
        eligible = [(rank, value) for rank, value in entries if rank <= target]
        if eligible:
            rank, value = max(eligible)
            return value, rank
    if "token_alignment" in row and pd.notna(row["token_alignment"]):
        try:
            rank = int(row.get("alignment_rank", target))
            return float(row["token_alignment"]), rank
        except (TypeError, ValueError):
            pass
    return math.nan, None


def geometry_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in PAPER_MODELS:
        variants = ("native_softmax",) if model == "gpt2" else (("native_raw",) if model == "based" else ("native",))
        subset = metrics[
            (metrics["model"] == model)
            & metrics["operator_variant"].isin(variants)
            & (metrics["input_condition"] == "native-prefix")
            & metrics["layer"].isin(LAYERS[model])
        ]
        for _, row in subset.iterrows():
            q = _rank_geometry(row.get("q_rank_metrics"))
            k = _rank_geometry(row.get("k_rank_metrics"))
            if q is None or k is None:
                continue
            dimension = _factor_dimension(row, q, k)
            alignment, alignment_rank = _alignment_at_rank(row)
            rows.append(
                {
                    "model": model,
                    "document_id": row["document_id"],
                    "input_condition": row["input_condition"],
                    "length": int(row["length"]),
                    "layer": int(row["layer"]),
                    "head_or_channel": int(row["head_or_channel"]),
                    "dimension": dimension,
                    "q_r95_fraction": float(q.get("r95_fraction", q["r95"] / dimension)),
                    "k_r95_fraction": float(k.get("r95_fraction", k["r95"] / dimension)),
                    "max_r95_fraction": max(
                        float(q.get("r95_fraction", q["r95"] / dimension)),
                        float(k.get("r95_fraction", k["r95"] / dimension)),
                    ),
                    "token_alignment": float(alignment),
                    "alignment_rank": alignment_rank,
                }
            )
    geometry = pd.DataFrame(rows)
    trunc_rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        model = row.get("model")
        if model not in ("gpt2", "based", "mamba2", "retnet"):
            continue
        variant = str(row.get("operator_variant"))
        common = {
            "model": model,
            "document_id": row.get("document_id"),
            "input_condition": row.get("input_condition"),
            "length": int(row["length"]),
            "layer": int(row["layer"]),
            "head_or_channel": int(row["head_or_channel"]),
        }
        if model == "gpt2" and variant.startswith("softmax_trunc_r"):
            trunc_rows.append(
                {
                    **common,
                    "truncation_rank": int(row.get("truncation_rank", variant.rsplit("r", 1)[-1])),
                    "q_energy_retained": float(row.get("q_squared_energy_retained", math.nan)),
                    "k_energy_retained": float(row.get("k_squared_energy_retained", math.nan)),
                    "mixer_error": float(row.get("mixer_relative_frobenius_error", math.nan)),
                    "srank_error": float(row.get("srank_relative_error", math.nan)),
                    "erank_1_error": float(row.get("erank_1_relative_error", math.nan)),
                    "erank_2_error": float(row.get("erank_2_relative_error", math.nan)),
                }
            )
        elif (model == "based" and variant == "native_raw") or (
            model in ("mamba2", "retnet") and variant == "native"
        ):
            parsed = parse_json(row.get("low_rank_reconstruction"))
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                if not isinstance(item, dict) or "truncation_rank" not in item:
                    continue
                trunc_rows.append(
                    {
                        **common,
                        "truncation_rank": int(item["truncation_rank"]),
                        "q_energy_retained": float(item.get("q_squared_energy_retained", item.get("q_retained_energy", math.nan))),
                        "k_energy_retained": float(item.get("k_squared_energy_retained", item.get("k_retained_energy", math.nan))),
                        "mixer_error": float(item.get("relative_mixer_frobenius_error", item.get("mixer_relative_frobenius_error", math.nan))),
                        "srank_error": float(item.get("relative_srank_error", item.get("srank_relative_error", math.nan))),
                        "erank_1_error": float(item.get("relative_erank_1_error", item.get("erank_1_relative_error", math.nan))),
                        "erank_2_error": float(item.get("relative_erank_2_error", item.get("erank_2_relative_error", math.nan))),
                    }
                )
    trunc = pd.DataFrame(trunc_rows)
    if geometry.empty:
        return trunc
    if trunc.empty:
        for column in (
            "truncation_rank",
            "q_energy_retained",
            "k_energy_retained",
            "mixer_error",
            "srank_error",
            "erank_1_error",
            "erank_2_error",
        ):
            geometry[column] = np.nan
        return geometry
    return geometry.merge(
        trunc,
        on=["model", "document_id", "input_condition", "length", "layer", "head_or_channel"],
        how="outer",
        validate="one_to_many",
    )


def memory_table(metrics: pd.DataFrame) -> pd.DataFrame:
    subset = metrics[(metrics["model"].isin(("mamba2", "retnet")))].copy()
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        base = {
            "model": row["model"],
            "document_id": row["document_id"],
            "length": int(row["length"]),
            "layer": int(row["layer"]),
            "head_or_channel": int(row["head_or_channel"]),
            "operator_variant": row["operator_variant"],
            "mask_R": float(row.get("mask_R", math.nan)),
            "mask_C": float(row.get("mask_C", math.nan)),
            "mask_kappa": float(row.get("mask_kappa", math.nan)),
            "srank": float(row.get("srank", math.nan)),
            "erank_1": float(row.get("erank_1", math.nan)),
            "erank_2": float(row.get("erank_2", math.nan)),
        }
        products = parse_json(row.get("block_products"))
        if isinstance(products, dict):
            for block, summary in products.items():
                if not isinstance(summary, dict):
                    continue
                rows.append(
                    {
                        **base,
                        "block_length": int(block),
                        "block_max": float(summary.get("max", math.nan)),
                        "block_q95": float(summary.get("q95", math.nan)),
                        "block_median": float(summary.get("median", math.nan)),
                    }
                )
        else:
            rows.append({**base, "block_length": pd.NA, "block_max": math.nan, "block_q95": math.nan, "block_median": math.nan})
    return pd.DataFrame(rows)


def memory_correlations(memory: pd.DataFrame, bootstrap: BootstrapEngine) -> pd.DataFrame:
    """Document-bootstrap Spearman correlations for WP4 memory diagnostics."""

    native = memory[memory["operator_variant"] == "native"].drop_duplicates(
        ["model", "document_id", "length", "layer", "head_or_channel"]
    )
    rows: list[dict[str, Any]] = []
    for (model, layer, length), group in native.groupby(["model", "layer", "length"], observed=True):
        documents = sorted(group["document_id"].unique())
        document_frames = [group[group["document_id"] == document] for document in documents]
        indices = bootstrap.indices(len(documents))
        for memory_metric in ("mask_R", "mask_C", "mask_kappa"):
            for rank_metric in RANK_METRICS:
                clean = group[[memory_metric, rank_metric]].replace([np.inf, -np.inf], np.nan).dropna()
                point = spearman(clean[memory_metric].to_numpy(float), clean[rank_metric].to_numpy(float))
                replicates = []
                for sample in indices:
                    sampled = pd.concat([document_frames[index] for index in sample], ignore_index=True)
                    sampled = sampled[[memory_metric, rank_metric]].replace([np.inf, -np.inf], np.nan).dropna()
                    replicates.append(
                        spearman(sampled[memory_metric].to_numpy(float), sampled[rank_metric].to_numpy(float))
                    )
                finite = np.asarray(replicates, dtype=float)
                finite = finite[np.isfinite(finite)]
                rows.append(
                    {
                        "model": model,
                        "layer": int(layer),
                        "length": int(length),
                        "memory_metric": memory_metric,
                        "rank_metric": rank_metric,
                        "spearman": point,
                        "ci_low": float(np.quantile(finite, 0.025)) if finite.size else math.nan,
                        "ci_high": float(np.quantile(finite, 0.975)) if finite.size else math.nan,
                        "n_documents": len(documents),
                        "n_observations": int(clean.shape[0]),
                        "n_resamples": bootstrap.resamples,
                        "seed": bootstrap.seed,
                    }
                )
    return pd.DataFrame(rows)


def _rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def spearman(values_x: np.ndarray, values_y: np.ndarray) -> float:
    x = _rankdata(np.asarray(values_x, dtype=float))
    y = _rankdata(np.asarray(values_y, dtype=float))
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def sink_correlations(metrics: pd.DataFrame, bootstrap: BootstrapEngine) -> pd.DataFrame:
    gpt = metrics[
        (metrics["model"] == "gpt2")
        & (metrics["operator_variant"] == "native_softmax")
        & metrics["srank"].notna()
    ].copy()
    gpt["log_D_over_S"] = np.log(pd.to_numeric(gpt["sink_D_over_S"], errors="coerce").clip(lower=1e-15))
    gpt["log_srank_minus_1"] = np.log((pd.to_numeric(gpt["srank"], errors="coerce") - 1.0).clip(lower=1e-15))
    gpt["rho2_log_length"] = pd.to_numeric(gpt["sink_rho2"], errors="coerce") * np.log(pd.to_numeric(gpt["length"], errors="coerce"))
    pairs = (
        ("log(D/S)", "log(srank-1)", "log_D_over_S", "log_srank_minus_1"),
        ("rho2", "best rank-one error", "sink_rho2", "best_rank1_error"),
        ("rho2 log(length)", "erank_2", "rho2_log_length", "erank_2"),
        ("first-position v1 mass", "srank", "score_v1_first_mass", "srank"),
        ("first-position v1 mass", "erank_2", "score_v1_first_mass", "erank_2"),
    )
    rows: list[dict[str, Any]] = []
    for (condition, length), group in gpt.groupby(["input_condition", "length"], observed=True):
        documents = sorted(group["document_id"].unique())
        indices = bootstrap.indices(len(documents))
        document_frames = [group[group["document_id"] == document] for document in documents]
        for x_label, y_label, x_name, y_name in pairs:
            if x_name not in group or y_name not in group:
                continue
            clean = group[["document_id", x_name, y_name]].replace([np.inf, -np.inf], np.nan).dropna()
            point = spearman(clean[x_name].to_numpy(float), clean[y_name].to_numpy(float))
            replicates = []
            for sample in indices:
                sampled = pd.concat([document_frames[index] for index in sample], ignore_index=True)
                sampled = sampled[[x_name, y_name]].replace([np.inf, -np.inf], np.nan).dropna()
                replicates.append(spearman(sampled[x_name].to_numpy(float), sampled[y_name].to_numpy(float)))
            finite = np.asarray(replicates, dtype=float)
            finite = finite[np.isfinite(finite)]
            rows.append(
                {
                    "input_condition": condition,
                    "length": int(length),
                    "x": x_label,
                    "y": y_label,
                    "spearman": point,
                    "ci_low": float(np.quantile(finite, 0.025)) if finite.size else math.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if finite.size else math.nan,
                    "n_documents": len(documents),
                    "n_observations": int(clean.shape[0]),
                    "n_resamples": bootstrap.resamples,
                    "seed": bootstrap.seed,
                }
            )
    return pd.DataFrame(rows)


def canonical_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in PRIMARY_REQUIRED_COLUMNS:
        if column not in result:
            result[column] = pd.NA if column in JSON_COLUMNS or column in ("checkpoint_revision", "corpus", "document_id", "input_condition", "operator_variant") else np.nan
    if "observation_id" not in result:
        raise AnalysisError("observation_id was not established because observation keys were invalid")
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].map(
            lambda value: canonical_json(json_safe(value)) if isinstance(value, (dict, list, tuple, np.ndarray)) else value
        )
    front = ["observation_id", "model", *PRIMARY_REQUIRED_COLUMNS]
    ordered = list(dict.fromkeys([name for name in front if name in result] + sorted(result.columns)))
    return result[ordered].sort_values(list(OBSERVATION_KEY), kind="stable").reset_index(drop=True)


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


COLORS = {
    "early": "#0072B2",
    "middle": "#009E73",
    "final": "#D55E00",
    "native_softmax": "#202020",
    "neutralize_first": "#0072B2",
    "late_spike": "#D55E00",
    "gpt2": "#0072B2",
    "based": "#009E73",
    "mamba2": "#D55E00",
    "retnet": "#CC79A7",
}


def save_figure(fig: Any, figures: Path, stem: str, caption: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / f"{stem}.svg", bbox_inches="tight")
    atomic_text(figures / f"{stem}_caption.txt", caption.rstrip() + "\n")


def plot_rank_scaling(curves: pd.DataFrame, figures: Path) -> None:
    plt = _setup_matplotlib()
    series = [spec.name for spec in PRIMARY_SERIES]
    fig, axes = plt.subplots(len(series), 3, figsize=(7.1, 8.0), sharex=True, squeeze=False)
    for row_index, name in enumerate(series):
        for column_index, metric in enumerate(RANK_METRICS):
            ax = axes[row_index, column_index]
            subset = curves[(curves["series"] == name) & (curves["metric"] == metric)]
            for role in ("early", "middle", "final"):
                line = subset[subset["layer_role"] == role].sort_values("length")
                if line.empty:
                    continue
                x = line["length"].to_numpy(float)
                ax.plot(x, line["estimate"], marker="o", ms=2.7, lw=1.2, color=COLORS[role], label=role)
                ax.fill_between(x, line["ci_low"], line["ci_high"], color=COLORS[role], alpha=0.16, linewidth=0)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(alpha=0.22, which="both")
            if row_index == 0:
                ax.set_title(RANK_LABELS[metric])
            if column_index == 0:
                ax.set_ylabel(name)
            if row_index == len(series) - 1:
                ax.set_xlabel(r"length $\ell$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(
        fig,
        figures,
        "figure1_native_rank_scaling",
        "Native-operator rank scaling. Heads are reduced by a median within each document; lines are medians over 32 documents and bands are 95% document-bootstrap intervals (2,000 resamples, seed 20270802).",
    )
    plt.close(fig)


def plot_gpt_heatmaps(metrics: pd.DataFrame, figures: Path) -> None:
    plt = _setup_matplotlib()
    gpt = metrics[
        (metrics["model"] == "gpt2")
        & (metrics["operator_variant"] == "native_softmax")
        & (metrics["input_condition"] == "native-prefix")
        & (metrics["length"] >= SLOPE_MIN_LENGTH)
        & (metrics["length"] <= 512)
    ]
    columns = ("sink_D_over_S", "srank", "erank_2")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), sharex=True, sharey=True, squeeze=False)
    for column_index, metric in enumerate(columns):
        slopes = np.full((len(ALL_GPT2_LAYERS), len(GPT2_INTENSIVE_HEADS)), np.nan)
        head_index = {head: index for index, head in enumerate(GPT2_INTENSIVE_HEADS)}
        for (layer, head), group in gpt.groupby(["layer", "head_or_channel"], observed=True):
            if int(head) not in head_index:
                continue
            # Documents are independent, so form one document median per length
            # before fitting a component-resolved descriptive slope.
            document_length = group.groupby(["document_id", "length"], observed=True)[metric].median().reset_index()
            curve = document_length.groupby("length", observed=True)[metric].median().sort_index()
            if len(curve) < 2 or (curve <= 0).any():
                continue
            slopes[int(layer), head_index[int(head)]] = float(
                np.polyfit(np.log(curve.index.to_numpy(float)), np.log(curve.to_numpy(float)), 1)[0]
            )
        if np.isnan(slopes).any():
            raise AnalysisError(f"Incomplete GPT slope heatmap for {metric}")
        limit = float(np.nanmax(np.abs(slopes)))
        ax = axes[0, column_index]
        image = ax.imshow(
            slopes,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-max(limit, 1e-12),
            vmax=max(limit, 1e-12),
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="log--log slope")
        ax.set_title(
            {"sink_D_over_S": r"sink $D/S$", "srank": "stable rank", "erank_2": RANK_LABELS["erank_2"]}[metric]
        )
        if column_index == 0:
            ax.set_ylabel("layer")
        ax.set_xlabel("head")
        ax.set_xticks(range(len(GPT2_INTENSIVE_HEADS)), GPT2_INTENSIVE_HEADS)
        ax.set_yticks(range(0, 12, 2), range(0, 12, 2))
    fig.tight_layout()
    save_figure(
        fig,
        figures,
        "figure2_gpt2_depth_head_sink",
        "GPT-2 native-prefix depth-by-head fitted log--log slopes over lengths 256--512 for preregistered heads 0, 3, 6, and 9. Each component curve first takes a median across documents at each length; D/S is the exact first-column residual-to-sink energy ratio.",
    )
    plt.close(fig)


def _curve_for_frame(frame: pd.DataFrame, metric: str, bootstrap: BootstrapEngine) -> pd.DataFrame:
    doc = _doc_head_medians(frame, metric, ("operator_variant", "length"))
    rows = []
    for (variant, length), group in doc.groupby(["operator_variant", "length"], observed=True):
        estimate, low, high, _ = bootstrap.median(group[metric].to_numpy(float))
        rows.append({"operator_variant": variant, "length": int(length), "estimate": estimate, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def plot_gpt_interventions(metrics: pd.DataFrame, bootstrap: BootstrapEngine, figures: Path) -> None:
    plt = _setup_matplotlib()
    variants = ("native_softmax", "neutralize_first", "late_spike")
    conditions = ("native-prefix", "explicit-boundary")
    final_layer = LAYERS["gpt2"][-1]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.5), sharex=True, squeeze=False)
    for row_index, condition in enumerate(conditions):
        subset = metrics[
            (metrics["model"] == "gpt2")
            & (metrics["input_condition"] == condition)
            & (metrics["layer"] == final_layer)
            & metrics["operator_variant"].isin(variants)
        ]
        for column_index, metric in enumerate(RANK_METRICS):
            ax = axes[row_index, column_index]
            curve = _curve_for_frame(subset, metric, bootstrap)
            for variant in variants:
                line = curve[curve["operator_variant"] == variant].sort_values("length")
                x = line["length"].to_numpy(float)
                ax.plot(x, line["estimate"], marker="o", ms=2.7, lw=1.2, color=COLORS[variant], label=variant.replace("_", " "))
                ax.fill_between(x, line["ci_low"], line["ci_high"], color=COLORS[variant], alpha=0.14, linewidth=0)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(alpha=0.22, which="both")
            if row_index == 0:
                ax.set_title(RANK_LABELS[metric])
            if column_index == 0:
                ax.set_ylabel(condition.replace("-", " "))
            if row_index == 1:
                ax.set_xlabel(r"length $\ell$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(
        fig,
        figures,
        "figure3_gpt2_sink_interventions",
        "GPT-2 final-layer analytical interventions. The first-column score advantage is either neutralized or moved to a late causal position; these operators are diagnostics, not language-model forward passes. Bands bootstrap documents with heads kept together.",
    )
    plt.close(fig)


def _document_summary(frame: pd.DataFrame, value: str, groups: Sequence[str], bootstrap: BootstrapEngine) -> pd.DataFrame:
    clean = frame[[*groups, "document_id", "head_or_channel", value]].copy()
    clean[value] = pd.to_numeric(clean[value], errors="coerce")
    clean = clean[np.isfinite(clean[value])]
    doc = clean.groupby([*groups, "document_id"], observed=True)[value].median().reset_index()
    rows = []
    for keys, group in doc.groupby(list(groups), observed=True):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        estimate, low, high, _ = bootstrap.median(group[value].to_numpy(float))
        rows.append({**dict(zip(groups, keys)), "estimate": estimate, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def plot_low_rank_dashboard(geometry: pd.DataFrame, bootstrap: BootstrapEngine, figures: Path) -> None:
    plt = _setup_matplotlib()
    models = ("gpt2", "based", "mamba2", "retnet")
    final = geometry[geometry.apply(lambda row: int(row["layer"]) == LAYERS[str(row["model"])][-1], axis=1)]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), squeeze=False)
    values = (
        ("max_r95_fraction", r"$\max(r_{95}(Q),r_{95}(K))/d_h$"),
        ("token_alignment", "token-space alignment"),
    )
    for ax, (column, title) in zip(axes[0, :2], values):
        summary = _document_summary(final.drop_duplicates(["model", "document_id", "length", "layer", "head_or_channel"]), column, ("model", "length"), bootstrap)
        for model in models:
            line = summary[summary["model"] == model].sort_values("length")
            x = line["length"].to_numpy(float)
            ax.plot(x, line["estimate"], marker="o", ms=2.7, lw=1.2, color=COLORS[model], label=model)
            ax.fill_between(x, line["ci_low"], line["ci_high"], color=COLORS[model], alpha=0.14, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_title(title)
        ax.set_xlabel(r"length $\ell$")
        ax.grid(alpha=0.22)
    target = final[final["truncation_rank"] == 16]
    summary = _document_summary(target, "mixer_error", ("model", "length"), bootstrap)
    ax = axes[0, 2]
    for model in models:
        line = summary[summary["model"] == model].sort_values("length")
        x = line["length"].to_numpy(float)
        ax.plot(x, line["estimate"], marker="o", ms=2.7, lw=1.2, color=COLORS[model], label=model)
        ax.fill_between(x, line["ci_low"], line["ci_high"], color=COLORS[model], alpha=0.14, linewidth=0)
    ax.axhline(0.10, color="#666666", ls="--", lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("rank-16 mixer error")
    ax.set_xlabel(r"length $\ell$")
    ax.grid(alpha=0.22, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(
        fig,
        figures,
        "figure4_low_rank_dashboard",
        "Learned-factor diagnostics at the final selected layer: factor energy dimension, token-side Q/K alignment at the largest available rank no greater than 16, and rank-16 operator reconstruction error. Heads are aggregated within document before document bootstrap.",
    )
    plt.close(fig)


def plot_mamba_memory(memory: pd.DataFrame, bootstrap: BootstrapEngine, figures: Path) -> None:
    plt = _setup_matplotlib()
    final_layer = LAYERS["mamba2"][-1]
    native = memory[
        (memory["model"] == "mamba2")
        & (memory["operator_variant"] == "native")
        & (memory["layer"] == final_layer)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7))
    maximum = native[native["length"] == max(PRIMARY_LENGTHS)]
    contraction = _document_summary(maximum, "block_q95", ("block_length",), bootstrap)
    axes[0].plot(contraction["block_length"], contraction["estimate"], marker="o", color=COLORS["mamba2"])
    axes[0].fill_between(contraction["block_length"].to_numpy(float), contraction["ci_low"], contraction["ci_high"], color=COLORS["mamba2"], alpha=0.16)
    axes[0].axhline(1.0, color="#666666", ls="--", lw=0.9)
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("block-product 95th percentile")
    axes[0].set_xlabel("block length")
    axes[0].grid(alpha=0.22)
    deduplicated = native.drop_duplicates(["document_id", "length", "layer", "head_or_channel"])
    for metric, label, color in (
        ("mask_R", r"$R_\ell$", "#0072B2"),
        ("mask_C", r"$C_\ell$", "#009E73"),
        ("mask_kappa", r"$\kappa_\ell$", "#D55E00"),
    ):
        summary = _document_summary(deduplicated, metric, ("length",), bootstrap).sort_values("length")
        x = summary["length"].to_numpy(float)
        axes[1].plot(x, summary["estimate"], marker="o", ms=2.7, color=color, label=label)
        axes[1].fill_between(x, summary["ci_low"], summary["ci_high"], color=color, alpha=0.14)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_title("cumulative memory and gain")
    axes[1].set_xlabel(r"length $\ell$")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.22, which="both")
    scatter = deduplicated[(deduplicated["length"] >= SLOPE_MIN_LENGTH)].copy()
    doc = scatter.groupby(["document_id", "length"], observed=True)[["mask_kappa", "srank"]].median().reset_index()
    colors = np.log2(doc["length"].to_numpy(float))
    image = axes[2].scatter(doc["mask_kappa"], doc["srank"], c=colors, cmap="viridis", s=10, alpha=0.65)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("native rank versus gain")
    axes[2].set_xlabel(r"$\kappa_\ell$")
    axes[2].set_ylabel("stable rank")
    fig.colorbar(image, ax=axes[2], fraction=0.05, pad=0.03, label=r"$\log_2\ell$")
    axes[2].grid(alpha=0.22, which="both")
    fig.tight_layout()
    save_figure(
        fig,
        figures,
        "figure5_mamba_memory_dashboard",
        "Trained Mamba memory diagnostics at the final selected layer: fixed-block contraction, cumulative row/column memory and mask gain, and document-level native stable rank against measured gain.",
    )
    plt.close(fig)


def frame_to_latex(frame: pd.DataFrame, path: Path, *, float_format: str = "%.4g") -> None:
    text = frame.to_latex(index=False, escape=True, float_format=lambda value: float_format % value, na_rep="--")
    atomic_text(path, text)


def reconstruction_table(value: Mapping[str, Any]) -> pd.DataFrame:
    acceptance = value.get("acceptance", {}) if isinstance(value, Mapping) else {}
    checks = value.get("checks", []) if isinstance(value, Mapping) else []
    rows = []
    for model, result in sorted(acceptance.items()):
        model_checks = [row for row in checks if row.get("checkpoint") == model]
        rows.append(
            {
                "checkpoint": model,
                "passed": bool(result.get("passed")),
                "median_relative_error": result.get("median_relative_error"),
                "maximum_relative_error": result.get("maximum_relative_error"),
                "maximum_fp32_reference_error": result.get("maximum_fp32_reference_error"),
                "bf16_tolerance_used": bool(result.get("bf16_tolerance_used")),
                "included_operations": " | ".join(sorted({str(row.get("included", row.get("included_operations", ""))) for row in model_checks})),
                "excluded_operations": " | ".join(sorted({str(row.get("excluded", row.get("excluded_operations", ""))) for row in model_checks})),
            }
        )
    return pd.DataFrame(rows)


def _metric_slope(
    frame: pd.DataFrame,
    metric: str,
    bootstrap: BootstrapEngine,
    *,
    min_length: int = SLOPE_MIN_LENGTH,
) -> tuple[float, float, float]:
    doc = _doc_head_medians(frame, metric, ("length",))
    doc = doc[doc["length"] >= min_length]
    lengths = sorted(doc["length"].unique())
    pivot = doc.pivot(index="document_id", columns="length", values=metric)
    pivot = pivot.dropna(subset=lengths)
    point, low, high, _ = bootstrap.slope(pivot[lengths].to_numpy(float), lengths)
    return point, low, high


def _metric_slope_by_layer(
    frame: pd.DataFrame,
    metric: str,
    bootstrap: BootstrapEngine,
    *,
    min_length: int = SLOPE_MIN_LENGTH,
) -> tuple[float, float, float]:
    """Slope with heads median-reduced inside document and layers kept paired."""

    doc = _doc_head_medians(frame, metric, ("layer", "length"))
    doc = doc[doc["length"] >= min_length]
    lengths = sorted(int(value) for value in doc["length"].unique())
    layers = sorted(int(value) for value in doc["layer"].unique())
    documents = sorted(doc["document_id"].unique())
    if len(lengths) < 2 or not layers or not documents:
        raise AnalysisError(f"Incomplete layer-paired slope grid for {metric}")
    matrices = []
    for layer in layers:
        pivot = doc[doc["layer"] == layer].pivot(index="document_id", columns="length", values=metric)
        pivot = pivot.reindex(index=documents, columns=lengths)
        if pivot.isna().any().any():
            raise AnalysisError(f"Incomplete document/layer/length grid for {metric}, layer {layer}")
        matrices.append(pivot.to_numpy(float))
    combined = np.median(np.stack(matrices, axis=0), axis=0)
    point, low, high, _ = bootstrap.slope(combined, lengths)
    return point, low, high


def decide_claims(
    metrics: pd.DataFrame,
    slopes: pd.DataFrame,
    paired: pd.DataFrame,
    geometry: pd.DataFrame,
    memory: pd.DataFrame,
    bootstrap: BootstrapEngine,
    sink_paired: pd.DataFrame,
) -> pd.DataFrame:
    decisions: list[dict[str, Any]] = []
    final_slopes = slopes[
        slopes["series"].isin(("Mamba-2", "RetNet"))
        & (slopes["layer_role"] == "final")
        & slopes["metric"].isin(RANK_METRICS)
    ]
    growth_pass = len(final_slopes) == 6 and bool((final_slopes["ci_low"] > 0.0).all())
    decisions.append(
        {
            "manuscript_claim": "Mamba and RetNet ranks grow with context",
            "evidence_required": "WP1 final-layer document-bootstrap slope intervals for all three rank proxies",
            "result": "PASS" if growth_pass else "FAIL/QUALIFY",
            "evidence_summary": "; ".join(
                f"{row.series} {row.metric}: {row.slope:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"
                for row in final_slopes.itertuples()
            ),
            "recommended_wording_action": (
                "Keep finite-range growth language and report bootstrap intervals; do not present point slopes as exact exponents."
                if growth_pass
                else "Report heterogeneous finite-range trends and remove unconditional/exact exponent language."
            ),
        }
    )
    gpt_native = metrics[
        (metrics["model"] == "gpt2")
        & (metrics["operator_variant"] == "native_softmax")
        & (metrics["layer"] == LAYERS["gpt2"][-1])
    ]
    sink_length_results = []
    asymptotic_results = []
    for condition in ("native-prefix", "explicit-boundary"):
        group = gpt_native[gpt_native["input_condition"] == condition]
        sink_length_results.append(_metric_slope_by_layer(group, "sink_D_over_S", bootstrap))
        asymptotic_results.append(_metric_slope_by_layer(group, "sink_D_over_S_log_length", bootstrap))
    ratio_doc = _doc_head_medians(gpt_native, "sink_to_best_rank1_error_ratio", ("input_condition", "length"))
    ratio_median = float(ratio_doc["sink_to_best_rank1_error_ratio"].median())
    neutral = paired[
        (paired["model"] == "gpt2")
        & (paired["intervention_variant"] == "neutralize_first")
        & (paired["layer"] == LAYERS["gpt2"][-1])
        & paired["metric"].isin(("srank", "erank_2"))
    ]
    neutral_pass = not neutral.empty and bool((neutral["ci_low"] > 0.0).all())
    depth_effects = sink_paired[
        (sink_paired["effect_type"] == "depth_final_minus_early_log_D_over_S")
        & (sink_paired["length"] >= SLOPE_MIN_LENGTH)
    ]
    depth_pass = not depth_effects.empty and bool((depth_effects["ci_high"] < 0.0).all())
    boundary_effects = sink_paired[
        (sink_paired["effect_type"] == "boundary_minus_native_log_D_over_S")
        & (sink_paired["layer"] == LAYERS["gpt2"][-1])
        & (sink_paired["length"] >= SLOPE_MIN_LENGTH)
    ]
    boundary_interpretation = (
        "token identity modulates the sink"
        if not boundary_effects.empty
        and bool(((boundary_effects["ci_low"] > 0.0) | (boundary_effects["ci_high"] < 0.0)).any())
        else "generic first-position effect is not distinguishable from token identity"
    )
    sink_falls = all(high < 0.0 for _, _, high in sink_length_results)
    sink_close = math.isfinite(ratio_median) and ratio_median <= 1.25
    sink_explains = sink_falls and depth_pass and sink_close and neutral_pass and not boundary_effects.empty
    ranks_available = not gpt_native[["srank", "erank_2"]].isna().any().any()
    sink_result = "EXPLAINS" if sink_explains else ("COMPATIBLE" if ranks_available else "NO SUPPORT")
    asymptotic_decreases = all(high < 0.0 for _, _, high in asymptotic_results)
    decisions.append(
        {
            "manuscript_claim": "Final GPT-2 energy-rank collapse is caused by an early sink",
            "evidence_required": "WP2 D/S trends, sink-vs-optimal rank-one error, paired counterfactual, and both boundary conditions",
            "result": sink_result,
            "evidence_summary": f"D/S slopes={sink_length_results}; paired final-minus-early CIs all negative={depth_pass}; "
            f"median sink/optimal error ratio={ratio_median:.3g}; all neutralization srank/erank2 CIs positive={neutral_pass}; "
            f"boundary contrast={boundary_interpretation}; (D/S)log(ell) decreases={asymptotic_decreases}",
            "recommended_wording_action": (
                "Use 'the sink mechanism explains the observed finite-length GPT-2 collapse'."
                + (" The finite-length asymptotic diagnostic also decreases." if asymptotic_decreases else " Do not claim verification of the erank_2 -> 1 asymptotic condition.")
                if sink_explains
                else "Use only 'the observations are compatible with the sink mechanism'; do not use causal explanation language."
                if ranks_available
                else "Remove the sink interpretation until WP2 is complete."
            ),
        }
    )
    strong_by_model = []
    if not geometry.empty:
        for model, group in geometry.groupby("model", observed=True):
            eligible = group[
                (group["truncation_rank"] <= np.minimum(16, np.floor(group["dimension"] / 4)))
                & (group["q_energy_retained"] >= 0.95)
                & (group["k_energy_retained"] >= 0.95)
            ].copy()
            if eligible.empty:
                strong_by_model.append((model, False, math.nan))
                continue
            eligible["max_proxy_error"] = eligible[["srank_error", "erank_1_error", "erank_2_error"]].max(axis=1)
            doc = eligible.groupby("document_id", observed=True)[["mixer_error", "max_proxy_error"]].median()
            passed = float(doc["mixer_error"].median()) <= 0.10 and float(doc["max_proxy_error"].median()) <= 0.10
            strong_by_model.append((model, passed, float(doc["mixer_error"].median())))
    low_rank_pass = len(strong_by_model) == 4 and all(value[1] for value in strong_by_model)
    decisions.append(
        {
            "manuscript_claim": "Fixed-rank theory models learned Q,K geometry",
            "evidence_required": "WP3 rank <= min(16,d_h/4), >=95% factor energy, and <=10% median mixer/rank-proxy errors",
            "result": "EMPIRICAL PROXY" if low_rank_pass else "STYLIZED ONLY",
            "evidence_summary": "; ".join(f"{model}: pass={passed}, median mixer error={error:.3g}" for model, passed, error in strong_by_model),
            "recommended_wording_action": (
                "Describe the fixed-rank construction as an empirically supported compression proxy, while retaining the finite-rank caveat."
                if low_rank_pass
                else "Retain 'deliberately stylized proxy'; do not say training learned the theorem's low-rank geometry."
            ),
        }
    )
    mamba = memory[(memory["model"] == "mamba2") & (memory["operator_variant"] == "native")]
    memory_slopes = []
    for metric in ("mask_R", "mask_C", "mask_kappa"):
        distinct = mamba.drop_duplicates(["document_id", "length", "layer", "head_or_channel"])
        memory_slopes.append((metric, *_metric_slope_by_layer(distinct, metric, bootstrap)))
    block_pass = False
    block_summary = ""
    if not mamba.empty:
        candidates = []
        for block in sorted(mamba["block_length"].dropna().unique()):
            selected = mamba[mamba["block_length"] == block]
            doc = selected.groupby(["document_id", "length"], observed=True)["block_q95"].median().reset_index()
            medians = doc.groupby("length", observed=True)["block_q95"].median()
            if len(medians) >= 2:
                trend = float(np.polyfit(np.log(medians.index.to_numpy(float)), np.log(medians.to_numpy(float).clip(min=1e-15)), 1)[0])
                candidates.append((int(block), float(medians.max()), trend))
        passing = [item for item in candidates if item[1] < 0.99 and item[2] <= 0.05]
        block_pass = bool(passing)
        block_summary = f"candidates={candidates}; passing={passing}"
    gain_pass = all(high <= 0.10 for _, _, _, high in memory_slopes)
    memory_pass = block_pass and gain_pass
    decisions.append(
        {
            "manuscript_claim": "Trained Mamba is in a bounded-memory regime",
            "evidence_required": "WP4 fixed-block q95 <0.99 without worsening (slope <=0.05) and R/C/kappa slope CI upper <=0.10",
            "result": "CONSISTENT" if memory_pass else "NOT ESTABLISHED",
            "evidence_summary": f"{block_summary}; memory slopes={memory_slopes}",
            "recommended_wording_action": (
                "Say the measured checkpoint is empirically consistent with bounded cumulative memory over the tested range; do not claim a uniform asymptotic proof."
                if memory_pass
                else "Keep the bounded-memory curve as a generic theorem rate only and report measured gain-dependent diagnostics."
            ),
        }
    )
    structured_effects = paired[
        paired["model"].isin(("mamba2", "retnet"))
        & (paired["intervention_variant"] == "all_one")
        & (paired["input_condition"] == "native-prefix")
        & (paired["length"] == max(PRIMARY_LENGTHS))
        & paired["metric"].isin(RANK_METRICS)
    ]
    structured_effects = structured_effects[
        structured_effects.apply(
            lambda row: int(row["layer"]) == LAYERS[str(row["model"])][-1], axis=1
        )
    ]
    # Effects are all-one minus native; the predicted broader native spectrum means negative effects.
    mask_pass = len(structured_effects) == 6 and bool((structured_effects["ci_high"] < 0.0).all())
    decisions.append(
        {
            "manuscript_claim": "Mask structure drives the rank separation",
            "evidence_required": "WP5 paired same-factor all-one minus native effects negative for all three ranks in Mamba and RetNet",
            "result": "MECHANISM SUPPORT" if mask_pass else "ASSOCIATION ONLY",
            "evidence_summary": "; ".join(
                f"{row.model} {row.metric}: {row.effect_intervention_minus_reference:.3g} [{row.ci_low:.3g},{row.ci_high:.3g}]"
                for row in structured_effects.itertuples()
            ),
            "recommended_wording_action": (
                "State that paired trained-factor interventions support a mask mechanism, while labeling them conditional operators."
                if mask_pass
                else "Present cross-model rank curves as architectural associations; do not say the mask alone explains them."
            ),
        }
    )
    return pd.DataFrame(decisions)


def assumption_audit_table(
    metrics: pd.DataFrame,
    reconstruction: pd.DataFrame,
    slopes: pd.DataFrame,
    geometry: pd.DataFrame,
    memory: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    reconstruction_map = reconstruction.set_index("checkpoint")["passed"].to_dict() if not reconstruction.empty else {}
    for model in PAPER_MODELS:
        model_rows = metrics[metrics["model"] == model]
        for layer in LAYERS[model]:
            subset = model_rows[model_rows["layer"] == layer]
            geometry_subset = geometry[(geometry.get("model") == model) & (geometry.get("layer") == layer)] if not geometry.empty else pd.DataFrame()
            memory_subset = memory[(memory.get("model") == model) & (memory.get("layer") == layer)] if not memory.empty else pd.DataFrame()
            slope_subset = slopes[(slopes["model"] == model) & (slopes["layer"] == layer)]
            rows.append(
                {
                    "checkpoint": model,
                    "layer": layer,
                    "wp0_reconstruction_passed": reconstruction_map.get(model, False),
                    "documents": int(subset["document_id"].nunique()),
                    "heads_or_channels": int(subset["head_or_channel"].nunique()),
                    "lengths": int(subset["length"].nunique()),
                    "native_rank_slope_intervals_available": int(slope_subset["metric"].nunique()) == 3,
                    "qk_geometry_available": not geometry_subset.empty,
                    "median_max_r95_fraction": float(geometry_subset["max_r95_fraction"].median()) if not geometry_subset.empty else math.nan,
                    "truncation_fidelity_available": bool(not geometry_subset.empty and geometry_subset["mixer_error"].notna().any()),
                    "sink_diagnostics_available": bool(model == "gpt2" and subset.get("sink_D_over_S", pd.Series(dtype=float)).notna().any()),
                    "memory_diagnostics_available": bool(model in ("mamba2", "retnet") and not memory_subset.empty),
                    "measured_mask_gain_available": bool(not memory_subset.empty and memory_subset["mask_kappa"].notna().any()),
                }
            )
    return pd.DataFrame(rows)


def write_tables(
    tables: Path,
    reconstruction: pd.DataFrame,
    slopes: pd.DataFrame,
    assumption: pd.DataFrame,
    decisions: pd.DataFrame,
    paired: pd.DataFrame,
    correlations: pd.DataFrame,
    memory_correlation_table: pd.DataFrame,
) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    required = {
        "table1_reconstruction_errors": reconstruction,
        "table2_bootstrap_slopes": slopes,
        "table3_assumption_audit": assumption,
        "table4_claim_decisions": decisions,
    }
    for stem, frame in required.items():
        atomic_csv(frame, tables / f"{stem}.csv")
        frame_to_latex(frame, tables / f"{stem}.tex")
    atomic_csv(paired, tables / "paired_intervention_effects.csv")
    atomic_csv(correlations, tables / "gpt2_sink_correlations.csv")
    atomic_csv(memory_correlation_table, tables / "memory_rank_correlations.csv")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame[list(columns)].copy()
    for column in selected:
        selected[column] = selected[column].map(
            lambda value: "--" if pd.isna(value) else f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
        )
    headers = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |" for _, row in selected.iterrows()]
    return "\n".join([headers, separator, *rows])


def write_reports(
    results_dir: Path,
    validation: Audit,
    metrics: pd.DataFrame,
    reconstruction: pd.DataFrame,
    slopes: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:
    final_slopes = slopes[slopes["layer_role"] == "final"]
    report = f"""# ICLR 2027 Paper Experiment Report

## Analysis contract

- Independent unit: WikiText document (`n={metrics['document_id'].nunique()}`).
- Within-document aggregation: median over heads/channels.
- Confidence intervals: 95% percentile document bootstrap with {BOOTSTRAP_RESAMPLES:,} resamples and seed {BASE_SEED}.
- Slopes: least-squares log--log slopes over `ell >= {SLOPE_MIN_LENGTH}`, refit inside every document bootstrap sample.
- Interventions: paired at document, layer, head, condition, and length; head-level differences are median-aggregated inside document before bootstrapping.
- Negative and heterogeneous results are retained. Automated decision thresholds are disclosed in the claim table and are not p-values.

## Completeness and reconstruction

All {sum(bool(row['required']) for row in validation.checks)} required validation checks passed before this report was generated. The canonical table contains {len(metrics):,} unique observations. Operator reconstruction results:

{_markdown_table(reconstruction, ('checkpoint', 'passed', 'median_relative_error', 'maximum_relative_error', 'maximum_fp32_reference_error'))}

## Rank scaling (WP1)

Final-layer document-bootstrap slopes:

{_markdown_table(final_slopes, ('series', 'metric', 'slope', 'ci_low', 'ci_high', 'n_documents'))}

The slope intervals, not point slopes alone, determine whether the report calls a curve increasing. Per-document slope distributions and all early/middle/final estimates are retained in `bootstrap/` and `tables/table2_bootstrap_slopes.csv`.

## GPT-2 sink mechanism (WP2)

The analysis uses the theorem's first causal column as the confirmatory sink. It separately evaluates native-prefix and explicit-boundary inputs, exact S/D residuals, pre-softmax margins and row violations, singular-vector localization, document-bootstrap correlations, and the paired neutralize-first/late-spike analytical counterfactuals. See Figures 2--3 and `tables/gpt2_sink_correlations.csv`.

## Learned factor geometry (WP3)

Figure 4 reports factor `r95/d_h`, token-space alignment, and rank-16 mixer error. The claim decision additionally applies the preregistered stronger rule: a rank no larger than `min(16,d_h/4)` must retain 95% of both factor energies and reproduce all mixer rank proxies within 10% on the median document-head. Algebraic rank bounded by head dimension is not counted as learned compression.

## Memory and mask interventions (WP4--WP5)

Figure 5 reports trained Mamba block contraction, cumulative row/column memory, measured gain, and rank against gain. Paired same-factor mask interventions are retained in `tables/paired_intervention_effects.csv`; these conditional operators are not language-model forward passes. The paper's claim-specific Figure 6 is generated separately by `plot_mask_replacement_figure.py`.

## Claim decisions

{_markdown_table(decisions, ('manuscript_claim', 'result', 'recommended_wording_action'))}

## Candidate artifacts

1. `figures/figure1_native_rank_scaling.pdf`
2. `figures/figure2_gpt2_depth_head_sink.pdf`
3. `figures/figure3_gpt2_sink_interventions.pdf`
4. `figures/figure4_low_rank_dashboard.pdf`
5. `figures/figure5_mamba_memory_dashboard.pdf`

The recommended main-paper candidates are Figure 1 and Figure 3 when the sink rule passes. The remaining diagnostics belong in the appendix because the main paper is page-limited.
"""
    atomic_text(results_dir / "REPORT.md", report)
    actions = f"""# Manuscript Actions from the ICLR 2027 Experiments

This file applies the preregistered decision rules before prose is edited. It does not select favorable subsets.

{_markdown_table(decisions, ('manuscript_claim', 'evidence_required', 'result', 'recommended_wording_action'))}

## Required global edits

- Replace the two-span/four-head setup with the 32-document nested-prefix protocol and document-bootstrap uncertainty.
- Report slope intervals over `ell >= 256`; do not reuse the old point slopes as exact exponents.
- Label every score, mask, normalization, and truncation intervention as an analytical conditional operator, not a valid model forward pass.
- State exact frozen checkpoint and WikiText revisions from `checkpoint_manifest.json`.
- Preserve the distinction between finite-length checkpoint evidence and asymptotic theorems.
- Remove the GPT-2 placeholder sentence and replace it only with the action selected in the claim-decision table.

## Placement

- Main text: robust rank-scaling result plus at most one mechanism figure.
- Experimental appendix: reconstruction table, full slope table, per-document slopes, GPT-2 heatmaps and correlations, learned-factor dashboard, memory dashboard, and matched interventions.
- Limitations: checkpoint confounds remain despite within-checkpoint interventions; no downstream behavior experiment was run, so spectral rank is not evidence of task-relevant context use.
"""
    atomic_text(results_dir / "MANUSCRIPT_ACTIONS.md", actions)


def write_analysis_metadata(results_dir: Path, metrics: pd.DataFrame) -> None:
    config = f"""analysis_version: 1
bootstrap_unit: document
within_document_head_aggregation: median
bootstrap_resamples: {BOOTSTRAP_RESAMPLES}
bootstrap_seed: {BASE_SEED}
slope_min_length: {SLOPE_MIN_LENGTH}
expected_documents: {EXPECTED_DOCUMENTS}
primary_lengths: [{', '.join(map(str, PRIMARY_LENGTHS))}]
"""
    atomic_text(results_dir / "config.yaml", config)
    readme = f"""# ICLR 2027 experiment artifacts

This directory contains the preregistered missing-experiment run and its strict analysis. `metrics.parquet` has one row per checkpoint/corpus/document/input-condition/length/layer/head/operator variant ({len(metrics):,} rows). Large arrays remain in `spectra/` or work-package shards and are referenced from the table.

Reproduce validation and analysis from the repository root:

```bash
python aggregate_experiments_2027.py validate --results-dir experiments_2027
python aggregate_experiments_2027.py analyze --results-dir experiments_2027
```

The analysis refuses incomplete trained-model designs, duplicate observation keys, nonfinite required metrics, failed WP0 operators, and broken NPZ references.
"""
    atomic_text(results_dir / "README.md", readme)


def run_validation(results_dir: Path) -> tuple[Audit, pd.DataFrame, dict[str, Any]]:
    audit = Audit()
    documents = _manifest_documents(results_dir, audit)
    validate_checkpoint_manifest(results_dir, audit)
    validate_run_provenance(results_dir, audit)
    reconstruction = validate_reconstruction(results_dir, audit)
    metrics, _ = load_metrics(results_dir, audit)
    validate_metrics_design(results_dir, metrics, documents, audit)
    return audit, metrics, reconstruction


def analyze(results_dir: Path) -> None:
    audit, raw_metrics, reconstruction_json = run_validation(results_dir)
    validation_path = results_dir / "logs" / "analysis_validation.json"
    atomic_json(validation_path, {"passed": not audit.failures, "checks": audit.checks})
    audit.require()
    metrics = canonical_metrics(raw_metrics)
    atomic_parquet(metrics, results_dir / "metrics.parquet")
    bootstrap = BootstrapEngine()
    curves, slopes, slope_replicates, per_document_slopes = curve_and_slope_tables(metrics, bootstrap)
    paired = paired_effect_table(metrics, bootstrap)
    geometry = geometry_table(metrics)
    memory = memory_table(metrics)
    correlations = sink_correlations(metrics, bootstrap)
    sink_paired = gpt2_sink_paired_table(metrics, bootstrap)
    memory_correlation_results = memory_correlations(memory, bootstrap)
    bootstrap_dir = results_dir / "bootstrap"
    atomic_parquet(curves, bootstrap_dir / "curve_summary.parquet")
    atomic_parquet(slope_replicates, bootstrap_dir / "slope_replicates.parquet")
    atomic_parquet(per_document_slopes, bootstrap_dir / "per_document_slopes.parquet")
    atomic_parquet(paired, bootstrap_dir / "paired_intervention_summary.parquet")
    atomic_parquet(correlations, bootstrap_dir / "gpt2_sink_correlations.parquet")
    atomic_parquet(sink_paired, bootstrap_dir / "gpt2_sink_paired_effects.parquet")
    atomic_parquet(geometry, bootstrap_dir / "geometry_observations.parquet")
    atomic_parquet(memory, bootstrap_dir / "memory_observations.parquet")
    atomic_parquet(memory_correlation_results, bootstrap_dir / "memory_rank_correlations.parquet")
    reconstruction = reconstruction_table(reconstruction_json)
    reconstruction = reconstruction[reconstruction["checkpoint"].isin(PAPER_MODELS)].copy()
    decisions = decide_claims(metrics, slopes, paired, geometry, memory, bootstrap, sink_paired)
    assumption = assumption_audit_table(metrics, reconstruction, slopes, geometry, memory)
    write_tables(
        results_dir / "tables",
        reconstruction,
        slopes,
        assumption,
        decisions,
        paired,
        correlations,
        memory_correlation_results,
    )
    atomic_csv(sink_paired, results_dir / "tables" / "gpt2_sink_paired_effects.csv")
    figures = results_dir / "figures"
    plot_rank_scaling(curves, figures)
    plot_gpt_heatmaps(metrics, figures)
    plot_gpt_interventions(metrics, bootstrap, figures)
    plot_low_rank_dashboard(geometry, bootstrap, figures)
    plot_mamba_memory(memory, bootstrap, figures)
    write_reports(results_dir, audit, metrics, reconstruction, slopes, decisions)
    write_analysis_metadata(results_dir, metrics)
    print(
        f"Analysis complete: {len(metrics):,} observations, {len(curves):,} curve estimates, "
        f"{len(slopes):,} slope intervals, {len(paired):,} paired effects.",
        flush=True,
    )


def finish_cached_analysis(results_dir: Path) -> None:
    """Finish plotting/reporting after bootstrap artifacts were written."""

    validation_path = results_dir / "logs" / "analysis_validation.json"
    if not validation_path.exists():
        raise AnalysisError("Cached finish requires logs/analysis_validation.json")
    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation_payload.get("passed"):
        raise AnalysisError("Cached finish refused because the saved validation did not pass")
    audit = Audit(checks=list(validation_payload.get("checks", [])))
    audit.require()

    required_paths = {
        "metrics": results_dir / "metrics.parquet",
        "curves": results_dir / "bootstrap" / "curve_summary.parquet",
        "slopes": results_dir / "tables" / "table2_bootstrap_slopes.csv",
        "paired": results_dir / "bootstrap" / "paired_intervention_summary.parquet",
        "geometry": results_dir / "bootstrap" / "geometry_observations.parquet",
        "memory": results_dir / "bootstrap" / "memory_observations.parquet",
        "sink_correlations": results_dir / "bootstrap" / "gpt2_sink_correlations.parquet",
        "sink_paired": results_dir / "bootstrap" / "gpt2_sink_paired_effects.parquet",
        "memory_correlations": results_dir / "bootstrap" / "memory_rank_correlations.parquet",
        "reconstruction": results_dir / "tables" / "table1_reconstruction_errors.csv",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise AnalysisError(f"Cached finish is missing required artifacts: {missing}")

    metrics = pd.read_parquet(required_paths["metrics"])
    metrics = metrics[metrics["model"].isin(PAPER_MODELS)].copy()
    curves = pd.read_parquet(required_paths["curves"])
    curves = curves[curves["model"].isin(PAPER_MODELS)].copy()
    slopes = pd.read_csv(required_paths["slopes"])
    slopes = slopes[slopes["model"].isin(PAPER_MODELS)].copy()
    paired = pd.read_parquet(required_paths["paired"])
    geometry = pd.read_parquet(required_paths["geometry"])
    memory = pd.read_parquet(required_paths["memory"])
    sink_correlations_table = pd.read_parquet(required_paths["sink_correlations"])
    sink_paired = pd.read_parquet(required_paths["sink_paired"])
    memory_correlation_table = pd.read_parquet(required_paths["memory_correlations"])
    reconstruction = pd.read_csv(required_paths["reconstruction"])
    reconstruction = reconstruction[reconstruction["checkpoint"].isin(PAPER_MODELS)].copy()

    bootstrap = BootstrapEngine()
    decisions = decide_claims(metrics, slopes, paired, geometry, memory, bootstrap, sink_paired)
    assumption = assumption_audit_table(metrics, reconstruction, slopes, geometry, memory)
    write_tables(
        results_dir / "tables",
        reconstruction,
        slopes,
        assumption,
        decisions,
        paired,
        sink_correlations_table,
        memory_correlation_table,
    )
    atomic_csv(sink_paired, results_dir / "tables" / "gpt2_sink_paired_effects.csv")
    figures = results_dir / "figures"
    plot_rank_scaling(curves, figures)
    plot_gpt_heatmaps(metrics, figures)
    plot_gpt_interventions(metrics, bootstrap, figures)
    plot_low_rank_dashboard(geometry, bootstrap, figures)
    plot_mamba_memory(memory, bootstrap, figures)
    write_reports(results_dir, audit, metrics, reconstruction, slopes, decisions)
    write_analysis_metadata(results_dir, metrics)
    print(f"Cached analysis finish complete: {len(metrics):,} observations; figures and reports regenerated.")


def self_test() -> None:
    bootstrap = BootstrapEngine(resamples=400, seed=BASE_SEED)
    values = np.asarray([1.0, 2.0, 50.0])
    point, low, high, replicates = bootstrap.median(values)
    assert point == 2.0 and low <= point <= high and replicates.shape == (400,)

    lengths = np.asarray([256, 512, 1024])
    document_scales = np.asarray([0.8, 1.0, 1.3, 2.0])[:, None]
    matrix = document_scales * lengths[None, :] ** 0.75
    slope, slope_low, slope_high, slope_replicates = bootstrap.slope(matrix, lengths)
    assert abs(slope - 0.75) < 1e-12
    assert abs(slope_low - 0.75) < 1e-12 and abs(slope_high - 0.75) < 1e-12
    assert slope_replicates.shape == (400,)

    head_rows = pd.DataFrame(
        [
            {"document_id": doc, "head_or_channel": head, "length": length, "metric": scale * length}
            for doc, scale in (("a", 1.0), ("b", 2.0))
            for head in (0, 1, 2)
            for length in (256, 512)
        ]
    )
    medians = _doc_head_medians(head_rows, "metric", ("length",))
    assert medians.shape[0] == 4 and set(medians["n_heads"]) == {3}

    duplicate_audit = Audit()
    duplicate = pd.DataFrame(
        [
            {
                "checkpoint": "x",
                "corpus": "c",
                "document_id": "d",
                "input_condition": "i",
                "length": 64,
                "layer": 0,
                "head_or_channel": 0,
                "operator_variant": "v",
            }
        ]
        * 2
    )
    duplicate = _normalize_model_table(duplicate, "gpt2", [Path("synthetic.jsonl")])
    duplicate_flag = duplicate.duplicated(list(OBSERVATION_KEY), keep=False)
    duplicate_audit.add("duplicate rejection", not duplicate_flag.any(), f"duplicates={duplicate_flag.sum()}")
    assert duplicate_audit.failures

    structured_geometry = {
        "1": {
            "rank": 1,
            "token_canonical_correlations": [0.8],
            "token_alignment_score": 0.64,
            "token_procrustes_distance": 0.2,
            "projection_scaled_entries": [0.1],
            "projection_overlap_singular_values": [0.1],
            "projection_normal_qq_probabilities": [0.5],
            "projection_normal_qq_theoretical": [0.0],
            "projection_normal_qq_empirical": [0.1],
        }
    }
    assert _structured_geometry_contract(structured_geometry, "geometry_by_rank", "canonical")
    assert _structured_geometry_contract(structured_geometry, "geometry_by_rank", "token")
    assert _structured_geometry_contract(structured_geometry, "geometry_by_rank", "projection")
    union_schema_audit = Audit()
    _json_candidate_contract_audit(
        union_schema_audit,
        "union-schema geometry",
        pd.DataFrame(
            {
                "token_alignment": [np.nan],
                "geometry_by_rank": [canonical_json(structured_geometry)],
            }
        ),
        ("token_alignment", "geometry_by_rank"),
        lambda value, column: _structured_geometry_contract(value, column, "token"),
    )
    assert not union_schema_audit.failures, union_schema_audit.failures
    alignment, rank = _alignment_at_rank(pd.Series({"geometry_by_rank": canonical_json(structured_geometry)}))
    assert alignment == 0.64 and rank == 1
    assert _factor_dimension({"length": 64, "head_dim": 16}, {"r95": 8}, {"r95": 7}) == 16
    assert _factor_dimension({"length": 64}, {"dimension": 24}, {"dimension": 20}) == 20

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "experiments_2027"
        path = root / "test.parquet"
        atomic_parquet(pd.DataFrame({"x": [1, 2], "label": ["a", "b"]}), path)
        roundtrip = pd.read_parquet(path)
        assert roundtrip.to_dict("list") == {"x": [1, 2], "label": ["a", "b"]}
        structured_archive = root / "structured_shards" / "based" / "spectra" / "unit.npz"
        structured_archive.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(structured_archive, **{"spectrum-structured": np.asarray([1.0, 0.5])})
        gpt_archive = root / "gpt2_runs" / "config" / "arrays" / "sample.npz"
        gpt_archive.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gpt_archive, **{"spectrum-gpt": np.asarray([1.0, 0.25])})
        spectrum_rows = pd.DataFrame(
            [
                {"model": "based", "spectrum_id": "spectrum-structured"},
                {
                    "model": "gpt2",
                    "spectrum_ref": "spectrum-gpt",
                    "array_archive": str(gpt_archive.relative_to(root)),
                },
            ]
        )
        spectrum_audit = Audit()
        _validate_spectrum_references(root, spectrum_rows, spectrum_audit)
        assert not spectrum_audit.failures, spectrum_audit.failures
    print(
        "Self-test passed: document bootstrap, slope recovery, head aggregation, "
        "duplicate rejection, structured geometry, spectrum archives, parquet I/O."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("validate", "analyze", "finish-cached", "self-test"),
        default="analyze",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("experiments_2027"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "self-test":
        self_test()
        return
    results_dir = args.results_dir.resolve()
    write_run_provenance(results_dir)
    if args.action == "finish-cached":
        finish_cached_analysis(results_dir)
        return
    if args.action == "validate":
        audit, _, _ = run_validation(results_dir)
        atomic_json(results_dir / "logs" / "analysis_validation.json", {"passed": not audit.failures, "checks": audit.checks})
        audit.require()
        print(f"Validation passed: {len(audit.checks)} required checks.")
        return
    analyze(results_dir)


if __name__ == "__main__":
    try:
        main()
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
