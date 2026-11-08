#!/usr/bin/env python3
"""Document-cluster analysis of the rebuttal functional and cross-cut runs.

No model or checkpoint is loaded. Input JSONL observations are retained, exact
duplicates (e.g. consolidated files plus shards) are removed, and conflicting
duplicates fail loudly. Heads/windows are averaged *within documents* before
bootstrap inference; they are never counted as independent samples.

Examples::

    python analyze_rebuttal.py analyze --functional-dir results/functional \
        --crosscut-dir results/crosscut --output-dir results/analysis
    python analyze_rebuttal.py self-test

Only plotting requires matplotlib. The analysis itself uses the standard
library. Confidence intervals are descriptive percentile document bootstraps,
not multiplicity-adjusted hypothesis tests or causal estimates of rank effects.
"""

from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:  # Optional acceleration; resampling weights are identical without NumPy.
    import numpy as np
except ImportError:
    np = None


DEFAULT_SEED = 20261108
DEFAULT_BOOTSTRAPS = 2000
RUN_DIMS = ("run_id", "checkpoint", "checkpoint_revision", "corpus", "input_condition", "length", "layer")
REGION_DIMS = ("region", "cut_fraction", "cut_index", "gap", "window_size")
FUNCTIONAL_METRICS = (
    "delta_loss", "native_loss", "intervention_loss", "native_srank",
    "native_erank_1", "native_erank_2", "sink_S", "sink_D", "sink_D_over_S",
    "sink_rho2", "av_fro", "contribution_fro", "contribution_token_variance_mean",
    "residual_input_fro", "relative_contribution_fro", "intervention_core_relative_change",
    "core_reconstruction_relative_error", "projected_reconstruction_relative_error",
    "zero_loss_abs_error", "zero_logits_max_abs_error",
    "native_operator_norm", "native_operator_fro", "native_best_rank1_error", "projected_output_fro",
    "intervention_srank", "intervention_erank_1", "intervention_erank_2",
    "intervention_contribution_delta_fro", "intervention_target_input_relative_error",
    "native_svd_frobenius_norm", "native_svd_frobenius_relative_error",
    "intervention_svd_frobenius_norm", "intervention_svd_frobenius_relative_error",
)
CROSSCUT_METRICS = (
    "srank", "erank_1", "erank_2", "operator_norm", "frobenius_norm", "nuclear_norm",
    "srank_fraction", "erank_1_fraction", "erank_2_fraction", "normalized_srank",
    "normalized_erank_1", "normalized_erank_2", "srank_normalized", "erank_1_normalized",
    "erank_2_normalized", "frobenius_energy_fraction_full", "value_output_frobenius_norm",
    "value_input_frobenius_norm", "value_output_to_input_gain", "diagonal_energy_fraction",
    "strict_past_energy_fraction", "core_reconstruction_relative_error",
    "native_core_reconstruction_relative_error",
    "future_energy_fraction", "entry_frobenius_norm", "svd_frobenius_norm",
    "svd_frobenius_relative_error", "svd_squared_energy_relative_error",
)
CORRELATION_PAIRS = (
    ("native_srank", "delta_loss"),
    ("native_erank_2", "delta_loss"),
    ("sink_rho2", "delta_loss"),
    ("sink_rho2", "relative_contribution_fro"),
    ("native_srank", "relative_contribution_fro"),
)


class AnalysisError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@lru_cache(maxsize=16)
def bootstrap_counts(n: int, seed: int, resamples: int) -> tuple:
    """Reuse the same document resamples across outcomes of the same size."""
    rng = random.Random(seed)
    return tuple(tuple(Counter(rng.randrange(n) for _ in range(n)).items()) for _ in range(resamples))


@lru_cache(maxsize=16)
def dense_bootstrap_counts(n: int, seed: int, resamples: int):
    result = np.zeros((resamples, n), dtype=float)
    for row, counts in enumerate(bootstrap_counts(n, seed, resamples)):
        for index, count in counts:
            result[row, index] = count
    return result


def bootstrap(values: list[float], *, seed: int, resamples: int) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    n = len(values)
    if np is not None:
        sampled = np.einsum("ij,j->i", dense_bootstrap_counts(n, seed, resamples), np.asarray(values)) / n
        low, high = np.quantile(sampled, [.025, .975])
        return float(low), float(high)
    sampled = [sum(values[index] * count for index, count in counts) / n
               for counts in bootstrap_counts(n, seed, resamples)]
    return quantile(sampled, 0.025), quantile(sampled, 0.975)


def keyed_seed(seed: int, key: Any) -> int:
    digest = hashlib.sha256(canonical(key).encode()).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def flatten_numeric(row: dict[str, Any], prefix: str = "") -> dict[str, float]:
    result = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_numeric(value, name))
        elif finite(value):
            result[name] = float(value)
    return result


def group_key(row: dict, keys: tuple[str, ...]) -> tuple:
    return tuple(row.get(key) for key in keys)


def group_dict(key: tuple, keys: tuple[str, ...]) -> dict:
    return dict(zip(keys, key))


def jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def normalize(row: dict, path: Path, line: int, family: str) -> dict:
    row = dict(row)
    row["checkpoint"] = row.get("checkpoint", row.get("model"))
    row["head_or_channel"] = row.get("head_or_channel", row.get("head"))
    row["sample_offset"] = row.get("sample_offset", 0)
    row["window_size"] = row.get("window_size", row.get("window"))
    row["input_condition"] = row.get("input_condition", "native-prefix")
    row["corpus"] = row.get("corpus", "unspecified")
    config_parent = next((part for part in path.parts if part.startswith("config-")), None)
    row["run_id"] = row.get("run_id") or row.get("config_id") or row.get("run_fingerprint") or config_parent or str(path.parent.resolve())
    row["status"] = row.get("status", "ok")
    row["_source_file"] = str(path.resolve())
    row["_source_line"] = line
    row["_family"] = family
    if family == "functional" and row["status"] == "ok":
        native, changed = row.get("native_loss"), row.get("intervention_loss")
        if finite(native) and finite(changed):
            computed_delta = changed - native
            recorded_delta = row.get("delta_loss")
            if finite(recorded_delta) and not math.isclose(recorded_delta, computed_delta, abs_tol=1e-6, rel_tol=1e-5):
                raise AnalysisError(f"Inconsistent delta_loss at {path}:{line}")
            row["delta_loss"] = computed_delta
        count = row.get("loss_token_count")
        if finite(count) and count > 0:
            for loss_key, sum_key in (("native_loss", "native_nll_sum"), ("intervention_loss", "intervention_nll_sum")):
                if finite(row.get(loss_key)) and finite(row.get(sum_key)) and not math.isclose(row[loss_key], row[sum_key] / count, abs_tol=2e-6, rel_tol=2e-5):
                    raise AnalysisError(f"Inconsistent {loss_key} and token sum at {path}:{line}")
    required = ("checkpoint", "document_id", "length", "layer", "head_or_channel")
    if row["status"] == "ok" and any(row.get(key) is None for key in required):
        raise AnalysisError(f"Missing observation identity at {path}:{line}")
    return row


def observation_key(row: dict, family: str) -> tuple:
    keys = RUN_DIMS + ("document_id", "sample_offset", "head_or_channel", "operator_variant")
    keys += ("intervention",) if family == "functional" else REGION_DIMS + ("row_start", "row_stop", "col_start", "col_stop")
    return group_key(row, keys)


def load_rows(inputs: list[str], family: str, quality: dict) -> list[dict]:
    paths = set()
    for value in inputs:
        path = Path(value)
        if not path.exists():
            raise AnalysisError(f"Input does not exist: {path}")
        paths.update(path.rglob("*.jsonl") if path.is_dir() else [path])
    seen = {}
    errors = []
    duplicates = 0
    skipped = 0
    inputs_audit = []
    for path in sorted(paths):
        # Ignore previous analysis exports when an enclosing results dir is passed.
        if path.name.startswith("raw_") or path.name.endswith("_documents.jsonl"):
            continue
        inputs_audit.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise AnalysisError(f"Invalid/nonfinite JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise AnalysisError(f"Expected JSON object at {path}:{line_number}")
            matches = ("intervention" in row or "native_loss" in row) if family == "functional" else row.get("region") in ("full", "crosscut", "window")
            if not matches:
                if row.get("status") not in (None, "ok"):
                    errors.append({**row, "_source_file": str(path.resolve()), "_source_line": line_number})
                else:
                    skipped += 1
                continue
            row = normalize(row, path, line_number, family)
            key = observation_key(row, family)
            payload = {key: value for key, value in row.items() if not key.startswith("_source")}
            if key in seen:
                previous = {key: value for key, value in seen[key].items() if not key.startswith("_source")}
                if canonical(payload) != canonical(previous):
                    raise AnalysisError(f"Conflicting duplicate observation: {path}:{line_number}; first at {seen[key]['_source_file']}:{seen[key]['_source_line']}")
                duplicates += 1
            else:
                seen[key] = row
    # Runner failures are JSON sidecars, not successful metric shards.
    sidecar_paths = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            sidecar_paths.update(candidate for candidate in path.rglob("*.json") if "errors" in candidate.parts)
    for path in sorted(sidecar_paths):
        try:
            errors.append({"_source_file": str(path.resolve()), "error_record": json.loads(path.read_text(encoding="utf-8"))})
        except (ValueError, OSError) as exc:
            errors.append({"_source_file": str(path.resolve()), "error_reading_sidecar": str(exc)})
    rows = list(seen.values())
    quality[family] = {
        "input_files": inputs_audit, "unique_rows": len(rows), "exact_duplicates_removed": duplicates,
        "unrelated_rows_skipped": skipped, "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "failure_records": errors,
        "unique_documents": len({row.get("document_id") for row in rows if row.get("document_id") is not None}),
    }
    return rows


def metric_names(rows: list[dict], family: str) -> tuple[str, ...]:
    available = {key for row in rows for key in flatten_numeric(row)}
    if family == "functional":
        return tuple(key for key in FUNCTIONAL_METRICS if key in available)
    dynamic = tuple(sorted(key for key in available if key.startswith(("tolerance_ess.", "relative_tolerance_rank.", "lag_gt_", "lag_energy_fractions."))))
    return tuple(key for key in CROSSCUT_METRICS if key in available) + dynamic


def summarize(rows: list[dict], dims: tuple[str, ...], metrics: tuple[str, ...], args) -> tuple[list[dict], list[dict]]:
    """First equal-weight observations within document; then bootstrap documents."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[group_key(row, dims)].append(row)
    summary, document_rows = [], []
    for key, observations in sorted(grouped.items(), key=lambda item: repr(item[0])):
        identity = group_dict(key, dims)
        documents = defaultdict(list)
        for row in observations:
            documents[str(row["document_id"])].append(row)
        for metric in metrics:
            values = []
            doc_ids = []
            n_observations = 0
            for document_id, doc_rows in sorted(documents.items()):
                measured = [row.get("_numeric_metrics", {}).get(metric) for row in doc_rows]
                valid = [value for value in measured if value is not None]
                if not valid:
                    continue
                value = mean(valid)
                values.append(value)
                doc_ids.append(document_id)
                n_observations += len(valid)
                document_rows.append({**identity, "document_id": document_id, "metric": metric, "value": value,
                                      "n_observations": len(valid), "n_available_observations": len(doc_rows),
                                      "n_heads": len({row["head_or_channel"] for row in doc_rows})})
            if not values:
                continue
            low, high = bootstrap(values, seed=args.seed, resamples=args.bootstrap_resamples)
            summary.append({**identity, "metric": metric, "estimate": mean(values), "ci_low": low, "ci_high": high,
                            "median": statistics.median(values), "min": min(values), "max": max(values),
                            "n_documents": len(values), "n_available_documents": len(documents),
                            "n_observations": n_observations,
                            "n_positive_documents": sum(value > 0 for value in values),
                            "n_negative_documents": sum(value < 0 for value in values),
                            "document_ids": ";".join(doc_ids)})
    return summary, document_rows


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        average = (start + stop - 1) / 2
        for index in ordered[start:stop]:
            result[index] = average
        start = stop
    return result


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    xx = sum((value - mx) ** 2 for value in rx)
    yy = sum((value - my) ** 2 for value in ry)
    if xx == 0 or yy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(xx * yy)


def functional_correlations(rows: list[dict], args) -> tuple[list[dict], list[dict]]:
    dims = RUN_DIMS + ("intervention",)
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            grouped[group_key(row, dims + ("document_id",))].append(row)
    correlations = []
    for key, doc_rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        identity = group_dict(key, dims + ("document_id",))
        heads = defaultdict(list)
        for row in doc_rows:
            heads[row["head_or_channel"]].append(row)
        for xname, yname in CORRELATION_PAIRS:
            pairs = []
            for head_rows in heads.values():
                valid = [(row[xname], row[yname]) for row in head_rows if finite(row.get(xname)) and finite(row.get(yname))]
                if valid:
                    pairs.append((mean(x for x, _ in valid), mean(y for _, y in valid)))
            rho = spearman([x for x, _ in pairs], [y for _, y in pairs])
            correlations.append({**identity, "metric": f"spearman({xname},{yname})", "value": rho,
                                 "n_heads": len(pairs), "status": "ok" if rho is not None else "insufficient_or_constant"})
    groups = defaultdict(list)
    for row in correlations:
        groups[group_key(row, dims + ("metric",))].append(row)
    summary = []
    for key, values in sorted(groups.items(), key=lambda item: repr(item[0])):
        valid = [row["value"] for row in values if finite(row["value"])]
        low, high = bootstrap(valid, seed=args.seed, resamples=args.bootstrap_resamples)
        summary.append({**group_dict(key, dims + ("metric",)), "estimate": mean(valid) if valid else None,
                        "ci_low": low, "ci_high": high, "n_documents": len(valid),
                        "n_available_documents": len(values), "n_undefined_documents": len(values) - len(valid)})
    return summary, correlations


def paired_crosscut(rows: list[dict], metrics: tuple[str, ...], args) -> tuple[list[dict], list[dict], list[dict]]:
    pair_dims = RUN_DIMS + ("document_id", "sample_offset", "head_or_channel") + REGION_DIMS + ("row_start", "row_stop", "col_start", "col_stop")
    groups = defaultdict(dict)
    for row in rows:
        if row["status"] == "ok":
            groups[group_key(row, pair_dims)][row.get("operator_variant", "native")] = row
    deltas, missing = [], []
    for key, variants in sorted(groups.items(), key=lambda item: repr(item[0])):
        for treatment, row in variants.items():
            baseline = {
                "all_one_dt": "native_dt",
                "all_one_dt_with_d": "native_dt_with_d",
            }.get(treatment, "native")
            if treatment in ("native", "native_dt", "native_dt_with_d"):
                continue
            if baseline not in variants:
                missing.append({**group_dict(key, pair_dims), "treatment": treatment, "missing_baseline": baseline})
                continue
            native = variants[baseline]
            changed_metrics, native_metrics = flatten_numeric(row), flatten_numeric(native)
            delta = {**group_dict(key, pair_dims), "status": "ok", "comparison": f"{treatment} minus {baseline}"}
            for metric in metrics:
                if metric in changed_metrics and metric in native_metrics:
                    delta[metric] = changed_metrics[metric] - native_metrics[metric]
            delta["_numeric_metrics"] = flatten_numeric(delta)
            deltas.append(delta)
    dims = RUN_DIMS + REGION_DIMS + ("comparison",)
    summary, documents = summarize(deltas, dims, metrics, args)
    return summary, documents, missing


def coverage_rows(rows: list[dict], family: str) -> list[dict]:
    dims = RUN_DIMS + (("intervention",) if family == "functional" else ("operator_variant",) + REGION_DIMS)
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row, dims)].append(row)
    return [{**group_dict(key, dims), "n_documents": len({row.get("document_id") for row in group}),
             "n_rows": len(group), "n_heads": len({row.get("head_or_channel") for row in group}),
             "status_counts": canonical(dict(Counter(row["status"] for row in group))),
             "min_heads_per_document": min(len({row.get("head_or_channel") for row in group if row.get("document_id") == doc}) for doc in {row.get("document_id") for row in group})}
            for key, group in sorted(grouped.items(), key=lambda item: repr(item[0]))]


def plot_results(output: Path, functional: list[dict], crosscut: list[dict], functional_rows: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    created = []
    functional_groups = defaultdict(list)
    for row in functional:
        if row["metric"] == "delta_loss":
            functional_groups[group_key(row, ("run_id", "checkpoint", "input_condition", "length"))].append(row)
    for key, rows in sorted(functional_groups.items(), key=lambda item: repr(item[0])):
        fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
        interventions = sorted({row["intervention"] for row in rows})
        for index, intervention in enumerate(interventions):
            group = sorted((row for row in rows if row["intervention"] == intervention), key=lambda row: row["layer"])
            x = [row["layer"] + (index - (len(interventions)-1)/2) * 0.12 for row in group]
            ax.plot(x, [row["estimate"] for row in group], marker="o", label=intervention)
            for position, row in zip(x, group):
                if row["ci_low"] is not None:
                    ax.vlines(position, row["ci_low"], row["ci_high"], alpha=.7)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set(xlabel="Layer", ylabel="Intervention minus native NLL (nats/token)",
               title=f"{key[1]} · context {key[3]} · {key[2]}\nMean across tested heads within each document")
        ax.legend(frameon=False)
        name = f"functional_effects_{hashlib.sha256(canonical(key).encode()).hexdigest()[:10]}"
        for suffix in ("pdf", "png"):
            path = figures / f"{name}.{suffix}"
            fig.savefig(path, dpi=180)
            created.append(str(path.relative_to(output)))
        plt.close(fig)
    # Head means use the same document intersection for both scatter axes.
    scatter_groups = defaultdict(list)
    for row in functional_rows:
        if row["status"] == "ok" and finite(row.get("sink_D_over_S")) and finite(row.get("delta_loss")):
            scatter_groups[group_key(row, ("run_id", "checkpoint", "input_condition", "length"))].append(row)
    for key, rows in sorted(scatter_groups.items(), key=lambda item: repr(item[0])):
        interventions = sorted({row["intervention"] for row in rows})
        fig, axes = plt.subplots(1, len(interventions), figsize=(5 * len(interventions), 3.8), constrained_layout=True, squeeze=False)
        for ax, intervention in zip(axes[0], interventions):
            heads = defaultdict(list)
            for row in rows:
                if row["intervention"] == intervention:
                    heads[(row["layer"], row["head_or_channel"])].append(row)
            for layer in sorted({layer for layer, _ in heads}):
                points = []
                for (head_layer, _), observations in heads.items():
                    if head_layer != layer:
                        continue
                    documents = defaultdict(list)
                    for row in observations:
                        documents[row["document_id"]].append(row)
                    pairs = [(mean(row["sink_D_over_S"] for row in doc), mean(row["delta_loss"] for row in doc)) for doc in documents.values()]
                    points.append((mean(x for x, _ in pairs), mean(y for _, y in pairs)))
                ax.scatter([x for x, _ in points], [y for _, y in points], label=f"layer {layer}", alpha=.8)
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_xscale("symlog", linthresh=1e-3)
            ax.set(xlabel="Native sink D/S (symmetric log scale)", ylabel="Intervention minus native NLL", title=intervention)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"{key[1]} · context {key[3]} · each point is one tested head\nAxes averaged over the same documents; descriptive association")
        name = f"functional_sink_scatter_{hashlib.sha256(canonical(key).encode()).hexdigest()[:10]}"
        for suffix in ("pdf", "png"):
            path = figures / f"{name}.{suffix}"
            fig.savefig(path, dpi=180)
            created.append(str(path.relative_to(output)))
        plt.close(fig)
    # Predetermined display: middle cut and longest available context for each
    # run/model/condition/layer. Full coverage remains in the CSV tables.
    cross_groups = defaultdict(list)
    for row in crosscut:
        if row.get("region") == "crosscut" and row.get("cut_fraction") == 0.5 and row["metric"] in ("erank_2", "frobenius_norm", "value_output_to_input_gain"):
            cross_groups[group_key(row, ("run_id", "checkpoint", "input_condition", "layer"))].append(row)
    for key, rows in sorted(cross_groups.items(), key=lambda item: repr(item[0])):
        longest = max(row["length"] for row in rows)
        rows = [row for row in rows if row["length"] == longest]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
        for ax, metric, label in zip(axes, ("erank_2", "frobenius_norm", "value_output_to_input_gain"),
                                      ("Squared-energy effective rank", "Cross-cut Frobenius norm", "Value output/input norm gain")):
            for variant in sorted({row["operator_variant"] for row in rows}):
                group = sorted((row for row in rows if row["operator_variant"] == variant and row["metric"] == metric), key=lambda row: row.get("gap") or 0)
                if not group:
                    continue
                x = [row.get("gap") or 0 for row in group]
                y = [row["estimate"] for row in group]
                ax.plot(x, y, marker="o", markersize=3, label=variant)
                for position, row in zip(x, group):
                    if row["ci_low"] is not None:
                        ax.vlines(position, row["ci_low"], row["ci_high"], alpha=.45)
            ax.set(xlabel="Gap after midpoint cut (tokens)", ylabel=label)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, frameon=False, fontsize=8)
        fig.suptitle(f"{key[1]} · layer {key[3]} · context {longest} · midpoint cut\nMeans across tested heads within document; 95% document bootstrap")
        name = f"crosscut_rank_gain_{hashlib.sha256(canonical(key).encode()).hexdigest()[:10]}"
        for suffix in ("pdf", "png"):
            path = figures / f"{name}.{suffix}"
            fig.savefig(path, dpi=180)
            created.append(str(path.relative_to(output)))
        plt.close(fig)
    return created


def report(output: Path, quality: dict, functional: list[dict], figures: list[str], args) -> None:
    lines = ["# Rebuttal experiment analysis", "", "This is an automatic descriptive analysis, not an adjudication of the paper's claims.", "",
             f"Bootstrap: {args.bootstrap_resamples} resamples; seed {args.seed}; equal-weight documents; percentile 95% intervals.", "",
             "Heads and multiple observations are averaged within each document before layer-level inference. Head-resolved tables bootstrap documents separately for each head. Cross-cut intervention differences are formed on exact matched observations before aggregation. No cross-model loss comparison or multiplicity-adjusted significance claim is made.", "",
             "Zero spectral ranks follow the runner's explicit convention. Always inspect absolute operator/output magnitudes alongside relative rank. Undefined/absent metrics remain absent rather than becoming zero. Failed rows and error sidecars are preserved in the audit; a missing run cannot be counted without its expected manifest.", "",
             "## Coverage", ""]
    for family in ("functional", "crosscut"):
        audit = quality[family]
        lines.append(f"- {family}: {audit['unique_rows']} unique rows, {audit['unique_documents']} document IDs, statuses {audit['status_counts']}; {len(audit['failure_records'])} failure records; {audit['exact_duplicates_removed']} exact duplicate rows removed.")
    lines += ["", "Per-config/model/context/layer coverage is in the coverage CSVs. Input hashes, failures, and pairing exclusions are in `data_quality.json`.", "", "## Functional interventions", "",
              "Positive delta means higher held-out NLL after intervention. Neither a rank/loss association nor head ablation establishes that rank itself causes predictive performance.", "",
              "| Model | Context | Layer | Intervention | Mean delta NLL | 95% interval | Documents |", "|---|---:|---:|---|---:|---|---:|"]
    for row in functional:
        if row["metric"] != "delta_loss":
            continue
        interval = "unavailable" if row["ci_low"] is None else f"[{row['ci_low']:.5g}, {row['ci_high']:.5g}]"
        lines.append(f"| {row['checkpoint']} | {row['length']} | {row['layer']} | {row['intervention']} | {row['estimate']:.5g} | {interval} | {row['n_documents']} |")
    lines += ["", "## Outputs", "",
              "- `raw_functional_rows.jsonl`, `raw_crosscut_rows.jsonl`: all unique observations, including failed statuses, with source provenance.",
              "- `functional_head_summary.csv`, `functional_layer_summary.csv`: all supported functional metrics.",
              "- `functional_layer_documents.csv`: the independent document summaries used for intervals.",
              "- `functional_correlations.csv`, `functional_document_correlations.csv`: within-document head Spearman associations, with undefined cases retained.",
              "- `crosscut_head_summary.csv`, `crosscut_layer_summary.csv`: all regions, cuts, gaps, variants and supported metrics.",
              "- `crosscut_paired_variant_effects.csv`: paired variant-minus-native changes, averaged within document.", "",
              "Cross-cut figures display the midpoint cut and longest measured context for each run/model/layer; all other measurements remain in the tables. Functional figures show all available context/intervention groups.", ""]
    for name in figures:
        if name.endswith(".png"):
            lines.append(f"![{Path(name).stem}]({name})")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args) -> dict:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    quality = {"bootstrap_resamples": args.bootstrap_resamples, "seed": args.seed, "confidence_level": .95,
               "independent_unit": "document_id within config/checkpoint/input/context/layer",
               "aggregation": "equal weight observations within document; equal weight documents",
               "limitations": ["No multiplicity correction", "No cross-model NLL comparison", "Partial coverage may be biased", "Within-document heads are not independent"]}
    functional = load_rows(args.functional_dir, "functional", quality)
    crosscut = load_rows(args.crosscut_dir, "crosscut", quality)
    jsonl_write(output / "raw_functional_rows.jsonl", functional)
    jsonl_write(output / "raw_crosscut_rows.jsonl", crosscut)
    for row in functional + crosscut:
        row["_numeric_metrics"] = flatten_numeric(row)
    csv_write(output / "functional_coverage.csv", coverage_rows(functional, "functional"))
    csv_write(output / "crosscut_coverage.csv", coverage_rows(crosscut, "crosscut"))
    fmetrics, cmetrics = metric_names(functional, "functional"), metric_names(crosscut, "crosscut")
    f_head, _ = summarize(functional, RUN_DIMS + ("intervention", "head_or_channel"), fmetrics, args)
    f_layer, f_docs = summarize(functional, RUN_DIMS + ("intervention",), fmetrics, args)
    c_head, _ = summarize(crosscut, RUN_DIMS + ("operator_variant",) + REGION_DIMS + ("head_or_channel",), cmetrics, args)
    c_layer, c_docs = summarize(crosscut, RUN_DIMS + ("operator_variant",) + REGION_DIMS, cmetrics, args)
    correlations, document_correlations = functional_correlations(functional, args)
    c_paired, c_paired_docs, missing = paired_crosscut(crosscut, cmetrics, args)
    for name, rows in (
        ("functional_head_summary", f_head), ("functional_layer_summary", f_layer),
        ("functional_layer_documents", f_docs), ("functional_correlations", correlations),
        ("functional_document_correlations", document_correlations), ("crosscut_head_summary", c_head),
        ("crosscut_layer_summary", c_layer), ("crosscut_layer_documents", c_docs),
        ("crosscut_paired_variant_effects", c_paired), ("crosscut_paired_documents", c_paired_docs),
    ):
        csv_write(output / f"{name}.csv", rows)
    quality["crosscut"]["missing_pair_baselines"] = missing
    quality["crosscut"]["metrics"] = cmetrics
    quality["functional"]["metrics"] = fmetrics
    figures = []
    if not args.no_plots:
        try:
            figures = plot_results(output, f_layer, c_layer, functional)
        except ImportError as exc:
            raise AnalysisError("Plotting requires matplotlib. Install it or explicitly pass --no-plots.") from exc
    quality["figures"] = figures
    json_write(output / "data_quality.json", quality)
    report(output, quality, f_layer, figures, args)
    return quality


def self_test() -> None:
    """Meaningful synthetic checks of clustering, pairing and missingness."""
    assert spearman([1, 2, 3], [3, 2, 1]) == -1
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    with tempfile.TemporaryDirectory(prefix="rebuttal-analysis-test-") as temporary:
        root = Path(temporary)
        functional, crosscut = [], []
        for doc, delta in (("a", .1), ("b", -.1), ("c", .3)):
            for head in range(3):
                base = {"config_id": "synthetic", "checkpoint": "test", "document_id": doc,
                        "length": 8, "layer": 0, "head_or_channel": head, "status": "ok"}
                functional.append({**base, "intervention": "ablate", "native_loss": 2.,
                                   "intervention_loss": 2 + delta, "loss_token_count": 7,
                                   "native_srank": head + 1., "relative_contribution_fro": head / 10.})
                for variant, value in (("native", 2.), ("all_one", 1.)):
                    crosscut.append({**base, "region": "crosscut", "cut_fraction": .5, "cut_index": 4,
                                     "gap": 0, "operator_variant": variant, "srank": value,
                                     "frobenius_norm": 0., "value_output_to_input_gain": None})
                # Distinct discretization/feedthrough conventions must never
                # use the canonical native row as their comparison baseline.
                for variant, value in (("native_dt", 4.), ("all_one_dt", 2.),
                                       ("native_dt_with_d", 7.), ("all_one_dt_with_d", 4.)):
                    crosscut.append({**base, "region": "crosscut", "cut_fraction": .5, "cut_index": 4,
                                     "gap": 0, "operator_variant": variant, "srank": value,
                                     "frobenius_norm": 0., "value_output_to_input_gain": None})
        jsonl_write(root / "functional.jsonl", functional)
        jsonl_write(root / "crosscut.jsonl", crosscut)
        args = argparse.Namespace(functional_dir=[str(root / "functional.jsonl")], crosscut_dir=[str(root / "crosscut.jsonl")],
                                  output_dir=str(root / "analysis"), seed=1, bootstrap_resamples=100, no_plots=True)
        analyze(args)
        with (root / "analysis/functional_layer_summary.csv").open() as handle:
            row = next(row for row in csv.DictReader(handle) if row["metric"] == "delta_loss")
        assert row["n_documents"] == "3" and row["n_observations"] == "9"
        assert math.isclose(float(row["estimate"]), .1, abs_tol=1e-12)
        with (root / "analysis/crosscut_paired_variant_effects.csv").open() as handle:
            paired = {row["comparison"]: row for row in csv.DictReader(handle) if row["metric"] == "srank"}
        for comparison, expected in (("all_one minus native", -1.),
                                     ("all_one_dt minus native_dt", -2.),
                                     ("all_one_dt_with_d minus native_dt_with_d", -3.)):
            assert float(paired[comparison]["estimate"]) == expected
            assert paired[comparison]["n_documents"] == "3"
        assert "all_one_dt_with_d minus native" not in paired
        unmatched = [crosscut[0], next(row for row in crosscut if row["operator_variant"] == "all_one_dt_with_d")]
        unmatched_summary, _, missing = paired_crosscut(unmatched, ("srank",), args)
        assert not unmatched_summary and len(missing) == 1
        assert missing[0]["missing_baseline"] == "native_dt_with_d"
        quality = json.loads((root / "analysis/data_quality.json").read_text())
        assert "value_output_to_input_gain" not in quality["crosscut"]["metrics"]
        jsonl_write(root / "duplicates.jsonl", [functional[0], functional[0]])
        audit = {}
        assert len(load_rows([str(root / "duplicates.jsonl")], "functional", audit)) == 1
        assert audit["functional"]["exact_duplicates_removed"] == 1
        conflicting = dict(functional[0], intervention_loss=3.)
        jsonl_write(root / "duplicates.jsonl", [functional[0], conflicting])
        try:
            load_rows([str(root / "duplicates.jsonl")], "functional", {})
        except AnalysisError:
            pass
        else:
            raise AssertionError("Conflicting duplicates were silently accepted")
    print("self-test passed: document clustering, convention-matched paired effects, missing baselines, null metrics, duplicate protection, correlations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--functional-dir", nargs="*", default=[], help="JSONL files or directories; accepts multiple paths")
    analyze_parser.add_argument("--crosscut-dir", nargs="*", default=[], help="JSONL files or directories; accepts multiple paths")
    analyze_parser.add_argument("--output-dir", required=True)
    analyze_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    analyze_parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAPS)
    analyze_parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        if not args.functional_dir and not args.crosscut_dir:
            parser.error("Provide at least one functional or cross-cut input")
        if args.bootstrap_resamples < 100:
            parser.error("Use at least 100 bootstrap resamples (default: 2000)")
        quality = analyze(args)
        print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "functional_rows": quality["functional"]["unique_rows"],
                          "crosscut_rows": quality["crosscut"]["unique_rows"], "figures": len(quality["figures"])}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except AnalysisError as exc:
        print(f"analysis error: {exc}", file=sys.stderr)
        sys.exit(2)
