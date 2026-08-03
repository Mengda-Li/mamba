#!/usr/bin/env python3
"""Validate and aggregate the Wikipedia Mamba-2/RetNet replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LENGTHS = (64, 128, 256, 384, 512, 768, 1024)
SLOPE_LENGTHS = (256, 384, 512, 768, 1024)
METRICS = ("srank", "erank_1", "erank_2")
MODELS = {
    "mamba2": {"layer": 63, "heads": 8, "variants": 5, "label": "Mamba-2"},
    "retnet": {"layer": 23, "heads": 8, "variants": 7, "label": "RetNet"},
}
BOOTSTRAP_SEED = 20270803
BOOTSTRAP_RESAMPLES = 2000


def bootstrap_indices(n: int) -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, n, size=(BOOTSTRAP_RESAMPLES, n), endpoint=False
    )


def fit_slopes(curves: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    log_x = np.log(lengths)
    centered_x = log_x - log_x.mean()
    log_y = np.log(curves)
    centered_y = log_y - log_y.mean(axis=-1, keepdims=True)
    return (centered_y * centered_x).sum(axis=-1) / np.square(centered_x).sum()


def median_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    samples = np.median(values[bootstrap_indices(values.size)], axis=1)
    return (
        float(np.median(values)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def slope_interval(document_length: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    pivot = document_length.pivot(index="document_id", columns="length", values=metric)
    matrix = pivot[list(SLOPE_LENGTHS)].to_numpy(float)
    if not np.isfinite(matrix).all() or np.any(matrix <= 0):
        raise RuntimeError(f"Nonfinite or nonpositive values in {metric} slope grid")
    lengths = np.asarray(SLOPE_LENGTHS, dtype=float)
    point = float(fit_slopes(np.median(matrix, axis=0, keepdims=True), lengths)[0])
    samples = matrix[bootstrap_indices(matrix.shape[0])]
    replicates = fit_slopes(np.median(samples, axis=1), lengths)
    return point, float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))


def load_and_validate(root: Path) -> pd.DataFrame:
    metadata = json.loads((root / "replication_metadata.json").read_text(encoding="utf-8"))
    source_parquet = metadata.pop("source_parquet", None)
    if metadata != {
        "config": "20231101.en",
        "dataset": "wikimedia/wikipedia",
        "document_count": 32,
        "scope": "Post-plan cross-corpus replication of final-layer WP1 and WP5.",
        "seed": 20270803,
        "selection": "Seeded sample from the first 512 articles with >=1056 tokens under both tokenizers.",
    }:
        raise RuntimeError("Wikipedia replication metadata differs from the frozen protocol")
    if not isinstance(source_parquet, str) or not source_parquet:
        raise RuntimeError("Wikipedia replication metadata is missing source_parquet")
    if not (root / "GPU_RUNS_COMPLETE").exists():
        raise RuntimeError("GPU_RUNS_COMPLETE marker is missing")

    frames = []
    for model, spec in MODELS.items():
        frame = pd.read_parquet(root / f"{model}_metrics.parquet")
        expected = 32 * len(LENGTHS) * spec["heads"] * spec["variants"]
        if len(frame) != expected:
            raise RuntimeError(f"{model}: expected {expected} observations, found {len(frame)}")
        if set(frame["checkpoint"].astype(str)) != {model}:
            raise RuntimeError(f"{model}: checkpoint identity mismatch")
        if set(frame["layer"].astype(int)) != {spec["layer"]}:
            raise RuntimeError(f"{model}: layer identity mismatch")
        if set(frame["length"].astype(int)) != set(LENGTHS):
            raise RuntimeError(f"{model}: length grid mismatch")
        if frame["document_id"].nunique() != 32 or frame["head_or_channel"].nunique() != spec["heads"]:
            raise RuntimeError(f"{model}: document or component grid mismatch")
        key = ["document_id", "length", "layer", "head_or_channel", "operator_variant"]
        if frame.duplicated(key).any():
            raise RuntimeError(f"{model}: duplicate observation keys")
        required = [*METRICS, "operator_norm", "frobenius_norm", "nuclear_norm"]
        values = frame[required].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise RuntimeError(f"{model}: nonfinite or nonpositive spectral metrics")
        frame = frame.copy()
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def aggregate(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict] = []
    slope_rows: list[dict] = []
    effect_rows: list[dict] = []
    for model, spec in MODELS.items():
        native = metrics[(metrics["model"] == model) & (metrics["operator_variant"] == "native")]
        for metric in METRICS:
            doc = native.groupby(["document_id", "length"], observed=True)[metric].median().reset_index()
            for length, group in doc.groupby("length", observed=True):
                estimate, low, high = median_interval(group[metric].to_numpy(float))
                curve_rows.append(
                    {
                        "model": model,
                        "model_label": spec["label"],
                        "metric": metric,
                        "length": int(length),
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "n_documents": 32,
                    }
                )
            slope, low, high = slope_interval(doc[doc["length"].isin(SLOPE_LENGTHS)], metric)
            slope_rows.append(
                {
                    "corpus": "Wikipedia 20231101.en",
                    "model": model,
                    "model_label": spec["label"],
                    "layer": spec["layer"],
                    "metric": metric,
                    "slope": slope,
                    "ci_low": low,
                    "ci_high": high,
                    "n_documents": 32,
                    "n_resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "min_length": min(SLOPE_LENGTHS),
                    "max_length": max(SLOPE_LENGTHS),
                }
            )

        pair_keys = ["document_id", "length", "head_or_channel"]
        all_one = metrics[(metrics["model"] == model) & (metrics["operator_variant"] == "all_one")]
        for metric in METRICS:
            merged = native[pair_keys + [metric]].merge(
                all_one[pair_keys + [metric]],
                on=pair_keys,
                suffixes=("_native", "_all_one"),
                validate="one_to_one",
            )
            merged["effect"] = merged[f"{metric}_all_one"] - merged[f"{metric}_native"]
            for length, group in merged.groupby("length", observed=True):
                doc_effect = group.groupby("document_id", observed=True)["effect"].median().to_numpy(float)
                estimate, low, high = median_interval(doc_effect)
                effect_rows.append(
                    {
                        "model": model,
                        "model_label": spec["label"],
                        "metric": metric,
                        "length": int(length),
                        "effect_all_one_minus_native": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "n_documents": 32,
                        "n_resamples": BOOTSTRAP_RESAMPLES,
                        "seed": BOOTSTRAP_SEED,
                    }
                )
    return pd.DataFrame(curve_rows), pd.DataFrame(slope_rows), pd.DataFrame(effect_rows)


def plot(curves: pd.DataFrame, effects: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"mamba2": "#D55E00", "retnet": "#0072B2"}
    labels = {"srank": "stable rank", "erank_1": r"$\mathrm{erank}_1$", "erank_2": r"$\mathrm{erank}_2$"}
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.5), sharex="col", squeeze=False)
    for column, metric in enumerate(METRICS):
        for model, spec in MODELS.items():
            line = curves[(curves["model"] == model) & (curves["metric"] == metric)].sort_values("length")
            x = line["length"].to_numpy(float)
            axes[0, column].plot(x, line["estimate"], marker="o", ms=2.8, color=colors[model], label=spec["label"])
            axes[0, column].fill_between(x, line["ci_low"], line["ci_high"], color=colors[model], alpha=0.15)
            effect = effects[(effects["model"] == model) & (effects["metric"] == metric)].sort_values("length")
            ex = effect["length"].to_numpy(float)
            axes[1, column].plot(ex, effect["effect_all_one_minus_native"], marker="o", ms=2.8, color=colors[model])
            axes[1, column].fill_between(ex, effect["ci_low"], effect["ci_high"], color=colors[model], alpha=0.15)
        axes[0, column].set_xscale("log", base=2)
        axes[0, column].set_yscale("log")
        axes[0, column].set_title(labels[metric])
        axes[0, column].grid(alpha=0.22, which="both")
        axes[1, column].axhline(0, color="#666666", lw=0.8)
        axes[1, column].set_xscale("log", base=2)
        axes[1, column].set_xlabel(r"length $\ell$")
        axes[1, column].grid(alpha=0.22)
    axes[0, 0].set_ylabel("native rank")
    axes[1, 0].set_ylabel("all-one minus native")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220}), ("svg", {})):
        fig.savefig(output / f"wikipedia_replication.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("experiments_2027_wikipedia_replication"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir / "analysis"
    metrics = load_and_validate(args.input_dir)
    curves, slopes, effects = aggregate(metrics)
    output.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output / "native_curve_summary.csv", index=False)
    slopes.to_csv(output / "native_slope_summary.csv", index=False)
    effects.to_csv(output / "all_one_paired_effects.csv", index=False)
    plot(curves, effects, output)
    metadata = {
        "validated": True,
        "observations": int(len(metrics)),
        "documents_per_model": 32,
        "bootstrap_unit": "document",
        "within_document_component_aggregation": "median",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(slopes.to_string(index=False))
    final_effects = effects[effects["length"] == max(LENGTHS)]
    print("\nAll-one minus native effects at length 1024:\n" + final_effects.to_string(index=False))


if __name__ == "__main__":
    main()
