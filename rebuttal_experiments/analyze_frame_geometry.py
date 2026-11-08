"""Aggregate completed frame-geometry shards by document, without Torch/pandas.

Primary X/Q/K curves retain solver-native signs. Mamba heads sharing one B/C
group are counted once, and configurations/source versions are never pooled.
Saved per-document CSVs support bootstrap reproduction; summary CSVs support
compact plotting while a pilot is running. Missing cohorts stay explicit.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import shutil

try:
    import numpy as np
except ImportError:
    np = None

ANGLE_METRICS = ("histogram_kl_uniform", "ks", "angular_w1_midpoint_grid", "circular_kuiper",
                 "cvm", "circular_watson_u2", "fourier_moment_abs_1", "fourier_moment_abs_2", "fourier_moment_abs_4")
GROUP = ("run_id", "model", "layer", "branch", "length", "sign_convention", "active_rank", "space", "metric")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    return sum(values) / len(values) if values else None


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def write_gz(path, rows):
    fields = sorted({key for row in rows for key in row})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(gzip.compress(output.getvalue().encode(), compresslevel=9, mtime=0))


def load_completed(roots, config_ids):
    paths = sorted({path.resolve() for root in roots for path in root.rglob("shards/*.jsonl")})
    rows, inventory, pending = [], [], []
    protocols = {}
    for path in paths:
        config = path.parent.parent
        run_id = config.name.removeprefix("config-")
        if config_ids and run_id not in config_ids:
            continue
        marker = config / "completed" / (path.stem + ".json")
        if not marker.exists():
            pending.append(str(path))
            continue
        stored = json.loads(marker.read_text())
        actual = sha(path)
        if stored.get("sha256") != actual or stored.get("run_id") != run_id:
            raise ValueError(f"Completed-shard checksum/run mismatch: {path}")
        if (config / "protocol.json").exists():
            protocols[run_id] = json.loads((config / "protocol.json").read_text())
        inventory.append(dict(path=str(path), sha256=actual, marker_sha256=sha(marker)))
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows, inventory, pending, protocols


def scientific_unit(row):
    # The same post-convolution B/C factors are shared within a Mamba group.
    if "factor_group" in row:
        return "group:" + str(row["factor_group"])
    return "head:" + str(row.get("head", "X"))


def deduplicate(rows):
    unique = {}
    removed = 0
    for row in sorted(rows, key=lambda r: (-1 if r.get("head") is None else r["head"])):
        key = (row["run_id"], row["document_id"], row["layer"], row["length"],
               row.get("branch", row.get("pair", "")), row.get("space", ""),
               row["diagnostic"], row.get("active_rank"), scientific_unit(row))
        if key in unique:
            removed += 1
            continue
        unique[key] = row
    return list(unique.values()), removed


def flatten(rows):
    observations, histograms, coverage = [], [], []
    for row in rows:
        base = dict(run_id=row["run_id"], model=row["model"], document_id=row["document_id"],
            layer=row["layer"], branch=row.get("branch", row.get("pair", "")), length=row["length"],
            active_rank=row.get("active_rank", ""), space=row.get("space", "token_left"),
            structural_completion_columns=row.get("structural_completion_columns", 0),
            numerical_nullspace_columns=row.get("numerical_nullspace_columns", 0),
            completion_dominates=int(row.get("completion_dominates", False)),
            factor_width=row.get("factor_width", ""), unit=scientific_unit(row))
        coverage.append({**base, "diagnostic": row["diagnostic"], "status": row.get("status", "ok")})
        if row.get("status") != "ok":
            continue
        if row["diagnostic"] == "square_u_eigenangles":
            for sign, metrics in row["angle_metrics"].items():
                for metric in ANGLE_METRICS:
                    if metric not in metrics or not finite(metrics[metric]):
                        continue
                    observation = {**base, "sign_convention": sign, "metric": metric, "value": metrics[metric]}
                    for family, comparison in row.get("processed_null_comparisons", {}).items():
                        null = comparison.get(sign, {}).get(metric, {})
                        prefix = "gaussian" if family == "isotropic_gaussian_factor" else "haar_processed"
                        for source, target in (("null_mean", "mean"), ("null_q05", "q05"), ("null_q95", "q95")):
                            if finite(null.get(source)):
                                observation[prefix + "_" + target] = null[source]
                    observations.append(observation)
                histograms.append({**base, "sign_convention": sign, "counts": metrics["histogram_counts"],
                    "angle_interval": row.get("angle_histogram_interval", [0, 2 * math.pi])})
        elif row["diagnostic"] in ("thin_frame", "qk_subspace_and_basis_alignment"):
            for metric, value in row.get("metrics", {}).items():
                if not finite(value):
                    continue
                observation = {**base, "sign_convention": "subspace_or_sign_invariant", "metric": metric, "value": value}
                null = row.get("haar_stiefel_null", row.get("independent_haar_null", {})).get(metric, {})
                for source, target in (("null_mean", "mean"), ("null_q05", "q05"), ("null_q95", "q95")):
                    if finite(null.get(source)):
                        observation["haar_reference_" + target] = null[source]
                observations.append(observation)
    return observations, histograms, coverage


def document_means(observations):
    cells = {}
    for row in observations:
        key = tuple(row.get(k, "") for k in GROUP) + (row["document_id"],)
        cells.setdefault(key, []).append(row)
    result = []
    for key, values in cells.items():
        row = {**dict(zip((*GROUP, "document_id"), key)), "estimate": mean([v["value"] for v in values]),
            "n_unique_factor_units": len(values),
            "structural_completion_columns": max(v["structural_completion_columns"] for v in values),
            "numerical_nullspace_columns": max(v["numerical_nullspace_columns"] for v in values),
            "completion_dominates": max(v["completion_dominates"] for v in values)}
        for field in sorted({k for v in values for k in v if k.startswith(("gaussian_", "haar_"))}):
            row[field] = mean([v[field] for v in values if field in v])
        result.append(row)
    return result


def bootstrap_summaries(documents, args):
    cells, index_cache = {}, {}
    for row in documents:
        cells.setdefault(tuple(row.get(k, "") for k in GROUP), []).append(row)
    output = []
    for key, values in sorted(cells.items(), key=lambda item: str(item[0])):
        values.sort(key=lambda r: r["document_id"])
        cohort = tuple(v["document_id"] for v in values)
        if cohort not in index_cache:
            rng = random.Random(args.seed ^ int(fingerprint(cohort)[:12], 16))
            indices = [[rng.randrange(len(values)) for _ in values] for _ in range(args.bootstrap_resamples)]
            index_cache[cohort] = np.asarray(indices) if np is not None else indices
        estimates = [v["estimate"] for v in values]
        indices = index_cache[cohort]
        samples = np.asarray(estimates)[indices].mean(axis=1).tolist() if np is not None else [mean([estimates[i] for i in draw]) for draw in indices]
        row = {**dict(zip(GROUP, key)), "estimate": mean(estimates), "ci_low": quantile(samples, .025),
            "ci_high": quantile(samples, .975), "n_documents": len(values),
            "expected_documents": args.expected_documents,
            "cohort_status": "complete_planned_cohort" if len(values) == args.expected_documents else "partial_cohort",
            "cohort_sha256": fingerprint(cohort),
            "structural_completion_columns": max(v["structural_completion_columns"] for v in values),
            "numerical_nullspace_columns": max(v["numerical_nullspace_columns"] for v in values),
            "completion_dominates": max(v["completion_dominates"] for v in values),
            "min_unique_factor_units_per_document": min(v["n_unique_factor_units"] for v in values),
            "max_unique_factor_units_per_document": max(v["n_unique_factor_units"] for v in values)}
        for field in sorted({k for v in values for k in v if k.startswith(("gaussian_", "haar_"))}):
            row[field] = mean([v[field] for v in values if field in v])
        output.append(row)
    return output


def histogram_tables(histograms):
    keys = ("run_id", "model", "layer", "branch", "length", "sign_convention")
    document_groups = {}
    for row in histograms:
        document_groups.setdefault(tuple(row[k] for k in (*keys, "document_id")), []).append(row)
    documents = []
    for key, rows in document_groups.items():
        counts = [sum(r["counts"][i] for r in rows) for i in range(len(rows[0]["counts"]))]
        total = sum(counts)
        for i, count in enumerate(counts):
            documents.append({**dict(zip((*keys, "document_id"), key)), "bin": i, "bins": len(counts),
                "count": count, "total_angles": total, "probability": count / total,
                "n_unique_factor_units": len(rows),
                "structural_completion_columns": max(r["structural_completion_columns"] for r in rows)})
    grouped = {}
    for row in documents:
        grouped.setdefault(tuple(row[k] for k in (*keys, "bin")), []).append(row)
    summary = [{**dict(zip((*keys, "bin"), key)), "bins": rows[0]["bins"],
        "mean_probability": mean([r["probability"] for r in rows]), "n_documents": len(rows),
        "pooled_angle_count": sum(r["count"] for r in rows),
        "structural_completion_columns": max(r["structural_completion_columns"] for r in rows)}
        for key, rows in grouped.items()]
    return documents, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--config-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-documents", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100 or args.expected_documents < 1:
        parser.error("Require at least100 bootstrap resamples and positive expected documents")
    raw, inventory, pending, protocols = load_completed(args.input_root, set(args.config_id))
    rows, removed = deduplicate(raw)
    observations, histograms, coverage = flatten(rows)
    documents = document_means(observations)
    summary = bootstrap_summaries(documents, args)
    hist_docs, hist_summary = histogram_tables(histograms)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("document_metrics.csv.gz", documents), ("summary.csv.gz", summary),
        ("document_histograms.csv.gz", hist_docs), ("histogram_summary.csv.gz", hist_summary),
        ("coverage.csv.gz", coverage)):
        write_gz(out / name, data)
    shutil.copy2(Path(__file__).with_name("plot_frame_geometry.py"), out / "plot_frame_geometry.py")
    models = sorted({r["model"] for r in rows})
    report = dict(schema_version=1, input_completed_shards=len(inventory), raw_rows=len(raw),
        unique_factor_rows=len(rows), duplicated_head_group_or_exact_rows_removed=removed,
        models=models, source_protocols=protocols, source_files=inventory,
        incomplete_shards_ignored=pending, expected_documents=args.expected_documents,
        bootstrap_resamples=args.bootstrap_resamples, seed=args.seed, confidence_level=.95,
        aggregation="unique factor units averaged within document; equal-weight document means and percentile bootstrap",
        primary_sign_convention="solver_native", partial_cohorts_present=any(r["cohort_status"] == "partial_cohort" for r in summary),
        null_bands="mean of matched-null 5–95% Monte Carlo reference quantiles, not data confidence intervals",
        w1_definition="angular W1 to deterministic uniform midpoint grid; differs from an independent uniform-sample reference",
        script_sha256=sha(Path(__file__)), plotting_script_sha256=sha(Path(__file__).with_name("plot_frame_geometry.py")),
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in out.glob("*.csv.gz")})
    (out / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text("""# Frame geometry aggregates

The X, Q and K branches are separate measurements; for Mamba Q=C and K=B.
Primary curves retain solver-native signs. Canonical and fixed randomized
signs are sensitivity analyses, not alternatives selected to improve fit.
Model/source configurations are separated. Within each document, shared
Mamba B/C groups are counted once, then unique factor units are averaged.
The means and 95% intervals use equal-weight document bootstrap (2,000
resamples by default); actual settings and planned/observed cohort sizes are
in manifest.json and summary.csv.gz. Partial cohorts are labeled explicitly.

`document_metrics.csv.gz` retains the compact per-document estimates used for
bootstrap. Histograms retain counts/probabilities, not matrices or raw angles.
Matched-null ranges are Monte Carlo references, not document confidence
intervals or iid-angle hypothesis tests. Completion columns are recorded
explicitly, including GPT-2 X at1024 and tall Q/K or C/B factors.

Replot using NumPy/Matplotlib (no Torch/pandas/checkpoints):

```bash
python plot_frame_geometry.py --input . --output figures
```
""")
    print(json.dumps(dict(output=str(out), completed_shards=len(inventory), document_rows=len(documents),
        summary_rows=len(summary), models=models, partial_cohorts=report["partial_cohorts_present"])))


if __name__ == "__main__":
    main()
