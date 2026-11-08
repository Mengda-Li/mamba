"""Build a small, offline plotting bundle from existing aggregate CSVs.

Standard-library only. Does not load checkpoints/raw JSONL or recompute CIs.
The original experiment results are read-only. Rerun after updated analyses.
"""
from __future__ import annotations
from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import zipfile

ROOT = REPO_ROOT
STATS = ("estimate", "ci_low", "ci_high", "n_documents", "n_available_documents", "n_observations")
DROP = {"document_ids", "median", "min", "max", "n_positive_documents", "n_negative_documents"}
EXCLUDED_PREFIXES = ("tolerance_ess.", "relative_tolerance_rank.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_gzip_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(gzip.compress(stream.getvalue().encode(), compresslevel=9, mtime=0))


def compact(source, destination):
    raw = read_csv(source)
    rows = [r for r in raw if not r.get("metric", "").startswith(EXCLUDED_PREFIXES)]
    fields = [k for k in raw[0] if k not in DROP and not k.endswith("_path")] if raw else []
    write_gzip_csv(destination, [{k: r[k] for k in fields} for r in rows], fields)
    return dict(source_sha256=sha(source), source_bytes=source.stat().st_size,
                source_rows=len(raw), cached_rows=len(rows), bytes=destination.stat().st_size,
                excluded_metrics=sorted({r.get("metric") for r in raw
                                         if r.get("metric", "").startswith(EXCLUDED_PREFIXES)}))


def scatter_points(source):
    groups = {}
    dimensions = ("run_id", "checkpoint", "corpus", "input_condition", "length", "layer", "intervention", "head_or_channel")
    for r in read_csv(source):
        if r["metric"] in {"sink_D_over_S", "delta_loss"}:
            groups.setdefault(tuple(r[k] for k in dimensions), {})[r["metric"]] = r
    points, omitted = [], 0
    for key, values in groups.items():
        x, y = values.get("sink_D_over_S"), values.get("delta_loss")
        if not x or not y:
            omitted += 1
            continue
        xdocs, ydocs = sorted(x["document_ids"].split(";")), sorted(y["document_ids"].split(";"))
        if xdocs != ydocs or len(xdocs) != 32 or int(x["n_documents"]) != 32 or int(y["n_documents"]) != 32:
            omitted += 1
            continue
        points.append({**dict(zip(dimensions, key)), "n_documents": 32,
            "cohort_sha256": hashlib.sha256("\n".join(xdocs).encode()).hexdigest(),
            "mean_sink_D_over_S": x["estimate"], "mean_delta_loss": y["estimate"],
            "delta_ci_low": y["ci_low"], "delta_ci_high": y["ci_high"]})
    return points, omitted


def source_archive(experiments, destination):
    files = (list(experiments.glob("*.py")) + list(experiments.glob("*.md"))
             + list(experiments.glob("*plan*.json")) + list((experiments / "historical_plans").glob("*.json"))
             + list((experiments / "historical_plans").glob("*.md")))
    legacy = REPO_ROOT
    files += [legacy / name for name in ("experiment_common.py", "gpt2_experiments.py",
        "compare_attention_ranks_wikitext.py", "extract_mamba_matrices.py",
        "run_structured_experiments.py", "validate_reconstruction.py")]
    inventory = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(set(files)):
            if not path.is_file():
                continue
            # Exclude transport/credential helpers; archive only explicit local code.
            if any(word in path.name.lower() for word in ("ssh", "credential", "server_info")):
                continue
            content = path.read_bytes()
            if b"".join((b"-----BEGIN ", b"OPENSSH PRIVATE KEY-----")) in content:
                continue
            if path.suffix == ".py":
                tree = ast.parse(content)
                if any((isinstance(n, ast.Import) and any(a.name.split(".")[0] == "paramiko" for a in n.names))
                       or (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "paramiko")
                       for n in ast.walk(tree)):
                    continue
            name = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
            inventory.append(dict(path=name, sha256=sha(path), bytes=len(content)))
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ARTIFACT_ROOT / "results")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "local_experiment_bundle")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data = out / "data"
    data.mkdir(exist_ok=True)
    inputs = {
        "functional_layer.csv.gz": args.results / "functional_analysis/functional_layer_summary.csv",
        "functional_head.csv.gz": args.results / "functional_analysis/functional_head_summary.csv",
        "functional_coverage.csv.gz": args.results / "functional_analysis/functional_coverage.csv",
        "functional_correlations.csv.gz": args.results / "functional_analysis/functional_correlations.csv",
        "crosscut_layer.csv.gz": args.results / "crossmodel_analysis/crosscut_layer_summary.csv",
        "crosscut_paired.csv.gz": args.results / "crossmodel_analysis/crosscut_paired_variant_effects.csv",
        "crosscut_coverage.csv.gz": args.results / "crossmodel_analysis/crosscut_coverage.csv",
    }
    inventory = {}
    for filename, source in inputs.items():
        if source.exists():
            inventory[filename] = compact(source, data / filename)
            inventory[filename]["source"] = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name
    points, omitted = scatter_points(inputs["functional_head.csv.gz"])
    fields = list(points[0]) if points else ["run_id", "checkpoint", "length", "layer", "intervention", "head_or_channel",
        "n_documents", "cohort_sha256", "mean_sink_D_over_S", "mean_delta_loss", "delta_ci_low", "delta_ci_high"]
    write_gzip_csv(data / "functional_scatter.csv.gz", points, fields)
    for filename in ("native_mixer_heatmaps.npz", "native_mixer_heatmaps.json"):
        source = args.results / "heatmaps" / filename
        if source.exists():
            shutil.copy2(source, data / filename)
            inventory[filename] = dict(source_sha256=sha(source), source_bytes=source.stat().st_size)
    quality = {}
    for name in ("functional_analysis", "crossmodel_analysis"):
        path = args.results / name / "data_quality.json"
        if path.exists():
            raw = json.loads(path.read_text())
            quality[name] = {k: raw.get(k) for k in ("aggregation", "bootstrap_resamples", "seed", "confidence_level")}
            quality[name]["source_sha256"] = sha(path)
    coverage = {name: read_csv(source) for name, source in inputs.items() if "coverage" in name and source.exists()}
    # Coverage is compact; it contains no document text or repeated document lists.
    write_json(out / "coverage.json", coverage)
    plotter = Path(__file__).with_name("plot_local_bundle.py")
    shutil.copy2(plotter, out / "plot_local_bundle.py")
    archived = source_archive(Path(__file__).parent, out / "code.zip")
    write_json(out / "code_inventory.json", archived)
    readme = """# Local plotting bundle

Run from the repository root (Python with NumPy and Matplotlib; no pandas/GPU):

```bash
python rebuttal_artifacts/local_experiment_bundle/plot_local_bundle.py --output rebuttal_artifacts/local_experiment_bundle/figures
```

The archived current code and historical cached measurements have separate
provenance. A new source archive does not relabel v1 measurements as corrected
v2 results. Rebuild the data cache only after corrected aggregate tables exist.

The gzip CSVs contain saved means, document-bootstrap 95% intervals and sample
counts for all retained conditions, context lengths and operator variants.
They support replotting these aggregate estimates; they do **not** contain the
raw observations required to rerun bootstrap, change cohorts or estimate new
statistics. `code.zip` preserves the new experiment code, required legacy
helpers and explicit run plans, without credentials or transport scripts.

Aggregation: average the separately applied head interventions/observations
within each document, then weight documents equally; 2,000 percentile
bootstrap resamples, seed 20261108. Head scatter points use matching complete
32-document cohorts; their x/y means are descriptive, not causal estimates.
The 2048-token groups can contain fewer than32 documents; the cached counts
and coverage report are authoritative. Missing groups are never filled in.

Mamba `native` is canonical M; `native_dt` maps pre-discretized values via
M diag(dt); `native_dt_with_d` additionally includes direct feedthrough D.
Their all-one controls keep the corresponding boundary fixed. They are
distinct operators, and the plot labels retain these conventions.

Exact/relative threshold counts are excluded from the main plot cache while
precision checks for tiny spectral tails are pending. Leading stable/entropy
ranks, operator/Frobenius norms, amplitudes and value gains remain cached.
No precision-sensitive tail-count claim follows from these figures.

Heatmaps store the predetermined first submitted document/head0 at the last
GPT-2 and Codestral layers, with independent raw-entry color scales and
different tokenization. See the accompanying JSON for selection and boundary.

Raw observations/checkpoints/logs remain on the server under
`/225040471/mamba/rebuttal_20261108/`; the source analyses and hashes are in
`manifest.json`. Rebuild after additional model analyses arrive:

```bash
python rebuttal_experiments/build_local_bundle.py --results /path/to/aggregate/results
```
"""
    (out / "README.md").write_text(readme)
    files = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in out.rglob("*") if p.is_file() and p.name != "manifest.json" and "figures" not in p.parts}
    manifest = dict(schema_version=1, source_aggregate_tables=inventory, analysis_protocols=quality,
        sources_zip_entries=len(archived), files=files, bundle_bytes_without_manifest=sum(v["bytes"] for v in files.values()),
        scatter_points=len(points), scatter_omitted_nonmatching_or_incomplete_cohorts=omitted,
        excluded_metric_prefixes=list(EXCLUDED_PREFIXES), precision_qc="tiny-tail absolute/relative counts pending; excluded",
        archived_code_protocol="v2 explicit gesvd and entry energy; cached source tables may remain historical v1",
        raw_server_root="/225040471/mamba/rebuttal_20261108/", scope="aggregate replotting; cannot rebootstrap raw data",
        builder_sha256=sha(Path(__file__)), plotter_sha256=sha(plotter))
    write_json(out / "manifest.json", manifest)
    # Verify integrity using only stdlib after writing the complete bundle.
    for name, expected in files.items():
        assert sha(out / name) == expected["sha256"]
    with zipfile.ZipFile(out / "code.zip") as archive:
        assert archive.testzip() is None
    print(json.dumps(dict(output=str(out), cached_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
                         scatter_points=len(points), source_code_files=len(archived)), sort_keys=True))


if __name__ == "__main__":
    main()
