"""Fixed prespecified document-cohort eigen-angle figures for the revised PDF.

Reads compact per-document geometry tables, recomputes document-bootstrap
intervals, and writes one6.6x6.1-inch primary-only figure per model. No Torch.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from analyze_frame_geometry import bootstrap_summaries


METRICS = (("histogram_kl_uniform", "KL divergence"), ("ks", "KS distance"),
           ("angular_w1_midpoint_grid", "Angular W1 (rad)"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_gz(path, rows):
    fields = sorted({key for row in rows for key in row})
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--frozen-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--documents", type=int, default=8)
    p.add_argument("--lengths", default="64,128,256,512,1024")
    p.add_argument("--seed", type=int, default=20261108)
    p.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frozen = [json.loads(line) for line in args.frozen_manifest.read_text().splitlines() if line.strip()]
    cohort = sorted({row["document_id"] for row in frozen})[:args.documents]
    if len(cohort) != args.documents:
        raise ValueError("Frozen manifest has fewer than the requested documents")
    lengths = sorted({int(v) for v in args.lengths.split(",")})
    with gzip.open(args.input / "document_metrics.csv.gz", "rt") as stream:
        source = list(csv.DictReader(stream))
    selected = [r for r in source if r["document_id"] in cohort and r["sign_convention"] == "solver_native"
        and r["branch"] in ("X", "Q", "K") and r["metric"] in {m for m, _ in METRICS}
        and int(r["length"]) in lengths]
    if not selected:
        raise ValueError("No matching completed native-sign geometry measurements")
    for row in selected:
        for key in [k for k in row if k == "estimate" or k.startswith(("gaussian_", "haar_"))]:
            if row.get(key):
                row[key] = float(row[key])
            else:
                row.pop(key, None)
        for key in ("n_unique_factor_units", "structural_completion_columns", "numerical_nullspace_columns", "completion_dominates"):
            row[key] = int(row[key])
    models = sorted({(r["run_id"], r["model"]) for r in selected})
    for run_id, model in models:
        layers = sorted({r["layer"] for r in selected if r["run_id"] == run_id and r["model"] == model}, key=int)
        for layer in layers:
            for branch in ("X", "Q", "K"):
                for length in lengths:
                    for metric, _ in METRICS:
                        found = [r for r in selected if (r["run_id"], r["model"], r["layer"], r["branch"], int(r["length"]), r["metric"])
                            == (run_id, model, layer, branch, length, metric)]
                        if {r["document_id"] for r in found} != set(cohort) or len(found) != len(cohort):
                            raise ValueError(f"Prespecified cohort incomplete: {model}/{layer}/{branch}/L{length}/{metric}")
    summary = bootstrap_summaries(selected, SimpleNamespace(seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples, expected_documents=args.documents))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_gz(out / "publication_summary.csv.gz", summary)
    write_gz(out / "publication_document_metrics.csv.gz", selected)
    created = []
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    for run_id, model in models:
        rows = [r for r in summary if r["run_id"] == run_id and r["model"] == model]
        layers = sorted({r["layer"] for r in rows}, key=int)
        fig, axes = plt.subplots(3, 3, figsize=(6.6, 6.1))
        legend = {}
        for col, branch in enumerate(("X", "Q", "K")):
            for j, (metric, ylabel) in enumerate(METRICS):
                ax = axes[j, col]
                chosen = [r for r in rows if r["branch"] == branch and r["metric"] == metric]
                for layer, color in zip(layers, ("#2768ad", "#cf6c24")):
                    series = sorted([r for r in chosen if r["layer"] == layer], key=lambda r: int(r["length"]))
                    x = [int(r["length"]) for r in series]
                    y, lo, hi = [[float(r[field]) for r in series] for field in ("estimate", "ci_low", "ci_high")]
                    label = f"Layer {layer}"
                    line, = ax.plot(x, y, color=color, linewidth=1.15, label=label)
                    legend[label] = line
                    ax.fill_between(x, lo, hi, color=color, alpha=.15, linewidth=0)
                    for xx, yy, row in zip(x, y, series):
                        ax.plot(xx, yy, "o", color=color, markersize=3.3, markeredgewidth=.8,
                                markerfacecolor="white" if row["structural_completion_columns"] else color)
                # Processed null is shape-dependent and equal for the two layers.
                null = []
                for length in lengths:
                    values = [float(r["gaussian_mean"]) for r in chosen if int(r["length"]) == length and r.get("gaussian_mean") is not None]
                    null.append(float(np.mean(values)) if values else np.nan)
                line, = ax.plot(lengths, null, "--", color="#222222", linewidth=.9, label="Matched Gaussian null")
                legend["Matched Gaussian null"] = line
                ax.set_xscale("log", base=2)
                ax.set_xticks(lengths, [str(v) for v in lengths], rotation=35)
                ax.grid(alpha=.16, linewidth=.4)
                ax.tick_params(axis="both", pad=1.8)
                ax.set_ylabel(ylabel if col == 0 else "")
                if j == 0:
                    title = branch
                    if model.startswith("mamba") and branch in ("Q", "K"):
                        title += " = " + ("C" if branch == "Q" else "B")
                    ax.set_title(title)
                if j == 2:
                    ax.set_xlabel("Sequence length")
        model_label = {"gpt2": "GPT-2", "mamba2": "Mamba-Codestral-7B", "mamba13": "Mamba2-1.3B", "retnet": "RetNet-1.3B"}.get(model, model)
        fig.suptitle(f"{model_label}: eigen-angle distances to uniform (fixed {args.documents} documents)", fontsize=9, y=.995)
        fig.legend(legend.values(), legend.keys(), loc="upper center", bbox_to_anchor=(.5, .965), ncol=3, frameon=False)
        fig.text(.5, .006, "Solver-native signs. Shading:95% document bootstrap. Open markers: structural SVD completion.", ha="center", fontsize=6.5)
        fig.subplots_adjust(left=.105, right=.99, bottom=.09, top=.89, hspace=.42, wspace=.30)
        for suffix in ("pdf", "svg", "png"):
            path = out / f"frame_primary_{model}_{run_id[:8]}.{suffix}"
            fig.savefig(path, dpi=220)
            created.append(path.name)
        plt.close(fig)
    manifest = dict(document_ids=cohort, cohort_sha256=hashlib.sha256("\n".join(cohort).encode()).hexdigest(),
        frozen_manifest_sha256=sha(args.frozen_manifest), source_analysis_manifest_sha256=sha(args.input / "manifest.json"),
        source_document_metrics_sha256=sha(args.input / "document_metrics.csv.gz"),
        models=models, layers="early and final as recorded", lengths=lengths, sign_convention="solver_native",
        documents=args.documents, bootstrap_resamples=args.bootstrap_resamples, seed=args.seed,
        w1_definition="angular distance to deterministic uniform midpoint grid", script_sha256=sha(Path(__file__)),
        complete_selected_cohort_verified=True, figure_size_inches=[6.6, 6.1], files=created)
    (out / "publication_cohort.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=str(out), cohort_sha256=manifest["cohort_sha256"], files=created)))


if __name__ == "__main__":
    main()
