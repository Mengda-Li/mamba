"""Plot compact frame-geometry aggregates without Torch or pandas."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import re


def read(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def groups(rows, keys):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = args.output or args.input / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = read(args.input / "summary.csv.gz")
    hist = read(args.input / "histogram_summary.csv.gz")
    created = []
    metrics = (("histogram_kl_uniform", "Histogram KL to uniform"), ("ks", "KS distance statistic"),
               ("angular_w1_midpoint_grid", "Angular W1 to uniform grid (radians)"))
    def save(fig, name):
        for extension in ("pdf", "png"):
            path = out / (name + "." + extension)
            fig.savefig(path, dpi=170, bbox_inches="tight")
            created.append(path.name)
        plt.close(fig)
    for key, values in groups(rows, ("run_id", "model", "layer")).items():
        identifier = re.sub(r"[^\w.-]", "_", f"{key[1]}_layer{key[2]}_{key[0][:8]}")
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 10), squeeze=False)
        handles = {}
        counts = []
        for j, branch in enumerate(("X", "Q", "K")):
            for i, (metric, ylabel) in enumerate(metrics):
                ax = axes[i, j]
                selected = [r for r in values if r["branch"] == branch and r["metric"] == metric]
                if not selected:
                    ax.text(.5, .5, "Not measured/completed", ha="center", transform=ax.transAxes)
                    continue
                for sign, color, label in (("solver_native", "#2266aa", "Solver-native signs (primary)"),
                    ("canonical", "#228855", "Canonical signs (sensitivity)"),
                    ("random_sign_sensitivity", "#aa5599", "Fixed random signs (sensitivity)")):
                    series = sorted([r for r in selected if r["sign_convention"] == sign], key=lambda r: int(r["length"]))
                    if not series:
                        continue
                    x = [int(r["length"]) for r in series]
                    y, lo, hi = [[number(r[field]) for r in series] for field in ("estimate", "ci_low", "ci_high")]
                    line, = ax.plot(x, y, color=color, linestyle="-" if sign == "solver_native" else ":", linewidth=1.4, label=label)
                    handles[label] = line
                    if sign == "solver_native":
                        ax.fill_between(x, lo, hi, color=color, alpha=.12)
                        for xx, yy, row in zip(x, y, series):
                            ax.scatter([xx], [yy], s=24, edgecolors=color,
                                facecolors="none" if number(row["structural_completion_columns"]) else color,
                                marker="s" if row["cohort_status"] != "complete_planned_cohort" else "o")
                            counts.append(int(row["n_documents"]))
                        for prefix, nullcolor, nulllabel in (("gaussian", "#555555", "Processed Gaussian null"),
                            ("haar_processed", "#c47a12", "Processed Haar-frame null")):
                            if any(r.get(prefix + "_mean") for r in series):
                                null, = ax.plot(x, [number(r.get(prefix + "_mean")) for r in series],
                                    color=nullcolor, linestyle="--", linewidth=1, label=nulllabel)
                                handles[nulllabel] = null
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(x, [str(v) for v in x], rotation=30)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.18)
                if i == 0:
                    suffix = " (C branch)" if branch == "Q" and key[1].startswith("mamba") else " (B branch)" if branch == "K" and key[1].startswith("mamba") else ""
                    ax.set_title(branch + suffix)
                if i == 2:
                    ax.set_xlabel("Context tokens")
        cohort = f"n={min(counts)}–{max(counts)} documents" if counts else "No completed angle cohorts"
        fig.suptitle(f"{key[1]} · layer {key[2]} · {cohort}\nMean distances to uniform for separate X/Q/K singular factors")
        if handles:
            fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, fontsize=8)
        fig.text(.5, .065, "Primary bands:95% document bootstrap. Open markers: structural null completion; squares:partial cohort.\nDashed nulls are processed Monte Carlo means. Sign sensitivities are reported without selecting a preferred result.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, .94))
        save(fig, "frame_uniform_distances_" + identifier)
        selected_hist = [r for r in hist if r["run_id"] == key[0] and r["model"] == key[1]
                         and r["layer"] == key[2] and r["sign_convention"] == "solver_native" and r["branch"] in ("X", "Q", "K")]
        if selected_hist:
            length = max(int(r["length"]) for r in selected_hist)
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
            for ax, branch in zip(axes, ("X", "Q", "K")):
                values_hist = sorted([r for r in selected_hist if r["branch"] == branch and int(r["length"]) == length], key=lambda r: int(r["bin"]))
                if not values_hist:
                    ax.text(.5, .5, "Not measured", ha="center", transform=ax.transAxes)
                    continue
                bins = int(values_hist[0]["bins"])
                x = (np.arange(bins) + .5) * 2 * math.pi / bins
                ax.bar(x, [number(r["mean_probability"]) for r in values_hist], width=2 * math.pi / bins, alpha=.75)
                ax.axhline(1 / bins, color="black", linestyle="--", label="Uniform bin mass")
                completion = values_hist[0]["structural_completion_columns"]
                ax.set_title(f"{branch} · completion columns={completion}")
                ax.set_xlabel("Eigen-angle, radians [0,2π)")
                ax.set_ylabel("Mean probability per bin")
            fig.suptitle(f"{key[1]} · layer {key[2]} · context {length} · solver-native signs")
            fig.tight_layout()
            save(fig, "frame_angle_histograms_" + identifier)
    (out / "plot_manifest.json").write_text(json.dumps(dict(files=created,
        source_manifest=str(args.input / "manifest.json"), scope="aggregate replot; no raw activations or bootstrap changes"), indent=2) + "\n")
    print(json.dumps(dict(output=str(out), figures=len(created))))


if __name__ == "__main__":
    main()
