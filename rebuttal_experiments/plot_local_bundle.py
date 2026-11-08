"""Replot saved aggregate estimates offline; requires NumPy/Matplotlib, not pandas."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


def read(path):
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def groups(rows, keys):
    result = {}
    for row in rows:
        result.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    return result


def label_variant(name):
    return {"native": "native / canonical M", "all_one": "all-one / canonical M",
        "native_dt": "native M diag(dt)", "all_one_dt": "all-one M diag(dt)",
        "native_dt_with_d": "native M diag(dt) + D", "all_one_dt_with_d": "all-one M diag(dt) + D"}.get(name, name)


def draw(ax, rows, label, color=None, linestyle="-", all_lengths=None):
    rows = sorted(rows, key=lambda r: int(r["length"]))
    lookup = {int(r["length"]): r for r in rows}
    xs = sorted(set(all_lengths or lookup))
    ys = [number(lookup[x].get("estimate")) if x in lookup else math.nan for x in xs]
    lo = [number(lookup[x].get("ci_low")) if x in lookup else math.nan for x in xs]
    hi = [number(lookup[x].get("ci_high")) if x in lookup else math.nan for x in xs]
    line, = ax.plot(xs, ys, marker="o", label=label, color=color, linestyle=linestyle, markersize=4)
    ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=.13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("Context tokens")
    ax.grid(alpha=.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.constrained_layout.use"] = True
    out = args.output or args.bundle / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = args.bundle / "data"
    created = []
    def save(fig, name):
        for suffix in ("pdf", "png"):
            path = out / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            created.append(path.name)
        plt.close(fig)
    functional = read(data / "functional_layer.csv.gz")
    for key, rows in groups(functional, ("run_id", "checkpoint", "input_condition")).items():
        layers = sorted({int(r["layer"]) for r in rows})
        lengths = sorted({int(r["length"]) for r in rows})
        fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3.5), squeeze=False)
        for ax, layer in zip(axes[0], layers):
            selected = [r for r in rows if int(r["layer"]) == layer and r["metric"] == "delta_loss"]
            for (mode,), series in groups(selected, ("intervention",)).items():
                draw(ax, series, mode, all_lengths=lengths)
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_title(f"Layer {layer}")
            ax.set_ylabel("Mean paired Δ next-token NLL")
            ax.legend(fontsize=8)
        fig.suptitle(f"{key[1]} · separately applied single-head interventions\nHead mean within document; document mean and 95% bootstrap interval")
        save(fig, "functional_delta_nll_" + key[0][:10])
    crosscut = read(data / "crosscut_layer.csv.gz")
    for key, rows in groups(crosscut, ("run_id", "checkpoint", "input_condition", "layer")).items():
        lengths = sorted({int(r["length"]) for r in rows})
        suffix = re.sub(r"[^\w.-]", "_", f"{key[1]}_layer{key[3]}_{key[0][:8]}")
        full = [r for r in rows if r["region"] == "full" and r["operator_variant"].startswith("native")]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, metric in zip(axes, ("srank", "erank_1", "erank_2")):
            for (variant,), series in groups([r for r in full if r["metric"] == metric], ("operator_variant",)).items():
                draw(ax, series, label_variant(variant), all_lengths=lengths)
            ax.set_ylabel(metric)
            ax.legend(fontsize=7)
        fig.suptitle(f"{key[1]} · layer {key[3]} · full native mixers\nConventions plotted separately; mean and 95% document-bootstrap interval")
        save(fig, "native_full_ranks_" + suffix)
        middle = [r for r in rows if r["region"] == "crosscut" and number(r["cut_fraction"]) == .5
                  and number(r["gap"]) == 0 and (r["operator_variant"].startswith("native") or r["operator_variant"].startswith("all_one"))]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, metric, ylabel in zip(axes, ("value_output_to_input_gain", "frobenius_norm", "erank_1"),
                                       ("Core value-output/input Frobenius gain", "Cross-cut operator Frobenius norm", "Cross-cut entropy effective rank")):
            chosen = [r for r in middle if r["metric"] == metric]
            if not chosen:
                ax.text(.5, .5, "Not measured", ha="center", transform=ax.transAxes)
            for (variant,), series in groups(chosen, ("operator_variant",)).items():
                draw(ax, series, label_variant(variant), linestyle="--" if variant.startswith("all_one") else "-", all_lengths=lengths)
            ax.set_ylabel(ylabel)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6)
        fig.suptitle(f"{key[1]} · layer {key[3]} · strict midpoint past-to-future block, gap=0\nEach curve retains its operator/value boundary; missing lengths are not filled")
        save(fig, "strict_midcut_" + suffix)
    scatter = read(data / "functional_scatter.csv.gz")
    for key, rows in groups(scatter, ("run_id", "checkpoint", "input_condition")).items():
        length = max(int(r["length"]) for r in rows)
        rows = [r for r in rows if int(r["length"]) == length]
        modes = sorted({r["intervention"] for r in rows})
        fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 3.7), squeeze=False)
        for ax, mode in zip(axes[0], modes):
            for (layer,), series in groups([r for r in rows if r["intervention"] == mode], ("layer",)).items():
                ax.scatter([number(r["mean_sink_D_over_S"]) for r in series],
                           [number(r["mean_delta_loss"]) for r in series], label=f"layer {layer}", s=25)
            ax.set_xscale("log")
            ax.set_xlabel("Mean native nonsink/sink energy D/S")
            ax.set_ylabel("Mean paired Δ NLL")
            ax.set_title(mode)
            ax.legend(fontsize=8)
        fig.suptitle(f"{key[1]} · context {length} · each head uses matching 32-document means\nDescriptive association; intervention effects differ across heads")
        save(fig, "functional_head_scatter_" + key[0][:10])
    npz = data / "native_mixer_heatmaps.npz"
    if npz.exists():
        arrays = np.load(npz, allow_pickle=False)
        meta = json.loads((data / "native_mixer_heatmaps.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, name in zip(axes, ("gpt2", "mamba2")):
            matrix = arrays[name]
            limit = float(np.abs(matrix).max())
            image = ax.imshow(matrix, cmap="viridis" if name == "gpt2" else "RdBu_r",
                vmin=0 if name == "gpt2" else -limit, vmax=limit, interpolation="nearest")
            ax.set_title(f"{name} · layer {meta[name]['layer']} / head {meta[name]['head']}")
            ax.set_xlabel("Source token")
            ax.set_ylabel("Output token")
            fig.colorbar(image, ax=ax, label="Raw operator entry", shrink=.8)
        fig.suptitle("Predetermined native heads · separate raw-entry scales · distinct tokenization")
        save(fig, "native_mixer_heatmaps")
    (out / "plot_manifest.json").write_text(json.dumps(dict(created=created,
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_bundle_manifest_sha256=hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(),
        scope="replot saved aggregate means/CIs; no bootstrap recomputation"), indent=2) + "\n")
    print(json.dumps(dict(output=str(out), figures=len(created), files=created)))


if __name__ == "__main__":
    main()
