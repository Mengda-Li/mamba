"""Compact vector rebuttal figures from corrected aggregate estimates only.

This script uses no checkpoints, Torch or pandas and performs no rebootstrap.
It preserves the saved document-bootstrap intervals and distinct operators.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


MODELS = (
    ("gpt2", "GPT-2"), ("bloom", "BLOOM-1b1"),
    ("pythia", "Pythia-1.4B"), ("retnet", "RetNet-1.3B"),
    ("mamba2", "Codestral-7B (HF)"), ("mamba2_1.3b", "Mamba2-1.3B"),
)


def read_csv(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bundle", type=Path)
    inputs.add_argument("--plotting-rows", type=Path,
        help="Replot the exact compact p1_p2_plotting_rows.csv.gz extract")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none"})
    args.output.mkdir(parents=True, exist_ok=True)
    if args.bundle:
        functional_path = args.bundle / "data/functional_layer.csv.gz"
        crosscut_path = args.bundle / "data/crosscut_layer.csv.gz"
        functional, crosscut = read_csv(functional_path), read_csv(crosscut_path)
        source_hashes = dict(source_bundle_manifest_sha256=sha(args.bundle / "manifest.json"),
            source_functional_csv_sha256=sha(functional_path), source_crosscut_csv_sha256=sha(crosscut_path))
    else:
        rows = read_csv(args.plotting_rows)
        functional = [r for r in rows if r.get("intervention")]
        crosscut = [r for r in rows if r.get("operator_variant")]
        source_hashes = dict(source_compact_csv_sha256=sha(args.plotting_rows))
    used, created = [], []

    def select(rows, **filters):
        return [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in filters.items())]

    def draw(ax, rows, label, color, style="-", marker="o"):
        if not rows:
            raise ValueError(f"No measured rows for {label}")
        rows = sorted(rows, key=lambda r: int(r["length"]))
        if len({r["length"] for r in rows}) != len(rows):
            raise ValueError(f"Repeated configurations for {label}; never pool runs implicitly")
        used.extend(rows)
        x = [int(r["length"]) for r in rows]
        y, lo, hi = [[float(r[k]) for r in rows] for k in ("estimate", "ci_low", "ci_high")]
        line, = ax.plot(x, y, style, color=color, marker=marker, markersize=3,
            linewidth=1.1, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=.15, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x], rotation=30)
        ax.grid(alpha=.18, linewidth=.4)
        ax.tick_params(pad=1.5)
        return line

    def save(fig, name, caption):
        for suffix in ("pdf", "svg", "png"):
            path = args.output / f"{name}.{suffix}"
            fig.savefig(path, dpi=220)
            created.append(path.name)
        (args.output / f"{name}.caption.md").write_text(caption + "\n")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.65))
    for col, layer in enumerate((0, 5, 11)):
        base = select(functional, checkpoint="gpt2", layer=layer)
        ax = axes[0, col]
        for intervention, label, color in (("ablate", "Head ablation", "#2768ad"),
            ("sink-neutralized", "Sink neutralization", "#cf6c24")):
            draw(ax, select(base, intervention=intervention, metric="delta_loss"), label, color)
        ax.axhline(0, color="#555555", linewidth=.6)
        ax.set_title(f"Layer {layer}")
        if col == 0:
            ax.set_ylabel("Paired next-token ΔNLL")
            ax.legend(frameon=False, loc="best", fontsize=6.2)
        ax = axes[1, col]
        draw(ax, select(base, intervention="ablate", metric="native_srank"), "Native", "#2768ad")
        draw(ax, select(base, intervention="sink-neutralized", metric="intervention_srank"),
            "Sink-neutralized", "#cf6c24")
        if col == 0:
            ax.set_ylabel("Full mixer stable rank")
            ax.legend(frameon=False, fontsize=6.2)
        ax.set_xlabel("Context tokens")
    fig.suptitle("GPT-2: separately applied single-head interventions", y=.994, fontsize=9)
    fig.text(.5, .009, "32 paired documents; head mean within document; shading: 95% document bootstrap.",
        ha="center", fontsize=6.6)
    fig.subplots_adjust(left=.11, right=.985, top=.91, bottom=.115, hspace=.43, wspace=.37)
    save(fig, "p1_functional_loss_spectrum",
        "GPT-2 single-head functional interventions and corrected operator spectra. Each head is intervened on separately through the full native forward pass; means first average heads within each document, then average the same 32 documents. Top: paired change in next-token NLL, with positive values indicating worse prediction. Layer-0 neutralization intervals cross zero at all three lengths; its small curve is compressed by the larger ablation scale. Bottom: the targeted head's full-mixer stable rank before and after causal sink neutralization; ablation is omitted from the rank panel because its operator is zero. Shading preserves 2000-resample document-bootstrap 95% intervals. Rank changes and loss effects are distinct measurements; these results do not establish a rank-only causal effect on task loss.")

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.75))
    for ax, (model, title) in zip(axes.flat, MODELS):
        base = select(crosscut, checkpoint=model, operator_variant="native", metric="srank")
        draw(ax, select(base, region="full"), "Full operator", "#2768ad")
        draw(ax, select(base, region="crosscut", cut_fraction="0.5", gap="0"),
            "Strict midpoint cross-cut", "#cf6c24", "--", "s")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Context tokens")
        ax.set_ylabel("Stable rank")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    fig.suptitle("Native full and past-to-future operators: final layers", y=.997, fontsize=9)
    fig.text(.5, .007, "Shading: 95% document bootstrap. Mamba: canonical M. At 2048, eligible documents contribute.",
        ha="center", fontsize=6.5)
    fig.subplots_adjust(left=.09, right=.985, top=.86, bottom=.12, hspace=.49, wspace=.40)
    save(fig, "p2_native_full_crosscut_srank",
        "Corrected final-layer native stable rank versus context length for six pretrained checkpoints. Solid curves use the full sequence-relation operator; dashed curves use its strict midpoint future-row/past-column block at gap 0, with no diagonal overlap. Mamba panels use canonical M, without absorbing dt or the D skip. Codestral-7B uses the fixed Transformers 4.57.1 implementation with global gated RMSNorm, rather than claiming parity with the official eight-group normalization; native here means the unmodified learned mask in that runtime. All spectra use the validated entry-energy/gesvd protocol; saved 95% document-bootstrap intervals are shaded. Each model uses its own native tokenizer, 32 frozen document identities through 1024 tokens, and the recorded eligible subset at 2048; GPT-2 ends at its 1024-token limit. Full rank and cross-cut rank quantify distinct operator properties and neither curve is a task-quality score.")

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.9))
    ax = axes[0, 0]
    for variant, label, color in (("native", "Native", "#2768ad"),
        ("alibi_m0.01", "ALiBi m=0.01", "#cf6c24"), ("alibi_m0.1", "ALiBi m=0.1", "#54924a")):
        draw(ax, select(crosscut, checkpoint="gpt2", region="full", operator_variant=variant, metric="srank"), label, color)
    ax.set_title("GPT-2: fixed-logits ALiBi")
    ax.set_ylabel("Full stable rank")
    ax.legend(frameon=False, fontsize=5.8)
    for ax, model, title in ((axes[0, 1], "mamba2", "Codestral-7B (HF): M/dt/D"),
        (axes[0, 2], "mamba2_1.3b", "Mamba-1.3B: M / dt / D")):
        for variant, label, color in (("native", "M", "#2768ad"),
            ("native_dt", "M diag(dt)", "#cf6c24"),
            ("native_dt_with_d", "M diag(dt)+D", "#54924a")):
            draw(ax, select(crosscut, checkpoint=model, region="full", operator_variant=variant, metric="srank"), label, color)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=5.8)
    for ax, model, title in ((axes[1, 0], "retnet", "RetNet-1.3B: mask control"),
        (axes[1, 1], "mamba2", "Codestral-7B (HF): mask"),
        (axes[1, 2], "mamba2_1.3b", "Mamba-1.3B: mask control")):
        for variant, label, color, style in (("native", "Native mask", "#2768ad", "-"),
            ("all_one", "All-one mask", "#cf6c24", "--")):
            draw(ax, select(crosscut, checkpoint=model, region="crosscut", cut_fraction="0.5",
                gap="0", operator_variant=variant, metric="value_output_to_input_gain"), label, color, style)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Context tokens")
        if model == "retnet":
            ax.set_ylabel("Strict cross-cut core gain")
        ax.legend(frameon=False, fontsize=5.8)
    fig.suptitle("Frozen-factor controls: position, discretization and mask", y=.997, fontsize=9)
    fig.text(.5, .008, "Core gain: canonical M; dt-scaled Mamba values. Shading: 95% document bootstrap.",
        ha="center", fontsize=6.4)
    fig.subplots_adjust(left=.10, right=.985, top=.91, bottom=.12, hspace=.48, wspace=.40)
    save(fig, "p2_frozen_factor_controls",
        "Frozen-factor controls at final layers. Top left: GPT-2 content logits are held fixed while causal ALiBi penalties with slopes 0.01 and 0.1 are added and each row is renormalized; this is an operator intervention, not a retrained ALiBi model. Top middle/right: native Mamba full stable rank under three explicit boundaries, canonical M, M diag(dt), and M diag(dt)+D; the diagonal skip can dominate full rank. Codestral-7B is the fixed Transformers 4.57.1 runtime (global rather than official eight-group gated RMSNorm); native denotes its unmodified learned mask. Bottom: strict midpoint cross-cut canonical core gain ||M_cut W_past||F/||W_past||F with native versus all-one masks while captured factors and value inputs remain fixed. For Mamba, W=diag(dt)V is the captured discretized value input in both masks, with the same denominator; dt is carried by values rather than absorbed into canonical M. D is absent from a strict cross-cut. The gain measures fixed-factor transport and excludes output gating/projection, so it is not a generation-quality measure. Shading preserves corrected 95% document-bootstrap intervals; model/tokenizer and eligible-cohort distinctions are retained.")

    # A compact, exact plotting-input extract accompanies the vectors.
    unique = {json.dumps(row, sort_keys=True): row for row in used}
    fields = sorted({k for row in unique.values() for k in row})
    input_path = args.output / "p1_p2_plotting_rows.csv.gz"
    with gzip.open(input_path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(unique.values())
    manifest = dict(script_sha256=sha(__file__), **source_hashes,
        compact_plotting_rows_sha256=sha(input_path), source_bundle="corrected_primary_bundle",
        scope="replot saved corrected means and document-bootstrap intervals; no raw-data rebootstrap",
        files={name: sha(args.output / name) for name in created}, n_selected_plotting_rows=len(unique),
        source_configurations=sorted({(r["checkpoint"], r["run_id"]) for r in unique.values()}),
        n_documents_range=sorted({int(r["n_documents"]) for r in unique.values()}))
    (args.output / "p1_p2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=str(args.output), files=created, plotting_rows=len(unique))))


if __name__ == "__main__":
    main()
