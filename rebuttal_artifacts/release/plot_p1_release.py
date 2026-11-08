"""Replot only the retained GPT-2 functional figure from its compact inputs.

The P1 drawing block follows the retained canonical producer, without the
redundant figure footer; statistical details remain in the adjacent caption.
Withdrawn full/cross-cut figures and their data are not loaded or exported.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

def read_csv(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plotting-rows", type=Path,
        default=Path(__file__).with_name("p1_plotting_rows.csv.gz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    functional = read_csv(args.plotting_rows)
    if len(functional) != 54 or any(row.get("checkpoint") != "gpt2" or not row.get("intervention") for row in functional):
        raise ValueError("Expected the 54 retained GPT-2 functional plotting rows")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none"})
    args.output.mkdir(parents=True, exist_ok=True)
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

    fig, axes = plt.subplots(3, 3, figsize=(6.6, 6.55))
    for col, layer in enumerate((0, 5, 11)):
        base = select(functional, checkpoint="gpt2", layer=layer)
        for row, metric, ylabel in (
                (0, "relative_loss_percent", "Relative loss change (%)"),
                (1, "relative_ppl_percent", "Relative PPL change (%)")):
            ax = axes[row, col]
            for intervention, label, color in (
                    ("ablate", "Head output zeroed", "#2768ad"),
                    ("sink-neutralized", "First-token reset", "#cf6c24")):
                draw(ax, select(base, intervention=intervention, metric=metric), label, color)
            ax.axhline(0, color="#555555", linewidth=.6)
            if row == 0:
                ax.set_title(f"Layer {layer}")
            if col == 0:
                ax.set_ylabel(ylabel)
                if row == 0:
                    ax.legend(frameon=False, loc="best", fontsize=6.2)
        ax = axes[2, col]
        draw(ax, select(base, intervention="ablate", metric="native_srank"), "Original head", "#2768ad")
        draw(ax, select(base, intervention="sink-neutralized", metric="intervention_srank"),
            "First-token reset", "#cf6c24")
        if col == 0:
            ax.set_ylabel("Full mixer stable rank")
            ax.legend(frameon=False, fontsize=6.2)
    fig.suptitle("GPT-2: one head modified at a time", y=.994, fontsize=9)
    fig.supxlabel("Context tokens", y=.018, fontsize=8)
    fig.subplots_adjust(left=.12, right=.985, top=.947, bottom=.10, hspace=.43, wspace=.42)
    save(fig, "p1_functional_loss_spectrum",
        "Separate GPT-2 head modifications: the evaluation code zeros one head output or resets its first-token score to the causally accessible non-first-token median, retaining the same input and model weights. Top: relative mean next-token prediction-loss change (%), 100*(modified NLL-native NLL)/native NLL. Middle: relative perplexity change (%), 100*(exp(modified NLL-native NLL)-1), where NLL=-mean log probability of the true next token and PPL=exp(NLL). These are two scales of the same final vocabulary predictions; positive changes mean worse prediction. Each percentage is computed per head/document trial, then heads are averaged within each document and documents are averaged. Bands bootstrap the 32 paired document summaries. Bottom: native and first-token-reset stable ranks; these are head means, whereas manuscript Figure1 uses medians. All head modifications are performed separately.")

    manifest = dict(script_sha256=sha(__file__),
        source_plotting_rows_sha256=sha(args.plotting_rows),
        scope="retained GPT-2 functional figure only; saved estimates/CIs, no rebootstrap",
        n_selected_plotting_rows=len(functional),
        files={name: sha(args.output/name) for name in created})
    (args.output/"p1_replot_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(output=str(args.output),files=created)))

if __name__ == "__main__": main()
