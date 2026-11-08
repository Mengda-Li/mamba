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
    if len(functional) != 36 or any(row.get("checkpoint") != "gpt2" or not row.get("intervention") for row in functional):
        raise ValueError("Expected the 36 retained GPT-2 functional plotting rows")
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
    fig.suptitle("GPT-2: separately applied single-head interventions", y=.994, fontsize=9)
    fig.supxlabel("Context tokens", y=.025, fontsize=8)
    fig.subplots_adjust(left=.11, right=.985, top=.91, bottom=.145, hspace=.43, wspace=.37)
    save(fig, "p1_functional_loss_spectrum",
        "GPT-2 single-head functional interventions and corrected operator spectra. Each head is intervened on separately through the full native forward pass; means first average heads within each document, then average the same 32 documents. These layer summaries use head means; manuscript Figure 1 reports head medians. Top: paired change in next-token NLL, with positive values indicating worse prediction. Layer-0 neutralization intervals cross zero at all three lengths; its small curve is compressed by the larger ablation scale. Bottom: the targeted head's full-mixer stable rank before and after causal sink neutralization; ablation is omitted from the rank panel because its operator is zero. Shading preserves 2000-resample document-bootstrap 95% intervals. Rank changes and loss effects are distinct measurements; these results do not establish a rank-only causal effect on task loss.")

    manifest = dict(script_sha256=sha(__file__),
        source_plotting_rows_sha256=sha(args.plotting_rows),
        scope="retained GPT-2 functional figure only; saved estimates/CIs, no rebootstrap",
        n_selected_plotting_rows=len(functional),
        files={name: sha(args.output/name) for name in created})
    (args.output/"p1_replot_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(output=str(args.output),files=created)))

if __name__ == "__main__": main()
