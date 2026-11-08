"""Plot the fixed-GPT-2-logit ALiBi control from saved aggregate rows."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plotting-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.plotting_rows, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(str(length), variant, metric)
        for length in (256, 512, 1024)
        for variant in ("native", "alibi_m0.01", "alibi_m0.1")
        for metric in ("srank", "erank_2")}
    keys = {(r["length"], r["operator_variant"], r["metric"]) for r in rows}
    if len(rows) != 18 or keys != expected:
        raise ValueError("Expected the 18 distinct full-operator ALiBi-control rows")
    if any(r["checkpoint"] != "gpt2" or r["region"] != "full"
           or r["layer"] != "11" or r["n_documents"] != "32" for r in rows):
        raise ValueError("Unexpected checkpoint, operator, layer or document cohort")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), layout="constrained")
    for ax, metric, title in zip(axes, ("srank", "erank_2"),
            ("(a) Stable rank", r"(b) Energy entropy rank $\mathrm{erank}_2$")):
        for variant, label, color in (
                ("native", "Native softmax", "#2768ad"),
                ("alibi_m0.01", r"ALiBi $m=0.01$", "#cf6c24"),
                ("alibi_m0.1", r"ALiBi $m=0.1$", "#54924a")):
            selected = sorted((r for r in rows if r["metric"] == metric
                and r["operator_variant"] == variant), key=lambda r: int(r["length"]))
            x = [int(r["length"]) for r in selected]
            y, lo, hi = [[float(r[field]) for r in selected]
                for field in ("estimate", "ci_low", "ci_high")]
            ax.plot(x, y, "o-", label=label, color=color, markersize=3, linewidth=1.2)
            ax.fill_between(x, lo, hi, color=color, alpha=.15, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([256, 512, 1024], ["256", "512", "1024"])
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_xlabel("Context tokens")
        ax.grid(alpha=.18, linewidth=.4)
    axes[0].set_ylabel("Full-operator spectral rank")
    axes[0].legend(frameon=False, loc="upper left")
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for suffix in ("pdf", "svg", "png"):
        path = args.output / ("alibi_full_rank_control." + suffix)
        fig.savefig(path, dpi=220)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    manifest = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plotting_rows_sha256": hashlib.sha256(args.plotting_rows.read_bytes()).hexdigest(),
        "selected_rows": 18, "documents_per_cell": 32,
        "scope": "fixed captured GPT-2 logits; saved full-operator means and document-bootstrap intervals",
        "files": files}
    (args.output / "alibi_replot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "files": list(files)}))


if __name__ == "__main__":
    main()
