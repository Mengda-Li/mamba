#!/usr/bin/env python3
"""Render the six claim-specific native-vs-all-one mask effects as SVG and PDF."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


FINAL_LAYERS = {"mamba2": 63, "retnet": 23}
MODELS = (("mamba2", "Mamba-2", "#D55E00"), ("retnet", "RetNet", "#CC79A7"))
METRICS = (
    ("srank", "stable rank"),
    ("erank_1", "entropy rank r1"),
    ("erank_2", "energy entropy rank r2"),
)
Effects = dict[tuple[str, str], dict[str, float]]


def read_effects(path: Path) -> Effects:
    selected: Effects = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            metric = row["metric"]
            if (
                model in FINAL_LAYERS
                and metric in {name for name, _ in METRICS}
                and row["reference_variant"] == "native"
                and row["intervention_variant"] == "all_one"
                and row["input_condition"] == "native-prefix"
                and int(row["length"]) == 1024
                and int(row["layer"]) == FINAL_LAYERS[model]
            ):
                selected[(model, metric)] = {
                    "effect": float(row["effect_intervention_minus_reference"]),
                    "low": float(row["ci_low"]),
                    "high": float(row["ci_high"]),
                }
    expected = {(model, metric) for model in FINAL_LAYERS for metric, _ in METRICS}
    if set(selected) != expected:
        raise RuntimeError(f"Expected six effects, found keys={sorted(selected)}")
    return selected


def metric_bounds(effects: Effects, metric: str) -> tuple[float, float]:
    rows = [effects[(model, metric)] for model, _, _ in MODELS]
    low = min(row["low"] for row in rows)
    high = max(0.0, max(row["high"] for row in rows))
    padding = max((high - low) * 0.10, 1.0)
    return low - padding, high + padding


def render_svg(effects: Effects) -> str:
    width, height = 960, 300
    panel_width, panel_gap = 230, 45
    first_x, plot_top, plot_bottom = 140, 82, 205
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#202020;letter-spacing:0}.title{font-size:17px;font-weight:600}.panel{font-size:15px}.label{font-size:14px}.tick{font-size:11px;fill:#555}.note{font-size:12px;fill:#444}</style>',
        '<text x="480" y="27" text-anchor="middle" class="title">Removing forgetting or decay concentrates the spectrum</text>',
    ]
    for panel_index, (metric, title) in enumerate(METRICS):
        x0 = first_x + panel_index * (panel_width + panel_gap)
        axis_low, axis_high = metric_bounds(effects, metric)

        def sx(value: float) -> float:
            return x0 + panel_width * (value - axis_low) / (axis_high - axis_low)

        zero_x = sx(0.0)
        parts.extend(
            [
                f'<text x="{x0 + panel_width / 2:.1f}" y="57" text-anchor="middle" class="panel">{html.escape(title)}</text>',
                f'<line x1="{x0}" y1="{plot_bottom}" x2="{x0 + panel_width}" y2="{plot_bottom}" stroke="#777" stroke-width="1"/>',
                f'<line x1="{zero_x:.1f}" y1="{plot_top}" x2="{zero_x:.1f}" y2="{plot_bottom}" stroke="#555" stroke-width="1.2"/>',
            ]
        )
        for fraction in (0.0, 0.5, 1.0):
            value = axis_low + fraction * (axis_high - axis_low)
            x = sx(value)
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 5}" stroke="#777"/>',
                    f'<text x="{x:.1f}" y="{plot_bottom + 20}" text-anchor="middle" class="tick">{value:.0f}</text>',
                ]
            )
        for row_index, (model, model_label, color) in enumerate(MODELS):
            y = 112 + row_index * 58
            row = effects[(model, metric)]
            x_low, x_mid, x_high = sx(row["low"]), sx(row["effect"]), sx(row["high"])
            if panel_index == 0:
                parts.append(f'<text x="{x0 - 18}" y="{y + 5}" text-anchor="end" class="label">{model_label}</text>')
            parts.extend(
                [
                    f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                    f'<line x1="{x_low:.1f}" y1="{y - 6}" x2="{x_low:.1f}" y2="{y + 6}" stroke="{color}" stroke-width="2"/>',
                    f'<line x1="{x_high:.1f}" y1="{y - 6}" x2="{x_high:.1f}" y2="{y + 6}" stroke="{color}" stroke-width="2"/>',
                    f'<circle cx="{x_mid:.1f}" cy="{y}" r="5.5" fill="{color}"/>',
                ]
            )
    parts.extend(
        [
            '<text x="530" y="274" text-anchor="middle" class="label">rank with all-one mask minus rank with native mask</text>',
            '<text x="530" y="293" text-anchor="middle" class="note">negative values mean lower rank after replacing only the mask</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts) + "\n"


def render_pdf(effects: Effects, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ("Helvetica", "Arial", "DejaVu Sans"),
            "font.size": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    fig, axes = plt.subplots(1, len(METRICS), figsize=(9.6, 3.0), sharey=True)
    y_positions = (1.0, 0.0)
    for panel_index, (ax, (metric, title)) in enumerate(zip(axes, METRICS)):
        axis_low, axis_high = metric_bounds(effects, metric)
        ax.axvline(0.0, color="#555555", linewidth=1.2, zorder=0)
        for y, (model, _model_label, color) in zip(y_positions, MODELS):
            row = effects[(model, metric)]
            ax.errorbar(
                row["effect"],
                y,
                xerr=(
                    [row["effect"] - row["low"]],
                    [row["high"] - row["effect"]],
                ),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=6.5,
                elinewidth=2.2,
                capsize=4,
                capthick=1.6,
                zorder=2,
            )
        ticks = [axis_low, (axis_low + axis_high) / 2.0, axis_high]
        ax.set_xlim(axis_low, axis_high)
        ax.set_xticks(ticks, [f"{value:.0f}" for value in ticks])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(title, pad=10)
        ax.tick_params(axis="y", length=0)
        if panel_index == 0:
            ax.set_yticks(y_positions, [label for _, label, _ in MODELS])
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Removing forgetting or decay concentrates the spectrum", fontsize=14, fontweight="semibold", y=0.95)
    fig.text(0.56, 0.10, "rank with all-one mask minus rank with native mask", ha="center", fontsize=11)
    fig.text(0.56, 0.035, "negative values mean lower rank after replacing only the mask", ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.73, bottom=0.30, wspace=0.20)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_svg", type=Path, help="SVG output path; the PDF defaults to the same stem")
    parser.add_argument("--output-pdf", type=Path, help="optional PDF output path")
    args = parser.parse_args()
    if args.output_svg.suffix.lower() != ".svg":
        parser.error("output_svg must end in .svg")
    output_pdf = args.output_pdf or args.output_svg.with_suffix(".pdf")
    effects = read_effects(args.input_csv)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(render_svg(effects), encoding="utf-8")
    render_pdf(effects, output_pdf)
    print(f"Wrote {args.output_svg} and {output_pdf}")


if __name__ == "__main__":
    main()
