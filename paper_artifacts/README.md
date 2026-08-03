# Paper artifacts

This directory contains the compact artifacts used by the ICLR 2027
manuscript. The full head-resolved metrics, spectra, model checkpoints, and
intermediate shards are intentionally not stored in Git.

## Figures

`figures/` contains the seven PDFs included by `init_sub.tex`:

- `figure1_native_rank_scaling.pdf`
- `figure2_gpt2_depth_head_sink.pdf`
- `figure3_gpt2_sink_interventions.pdf`
- `figure4_low_rank_dashboard.pdf`
- `figure5_mamba_memory_dashboard.pdf`
- `figure6_matched_factor_interventions.pdf`
- `wikipedia_replication.pdf`

Figures 1--5 are produced by `aggregate_experiments_2027.py`. Figure 6 uses
`plot_mask_replacement_figure.py` and
`tables/paired_intervention_effects.csv`. The Wikipedia figure is produced by
`analyze_wikipedia_replication.py`.

From the repository root, regenerate both Figure 6 formats with one command:

```bash
uv run python plot_mask_replacement_figure.py \
  experiments_2027/tables/paired_intervention_effects.csv \
  experiments_2027/figures/figure6_matched_factor_interventions.svg
```

The script writes the SVG at the requested path and a same-stem PDF beside it.

## Tables

`tables/` contains the reconstruction results, document-bootstrap slopes,
assumption audit, claim decisions, and compact diagnostics reported or
discussed in the paper. Rows for the archived BERT and initialization controls
are excluded because those experiments are outside the submitted manuscript.

`wikipedia/` contains the compact curve, slope, and paired-effect summaries for
the cross-corpus replication.

To recompute these artifacts from scratch, run the experiment pipeline
described in the repository README. The raw experiment directories are ignored
because they are large and contain machine-specific paths.
