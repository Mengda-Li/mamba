# Attention Non-Collapse in State-Space Duality Model

This repository contains the experiment code and compact paper artifacts for
the ICLR 2027 submission. It reconstructs the token-mixing operators of GPT-2,
Based-360M, Mamba-Codestral-7B, and RetNet-1.3B, then measures how their
singular-value ranks change with context length.

The public release is intentionally paper-only. Archived BERT controls,
initialization simulations, notebooks, checkpoints, raw spectra, and
machine-specific experiment directories are not included.

## Repository map

| Path | Purpose |
| --- | --- |
| `build_sample_manifest.py` | Select 32 document-disjoint WikiText-103 articles and create nested token prefixes. |
| `validate_reconstruction.py` | Check that each reconstructed operator reproduces the native mixer output. |
| `run_gpt2_suite.py`, `gpt2_experiments.py` | Run GPT-2 rank, sink, factor-geometry, and score-intervention experiments. |
| `run_structured_experiments.py` | Run Based, RetNet, and Mamba-2 rank, factor, memory, and mask-intervention experiments. |
| `run_mamba_isolated.sh` | Run Mamba document-length units in separate processes to release CUDA allocator state. |
| `replay_mamba_memory_transitions.py` | Recreate the selected Mamba transition arrays required by the exact memory audit. |
| `audit_structured_memory_svd.py` | Compare stored memory/gain estimates with exact singular-value calculations. |
| `aggregate_experiments_2027.py` | Validate the completed design, bootstrap documents, write tables, and generate Figures 1--5. |
| `plot_mask_replacement_figure.py` | Generate the claim-specific Figure 6 as SVG and PDF. |
| `build_wikipedia_replication_manifest.py`, `run_wikipedia_replication.sh`, `analyze_wikipedia_replication.py` | Run and analyze the final-layer Wikipedia replication. |
| `paper_artifacts/` | The seven PDFs used by the paper and compact supporting tables. |

`compare_attention_ranks_wikitext.py` and `extract_mamba_matrices.py` provide
the Based and Mamba reconstruction helpers imported by the production runners.
The former also retains the original small comparison CLI for diagnostics.

## Rebuttal additions and portable artifacts

The supplemental sources in [`rebuttal_experiments/`](rebuttal_experiments/README.md)
run from this standalone repository. The current rebuttal publication retains
GPT-2 value-path interventions, trained-frame diagnostics and the focused
fixed-GPT-logit ALiBi full-spectrum control. The earlier full/cross-cut and
combined mask/feedthrough figures remain withdrawn; their plotting data and
vectors are excluded from the current release.
Historical analysis utilities and results remain preserved separately.

The audited vector figures, compact inputs and replot commands are in
[`rebuttal_artifacts/release/`](rebuttal_artifacts/release/README.md).
[`REVISED_FIGURES.md`](rebuttal_artifacts/release/REVISED_FIGURES.md) maps them
to the revised manuscript. No checkpoint or GPU is needed to replot the saved
means and document-bootstrap intervals. Codestral measurements use the explicit
Transformers 4.57.1 normalization convention documented with those figures.

## Environment

The production environment used Python 3.14, PyTorch with CUDA 12.8,
Transformers 4.57.1, and Flash Linear Attention 0.5.1. Install the environment
from `pyproject.toml` with:

```bash
uv sync
```

The full sweep requires Linux and CUDA. GPT-2 and the aggregation tests can be
run on smaller machines, but the 7B Mamba checkpoint and the dense length-1024
SVDs require substantially more GPU memory. The provided Wikipedia launcher
uses GPUs 0, 1, and 2 in parallel; its three commands can also be run
sequentially.

By default, checkpoints are read from `models/` and datasets from `data/`:

```text
models/
  gpt2/
  based-360m/
  Mamba-Codestral-7B-v0.1/
  retnet-1.3B-100B/
data/
  wikitext/wikitext-103-raw-v1/validation-00000-of-00001.parquet
  wikipedia/20231101.en/train-00000-of-00041.parquet
```

The locations can be changed without editing code:

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export GPT2_MODEL_PATH=/path/to/gpt2
export BASED_MODEL_PATH=/path/to/based-360m
export MAMBA2_MODEL_PATH=/path/to/Mamba-Codestral-7B-v0.1
export RETNET_MODEL_PATH=/path/to/retnet-1.3B-100B
export WIKITEXT_PARQUET=/path/to/validation.parquet
export WIKIPEDIA_PARQUET=/path/to/wikipedia.parquet
```

Explicit `--checkpoint` or model-specific checkpoint arguments take priority
over environment variables. All model and dataset revisions used by the paper
are pinned in `experiment_common.py` and written into generated provenance.

## Quick verification

These checks do not load model checkpoints:

```bash
uv run python aggregate_experiments_2027.py self-test
uv run python -m unittest \
  test_run_gpt2_suite.py \
  test_replay_mamba_memory_transitions.py \
  test_audit_structured_memory_svd.py
bash -n run_mamba_isolated.sh
bash -n run_mamba_isolated_replication.sh
bash -n run_wikipedia_replication.sh
```

## Main WikiText experiment

Raw outputs are written under `experiments_2027/`, which is ignored by Git.
The production protocol uses 32 documents, lengths
`64,128,256,384,512,768,1024`, and document-level bootstrap uncertainty.

1. Build the frozen nested-prefix manifest:

```bash
uv run python build_sample_manifest.py --out-dir experiments_2027
```

2. Validate each reconstructed operator:

```bash
for model in gpt2 based retnet mamba2; do
  uv run python validate_reconstruction.py \
    --model "${model}" \
    --manifest experiments_2027/sample_manifest.jsonl \
    --output experiments_2027/reconstruction_checks.json
done
```

3. Run GPT-2, Based, RetNet, and Mamba-2:

```bash
uv run python run_gpt2_suite.py
uv run python run_structured_experiments.py --model based
uv run python run_structured_experiments.py --model retnet
bash run_mamba_isolated.sh
```

The runners support document ranges and resumable shards. For large runs,
split `--document-start` and `--document-stop` across GPUs, then use the
runner's consolidation mode.

4. Replay and audit the Mamba transitions, then aggregate all paper results:

```bash
uv run python replay_mamba_memory_transitions.py
uv run python audit_structured_memory_svd.py --results-dir experiments_2027
uv run python aggregate_experiments_2027.py validate \
  --results-dir experiments_2027
uv run python aggregate_experiments_2027.py analyze \
  --results-dir experiments_2027
```

When validation and bootstrap caches are already complete, regenerate the
paper-only tables and Figures 1--5 with:

```bash
uv run python aggregate_experiments_2027.py finish-cached \
  --results-dir experiments_2027
```

Figure 6 is the final-layer, length-1024 all-one-minus-native mask contrast:

```bash
uv run python plot_mask_replacement_figure.py \
  experiments_2027/tables/paired_intervention_effects.csv \
  experiments_2027/figures/figure6_matched_factor_interventions.svg
```

That single command preserves the hand-rendered SVG and writes the paper PDF
beside it as `figure6_matched_factor_interventions.pdf`. Use `--output-pdf`
only when the PDF should have a different path.

## Wikipedia replication

The replication repeats final-layer rank growth and the matched-factor mask
intervention for Mamba-2 and RetNet on 32 English Wikipedia articles:

```bash
uv run python build_wikipedia_replication_manifest.py \
  --parquet "${WIKIPEDIA_PARQUET}"
bash run_wikipedia_replication.sh
uv run python analyze_wikipedia_replication.py
```

The analysis writes the native rank curves, slope intervals, paired mask
effects, and `wikipedia_replication.pdf`.

## Artifact policy

`paper_artifacts/figures/` contains the seven PDFs from the original experiment
package. `paper_artifacts/tables/` and `paper_artifacts/wikipedia/` contain
compact summaries used for the reported numbers. Full head-resolved metrics,
model checkpoints, downloaded datasets, logs, spectra, and temporary shards
remain outside Git because they are large and contain local paths.
