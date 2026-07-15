# Last-layer WikiText attention-rank comparison

This directory repeats the trained-model rank experiment at the final relevant
token-mixing layer of each checkpoint.

| Model | Evaluated layer | Reason |
|---|---:|---|
| Based-360M | 21 | Last global Taylor-feature linear-attention block; later blocks are sliding attention or `BaseConv` |
| Mamba-Codestral-7B | 63 | Final Mamba-2 block |
| RetNet-1.3B | 23 | Final retention block |
| GPT-2 | 11 | Final causal-softmax attention block |

Based's model has 27 zero-indexed blocks, but its global linear-attention
blocks are only `[1, 6, 11, 16, 21]`. Layer 21 is therefore the last Based
layer with the causal-linear operator being compared in the 3x3 figure.

## Sampling

- WikiText-103 raw validation split.
- Requested lengths: 64, 128, 256, 384, 512, 768, and 1024 tokens.
- Two independently evaluated source-text spans per length.
- Four evenly spaced heads per model.
- Each plotted point pools eight separately computed values: two spans times
  four heads. Lines are medians, dark bands are IQRs, and light bands are
  10th--90th percentile ranges.

## Files

- `based_mamba2_retnet_rank_vs_length_iqr_3x3.svg` / `.png`: requested 3x3
  structured-model comparison.
- `gpt2_rank_vs_length_iqr.svg` / `.png`: requested final-layer GPT-2 figure.
- `{based,mamba2,retnet}_rank_vs_length_iqr.svg` / `.png`: individual
  final-layer structured-model figures.
- `rank_results.csv`: all 280 per-head/per-window measurements.
- `rank_summary.csv`: medians and percentile summaries.
- `asymptotic_loglog_slopes.csv`: descriptive log--log slopes over lengths
  256--1024.
- `windows.csv`: sampled source text and reference-token offsets.

The dashed structured-model curves use unit-prefactor theoretical rates. The
unknown theorem constants are not fitted, and the Based reference applies only
to its raw operator. GPT-2 has no theoretical curve.

## Reproduction

```bash
uv run python compare_attention_ranks_wikitext.py \
  --models based,retnet,gpt2,mamba2 \
  --lengths 64,128,256,384,512,768,1024 \
  --windows-per-length 2 \
  --heads-per-model 4 \
  --based-layer 21 \
  --mamba-layer 63 \
  --retnet-layer 23 \
  --gpt2-layer 11 \
  --based-device cuda \
  --mamba-device cuda \
  --mamba-chunk-size 64 \
  --retnet-device cuda \
  --gpt2-device cuda \
  --rank-device cuda \
  --out-dir attention_rank_comparison_last_layer_outputs
```

The Mamba chunk-size override evaluates the same SSD recurrence with smaller
temporary blocks to fit the 7B checkpoint on a 24 GiB GPU. A length-64
regression check produced identical rank measurements for chunk sizes 256 and
64. The generalized pure-PyTorch Based extractor reproduces the previous
layer-1 hidden states and matrices byte-for-byte before being used at layer 21.

Regenerate summaries and figures from `rank_results.csv` without loading any
checkpoint using:

```bash
uv run python compare_attention_ranks_wikitext.py \
  --plot-only \
  --out-dir attention_rank_comparison_last_layer_outputs
```
