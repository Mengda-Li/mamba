# WikiText attention-rank comparison

This directory contains the trained-model experiment requested for
`article/draft.tex`. The rank proxies use the singular values `s_j` exactly as
defined in the draft:

- `stable_rank = sum(s_j^2) / max(s_j)^2`
- `erank_1 = exp(H(s_j / sum(s_j)))`
- `erank_2 = exp(H(s_j^2 / sum(s_j^2)))`

## Operators

| Series | Checkpoint and layer | Matrix analyzed |
|---|---|---|
| Mamba-2 (SSD) | `mistralai/Mamba-Codestral-7B-v0.1`, layer 0 | `L * (C B^T)` reconstructed from the trained Mamba-2 mixer |
| RetNet (decay) | `fla-hub/retnet-1.3B-100B`, layer 0 | `tril(gamma_h^(i-j) Q K^T / sqrt(d))`, including trained projections and rotary positions |
| Based (raw causal linear) | `hazyresearch/based-360m`, first global linear layer (layer 1) | `tril(Phi(Q) Phi(K)^T)` using Based's trained second-order Taylor feature map |
| Based (row-normalized) | Same Based layer | The preceding raw matrix divided by each causal row sum, as in the actual Based mixer |
| GPT-2 (causal softmax) | `openai-community/gpt2`, layer 0 | `softmax(tril(Q K^T / sqrt(d)))` |

Based uses zero-based layer indices, but its layer 0 mixer is `BaseConv`, not
linear attention. The checkpoint declares its global Taylor-feature linear
attention blocks in `alt_mixer_layers = [1, 6, 11, 16, 21]`; consequently,
layer 1 is the first layer that has the causal-linear matrix analyzed here.
The extractor runs the embedding and complete layer-0 convolution/MLP first,
then constructs `Q` and `K` from the trained input to layer 1.

Kimi Linear was not used. Its local BF16 weights require 91.5 GiB before
runtime memory, and KDA is a delta-rule operator rather than the scalar causal
mask studied in the draft. Based is both tractable on the available A10 and a
closer learned causal-linear baseline.

## Sampling

- Dataset: WikiText-103 raw validation split.
- Requested lengths: 64, 128, 256, 384, 512, 768, and 1024 tokens.
- Two independently evaluated source-text spans per length: one begins at
  reference token 0 and one at a seeded random offset. The same decoded spans
  are shared across models, but each model tokenizes them independently.
- Four evenly spaced heads per model.
- Each model tokenizes the same text and truncates it to the requested length;
  all runs produced exactly the requested token count.
- For each span, ranks are computed separately for each head; curves then pool
  the resulting eight values (two spans times four heads) at each length. Lines are
  medians, dark bands are interquartile ranges, and light bands span the
  10th--90th percentiles.

The checkpoints differ in scale, training corpus, training budget, and
tokenizer. The comparison is therefore architectural evidence, not a
controlled causal comparison of training recipes.

## Files

- `rank_results.csv`: all 280 per-head observations.
- `rank_summary.csv`: medians and 10/25/75/90 percentiles by series and length.
- `asymptotic_loglog_slopes.csv`: fitted slopes of `log(rank)` against
  `log(length)` over lengths 256--1024.
- `mamba2_rank_vs_length_iqr.svg` / `.png`: Mamba-2 `srank`, `erank_1`, and
  `erank_2` versus sequence length, with structured-mask theory rates.
- `retnet_rank_vs_length_iqr.svg` / `.png`: the corresponding RetNet panels
  and decay-mask theory rates.
- `based_rank_vs_length_iqr.svg` / `.png`: raw and row-normalized Based panels;
  dashed theory rates apply only to the raw causal-linear operator.
- `gpt2_rank_vs_length_iqr.svg` / `.png`: GPT-2 panels without a theory curve.
- `based_mamba2_retnet_rank_vs_length_iqr_3x3.svg` / `.png`: the combined
  3-model by 3-rank-proxy grid, ordered Based, Mamba-2, and RetNet by row.
- `based_mamba2_retnet_gpt2_rank_vs_length_iqr_4x3.svg` / `.png`: the
  four-model grid, ordered GPT-2, Based, Mamba-2, and RetNet by row.
- `windows.csv`: exact source text and reference-token offsets.

## Theoretical reference curves

The dashed curves show the sequence-length dependence of the bounds in
`article/draft.tex`:

| Operator | stable rank | `erank_1` | `erank_2` |
|---|---:|---:|---:|
| Mamba-2 / stable semiseparable | `l / (log l)^2` | `l / log l` | `l / (log l)^2` |
| RetNet / decay | `l / (log l)^2` | `l / log l` | `l / (log l)^2` |
| Based / raw causal linear | `1` | `log l` | `1` |

The theorems contain model-dependent multiplicative constants that are not
known for these trained checkpoints. The plots therefore use a unit prefactor:
they compare growth-rate shapes and should not be read as numerical tests with
constant exactly one. The raw causal-linear theorem does not cover Based's
row-normalized operator. No theoretical reference is drawn for GPT-2.

## Exploratory slopes

The fitted slopes over lengths 256--1024 are:

| Series | stable rank | erank_1 | erank_2 |
|---|---:|---:|---:|
| Mamba-2 (SSD) | 0.839 | 0.965 | 0.982 |
| RetNet (decay) | 0.511 | 0.920 | 0.708 |
| Based (raw causal linear) | 0.028 | 0.766 | 0.246 |
| Based (row-normalized) | 0.039 | 0.693 | 0.239 |
| GPT-2 (causal softmax) | 0.551 | 1.058 | 1.063 |

Mamba-2 has near-linear `erank_1` and `erank_2` growth in this range. RetNet's
three proxies also grow, but its stable-rank and `erank_2` slopes are smaller.
The raw Based operator has almost constant stable rank and slowly growing
`erank_2`, consistent with a dominant direction in undiscounted causal linear
attention.

Layer-0 GPT-2 does **not** exhibit bounded entropy rank here: both entropy-rank
fits are approximately linear. This does not directly contradict the draft's
softmax results, which assume bounded low-rank score geometry, but it means the
current trained layer-0 GPT-2 experiment is not empirical evidence for
constant-rank softmax collapse. A follow-up should examine all GPT-2 layers and
measure the rank/geometry of `Q` and `K` alongside the attention matrix.

## Reproduction

```bash
uv run python compare_attention_ranks_wikitext.py \
  --models based,retnet,gpt2,mamba2 \
  --lengths 64,128,256,384,512,768,1024 \
  --windows-per-length 2 \
  --heads-per-model 4 \
  --out-dir attention_rank_comparison_outputs
```

The runner appends after every window and skips completed rows on restart.
Regenerate summaries and figures without loading checkpoints using:

```bash
uv run python compare_attention_ranks_wikitext.py \
  --plot-only --out-dir attention_rank_comparison_outputs
```
