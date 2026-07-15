# Attention-rank comparison

This repository contains the code and compact results used to compare the
sequence-length growth of stable rank, `erank_1`, and `erank_2` for trained
Based, Mamba-2, RetNet, and GPT-2 token-mixing matrices on WikiText-103.

The main experiment is implemented in
`compare_attention_ranks_wikitext.py`. Recreate all plots from the committed
rank measurements without loading model checkpoints using:

```bash
uv sync
uv run python compare_attention_ranks_wikitext.py \
  --plot-only \
  --out-dir attention_rank_comparison_outputs
```

The four-model first-layer figure is
`attention_rank_comparison_outputs/based_mamba2_retnet_gpt2_rank_vs_length_iqr_4x3.svg`.
Full checkpoint paths, sampling details, operator definitions, and commands for
rerunning model inference are documented in
`attention_rank_comparison_outputs/README.md`.

Final-layer results are in `attention_rank_comparison_last_layer_outputs/`.
They evaluate Based layer 21 (its last global linear-attention block), Mamba-2
layer 63, RetNet layer 23, and GPT-2 layer 11. The four-model grid is
`based_mamba2_retnet_gpt2_rank_vs_length_iqr_4x3.svg`; see that directory's
`README.md` for the full command and memory settings.
