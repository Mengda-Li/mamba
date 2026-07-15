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

The combined Based/Mamba-2/RetNet figure is
`attention_rank_comparison_outputs/based_mamba2_retnet_rank_vs_length_iqr_3x3.svg`.
Full checkpoint paths, sampling details, operator definitions, and commands for
rerunning model inference are documented in
`attention_rank_comparison_outputs/README.md`.
