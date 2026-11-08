# Token-frame geometry and eigen-angle measurements

`run_frame_geometry.py` measures actual model activations. It captures the
normalized mixer input `X`, GPT-2 Q/K, RetNet Q/K before and after rotary, and
Mamba C/B after its native convolution/activation. Mamba factor-group IDs label
heads that share B/C; these copies are not independent observations.

The first protocol uses the first eight sorted frozen documents, early/final
layers, head0 and the midpoint head, and contexts64/128/256/512/1024/2048 when
available. GPT-2 stops at its native1024-token limit. Full square left-SVD
eigen-angles are measured through1024; longer contexts use thin active frames.
The input `X` measurement most directly extends the cited paper's Figure3;
Q/K and C/B measurements test the factors used in this paper's own assumptions.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python rebuttal_experiments/run_frame_geometry.py \
  --model mamba2 --checkpoint /models/Mamba-Codestral-7B-v0.1 \
  --manifest /data/structured_manifest.jsonl --output /results/frame_geometry/mamba2 \
  --device cuda:0 --lengths 64,128,256,512,1024,2048 --layers 0,-1 --heads 0,mid \
  --doc-stop 8 --eigen-max-length 1024 --null-replicates 64 --eigen-null-replicates 8
```

Use `--model mamba13` with its separately retokenized `mamba13` manifest and
validated converted checkpoint; it runs FP32. Other model labels are `gpt2`
and `retnet`; `--manifest-label` can select an explicit source label.

Primary eigen-angle statistics retain the solver's native signs. Two
prespecified sensitivity analyses also report maximum-pivot-positive signs
and one fixed-seed random sign draw, without choosing the better-looking one.
Histograms use50 bins and KL regularization1e-10. KS, circular Kuiper/Watson,
Fourier moments and midpoint-grid angular Wasserstein summaries are
descriptive; no iid-angle p-values are used.

Thin ranks4/8/16 compare leverage/coherence and second/fourth moments with
Haar-Stiefel frames. Principal angles, ordered squared column overlaps,
off-diagonal overlap and Procrustes error separately measure shared span and
shared basis, for Q/K, X/Q, X/K and feature-side V_Q/V_K. RetNet additionally
checks its pre-rotary geometry. Factor energy ranks and retained-energy
fractions are measured separately for each factor.

Square-U nulls use shape-matched Gaussian factors and Haar active frames with
distinct singular values, followed by the identical SVD/completion/sign
procedure. Structural completion count, numerical nullspace size and a
completion-dominates flag accompany each measurement. For example, GPT-2
X at1024 tokens has256 structural completion columns because its width is768;
its512-token X measurement has none. A uniform completed-U histogram alone is
not interpreted as verifying an active-frame distribution or common basis.

The script writes atomic per-document/length/layer JSONL shards, checksummed
completion markers, cached seeded null references and a hashed protocol.
Missing long documents are recorded, not replaced. The raw model observations
must be completed before reporting results. Offline sanity checks require
PyTorch, but no checkpoint or download:

```bash
python rebuttal_experiments/run_frame_geometry.py --self-test
```

The self-test contrasts isotropic and token-localized frames, labels nullspace
completion explicitly, and verifies that a shared span need not share the
ordered singular basis. Score-matrix U/V eigen-angles are not part of this
first pass; their rank-deficient nulls require a separate matched-spectrum
protocol.
