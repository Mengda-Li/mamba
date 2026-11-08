# Audited rebuttal figures

The final release contains **five figure sets**, each as vector PDF and SVG:

| File | Scientific scope |
|---|---|
| `p1_functional_loss_spectrum` | Separate single-head GPT-2 effects on next-token loss, alongside targeted-head stable rank. Layer-0 neutralization intervals cross zero; rank alone is not causally isolated. |
| `p2_native_full_crosscut_srank` | Six fixed-runtime checkpoints, final-layer full versus strict midpoint cross-cut stable rank. |
| `p2_frozen_factor_controls` | Fixed-logits ALiBi; explicit Mamba M/dt/D boundaries; fixed-value native/all-one mask transport gains. |
| `audited_geometry/input_frame_angles` | Final-layer input X angles, only numerically full-row-rank points, on32 documents. Descriptive distances, not validation of a joint Haar law. |
| `audited_geometry/active_frame_audit` | Active Q/K coherence and partial shared-subspace/ordered-basis overlap; empirical document CIs and Haar frame-variation bands are different quantities. |

Codestral-7B(HF) denotes the completed Transformers4.57.1 experimental object.
Its gated RMSNorm is global over8192 channels, unlike the official eight
1024-channel groups. No equivalence to the official runtime is claimed, and
no unvalidated native-norm patch or rerun is included. “Native” in these
figures means the unmodified learned mask in that fixed runtime.

The input-angle branch remains modulo[0,2pi), and angular W1 uses a
deterministic uniform midpoint grid. The cited NeurIPS implementation instead
uses np.angle+pi and an independently sampled uniform reference, so this is
not an exact reproduction of that metric protocol. Full-U Q/K completions
are excluded from the final figures. Coherence departures and incomplete
ordered alignment remain visible and prevent a blanket Haar/common-frame
validation claim.

Adjacent `.caption.md` files give definitions and interpretation limits.
`p1_p2_plotting_rows.csv.gz` retains exactly128 corrected aggregate rows.
`audited_geometry/audited_geometry_plotting_rows.csv.gz` retains exactly73
verified aggregate rows. Replotting needs only NumPy and Matplotlib, with no
Torch, checkpoint, dataset or network access:

```bash
python rebuttal_experiments/plot_rebuttal_figures.py \
  --plotting-rows rebuttal_artifacts/release/p1_p2_plotting_rows.csv.gz \
  --output /tmp/rebuttal_p1_p2

python rebuttal_experiments/plot_audited_geometry.py \
  --plotting-rows rebuttal_artifacts/release/audited_geometry/audited_geometry_plotting_rows.csv.gz \
  --output /tmp/rebuttal_audited_geometry
```

These compact inputs preserve the saved estimates and intervals; they do not
support a fresh bootstrap from raw model observations. Raw-source/config and
summary hashes remain in the manifests. All final vectors were rendered and
visually checked; they contain no embedded raster images. PNG previews are
local only and excluded from the publication allowlist.

`publication_code.zip` contains the explicit canonical source plus required
legacy helpers, without Git metadata, credentials, SSH wrappers, historical
server plans, weights or raw JSONL. Source and final figure hashes/sizes are
listed in `PUBLICATION_ALLOWLIST.json`. No commit or push was performed.
