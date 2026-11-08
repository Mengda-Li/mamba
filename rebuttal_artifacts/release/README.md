# Retained rebuttal figures

The current release contains three retained manuscript figures and one supporting
geometry diagnostic, each as vector PDF and SVG:

| File | Scientific scope |
|---|---|
| `p1_functional_loss_spectrum` | Separate single-head GPT-2 effects on next-token loss and associated stable rank. This figure uses head means within documents; the original native-rank curves use head medians. Layer-0 neutralization intervals cross zero. |
| `audited_geometry/input_frame_angles` | Final-layer input X angles at numerically full-row-rank points, on 32 documents. Descriptive distances do not validate a joint Haar law. |
| `alibi_full_rank_control` | Fixed GPT-2 logits with geometric distance penalties; full stable rank and erank2 only. Head means within documents and document means; no separately trained ALiBi checkpoint. |
| `audited_geometry/active_frame_audit` | Active Q/K coherence and partial shared-subspace/ordered-basis overlap. Empirical document CIs and Haar frame-variation bands describe different quantities. |

The earlier full/cross-cut comparison and combined mask/feedthrough figures
remain withdrawn. Only the focused fixed-logit ALiBi full-spectrum control is
restored here. Other completed comparison data and generic analysis code remain
archived separately.

Codestral-7B (HF) denotes the recorded Transformers 4.57.1 implementation.
Its gated RMSNorm is global over 8192 channels, unlike the official eight
1024-channel groups. The retained geometry measurements describe that fixed
implementation.

Input-angle diagnostics use modulo [0,2pi) and a deterministic uniform
midpoint reference. The cited NeurIPS implementation uses np.angle+pi and
an independently sampled uniform reference, so the metric protocols differ.
Full-U Q/K completions are excluded from the retained figures. Active-frame
concentration and incomplete ordered alignment remain visible.

Adjacent `.caption.md` files provide definitions. `p1_plotting_rows.csv.gz`
contains exactly 36 retained functional aggregate rows; `alibi_plotting_rows.csv.gz`
contains 18 full-operator rows; the geometry cache contains 73 verified aggregate
rows. Replotting needs NumPy and Matplotlib,
without Torch, checkpoints, datasets or network access:

```bash
python rebuttal_artifacts/release/plot_p1_release.py \
  --plotting-rows rebuttal_artifacts/release/p1_plotting_rows.csv.gz \
  --output /tmp/rebuttal_p1

python rebuttal_experiments/plot_alibi_control.py \
  --plotting-rows rebuttal_artifacts/release/alibi_plotting_rows.csv.gz \
  --output /tmp/rebuttal_alibi

python rebuttal_experiments/plot_audited_geometry.py \
  --plotting-rows rebuttal_artifacts/release/audited_geometry/audited_geometry_plotting_rows.csv.gz \
  --output /tmp/rebuttal_audited_geometry
```

These caches preserve saved estimates and intervals and do not support a fresh
bootstrap from raw observations. The manifests retain source hashes and the
selection provenance. PNG previews are local only. The public figure inventory
is `PUBLICATION_MANIFEST.json`; it does not list withdrawn vectors as exports.

`publication_code.zip` contains current canonical contents from its preserved
source-file allowlist,
including shared and historical analysis utilities. Current figure exports
are defined by the release manifest, rather than by every capability of those
utilities. No credentials, SSH wrappers, weights or raw JSONL are included.
