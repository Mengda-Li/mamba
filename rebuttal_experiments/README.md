# Rebuttal experiments

This is the canonical source directory. Helpers are imported from the parent
experiment repository through `repo_paths.py`, so a standalone GitHub clone
works without the paper's directory layout. Run commands from repository root.
Checkpoints, datasets and frozen manifests are explicit local inputs.

The current publication retains the GPT-2 functional figure and the audited
frame-geometry figures. Completed full/cross-cut and frozen-factor figure sets
were withdrawn from the manuscript; their vectors and aggregate plotting rows
are excluded from `rebuttal_artifacts/release/`. Generic runners and historical
source remain available without redefining the current publication inventory.

```bash
python -m unittest discover -s rebuttal_experiments -p test_gpt2_functional.py -v
python rebuttal_experiments/analyze_rebuttal.py self-test
```

The corrected spectral protocol defaults to `--svd-driver gesvd`; CUDA passes
the driver explicitly and CPU omits the unsupported driver argument. Frobenius
norm, stable-rank numerator and locality fractions use FP64 sums of matrix
entries, independently of singular values. Every spectrum records the SVD
Frobenius discrepancy. QR discrepancy above `1e-5` fails the sample instead of
silently writing a completion marker. This does not recover tiny tails lost
when an operator was formed in FP32. Precision-sensitive algebraic-rank/tiny
absolute-threshold claims need FP64 formation or a factorized check.

Use new output directories for corrected runs. Source hashes and schema2
separate them from historical results. Functional loss/ablation logic is
unchanged; its associated spectra use the corrected solver/energy checks.
Historical local plotting bundles are separate from the current publication;
read their manifests before using any archived numbers. Retained figure inputs
and replot commands are documented in `rebuttal_artifacts/release/README.md`.

Example explicit queue generation (does not launch jobs):

```bash
python rebuttal_experiments/make_job_plan.py \
  --frozen-manifest /data/submitted_manifest.jsonl \
  --structured-manifest /data/long_structured_manifest.jsonl \
  --modern-manifest /data/modern_manifest.jsonl --model-root /models \
  --run-root /results/rebuttal_v2 --output /results/plan_v2.json
python rebuttal_experiments/supervise.py --plan /results/plan_v2.json
```

`historical_plans/` preserves old machine-specific queue records for provenance.
The conservative Mamba memory patch preserves native broadcast/reduction
expressions in head chunks; its provenance and numerical checks remain part
of the runner. The 1.3B adapter requires independent serial-equation validation
before the corresponding runner accepts its success marker.

Compact aggregate caches, 128-token heatmap arrays and their manifests live
under `rebuttal_artifacts/`. They support replotting, not new bootstrap analyses:

```bash
python rebuttal_experiments/build_local_bundle.py \
  --results /path/to/aggregate/results \
  --output rebuttal_artifacts/local_experiment_bundle
python rebuttal_artifacts/local_experiment_bundle/plot_local_bundle.py \
  --output rebuttal_artifacts/local_experiment_bundle/figures
```

No checkpoints, raw spectra, large JSONL or credentials belong in this repo.
The existing top-level submission runners and their historical outputs are
preserved; no Git commit or push is performed by these scripts.
