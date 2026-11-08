"""Mamba13-only small-scale SVD patch, with a distinct source fingerprint.

Use a NEW output directory; do not overwrite or relabel frozen v2 output.
Native extraction, values, masks, driver and QC tolerance remain unchanged.
"""
from repo_paths import REPO_ROOT, ARTIFACT_ROOT
from pathlib import Path
import sys

import run_mamba13_crosscut as validated_launcher
import spectral_qc_scaled as scaled

runner = validated_launcher.runner
frozen_provenance = runner.provenance
runner.exact_svdvals = scaled.exact_svdvals
runner.entry_energy_qc = scaled.entry_energy_qc


def provenance(args):
    metadata, _ = frozen_provenance(args)
    metadata.pop("run_fingerprint", None)
    metadata["small_scale_svd_patch"] = {
        "policy": scaled.POLICY,
        "source_sha256": runner.file_sha(Path(scaled.__file__)),
        "launcher_sha256": runner.file_sha(Path(__file__)),
        "frozen_launcher_sha256": runner.file_sha(Path(validated_launcher.__file__)),
        "normalization": "FP64 division, FP32 explicit QR, FP64 singular-value restoration",
        "scope": "only FP32 matrices with max absolute entry below sqrt(float32 minimum normal)",
        "qc_tolerance_unchanged": args.svd_energy_tolerance,
    }
    run_id = runner.digest(metadata)
    metadata["run_fingerprint"] = run_id
    return metadata, run_id


runner.provenance = provenance

if __name__ == "__main__":
    if "--model" not in sys.argv or sys.argv[sys.argv.index("--model")+1] != "mamba2":
        raise SystemExit("This launcher requires --model mamba2")
    runner.main()
