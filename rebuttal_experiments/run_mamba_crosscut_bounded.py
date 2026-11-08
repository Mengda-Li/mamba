"""Run the unchanged cross-cut protocol with audited process-local Mamba contractions."""

from repo_paths import REPO_ROOT, ARTIFACT_ROOT
from pathlib import Path
import run_crosscut as runner
from mamba_memory_patch_exact import install_patch

patch = install_patch()
original_provenance = runner.provenance


def provenance(args):
    metadata, _ = original_provenance(args)
    metadata.pop('run_fingerprint', None)
    metadata['runtime_memory_patch'] = patch
    metadata['launcher_sha256'] = runner.file_sha(Path(__file__))
    run_id = runner.digest(metadata)
    metadata['run_fingerprint'] = run_id
    return metadata, run_id


runner.provenance = provenance

if __name__ == '__main__':
    runner.main()
