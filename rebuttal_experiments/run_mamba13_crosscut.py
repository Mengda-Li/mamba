"""Run the existing SSD diagnostics on the separately validated 1.3B checkpoint."""

from repo_paths import REPO_ROOT, ARTIFACT_ROOT
from copy import copy
import json
from pathlib import Path
import sys

import run_crosscut as runner
from mamba_memory_patch_exact import install_patch

patch = install_patch(head_chunk_size=8)
original_provenance = runner.provenance
original_documents = runner.read_documents
original_base = runner.base_row


def provenance(args):
    marker = Path(args.checkpoint) / 'ADAPTER_VALIDATED.json'
    if not marker.is_file():
        raise RuntimeError('Converted checkpoint has no successful adapter-validation marker')
    report = json.loads(marker.read_text())
    validation = marker.parent / 'validation_report.json'
    if report.get('status') != 'passed' or report.get('validated_dtype') != 'float32':
        raise RuntimeError('Adapter has not passed FP32 validation')
    if not validation.is_file() or runner.file_sha(validation) != report.get('validation_report_sha256'):
        raise RuntimeError('Adapter numerical-validation report is missing or changed')
    metadata, _ = original_provenance(args)
    metadata.pop('run_fingerprint', None)
    metadata.update(runtime_memory_patch=patch,
                    launcher_sha256=runner.file_sha(Path(__file__)),
                    adapter_validation_marker=report,
                    adapter_validation_marker_sha256=runner.file_sha(marker),
                    checkpoint_label='mamba2_1.3b', manifest_label='mamba13')
    run_id = runner.digest(metadata)
    metadata['run_fingerprint'] = run_id
    return metadata, run_id


def documents(args):
    selection = copy(args)
    selection.model = 'mamba13'
    return original_documents(selection)


def base_row(*args, **kwargs):
    row = original_base(*args, **kwargs)
    row.update(model='mamba2_1.3b', checkpoint='mamba2_1.3b')
    return row


runner.provenance = provenance
runner.read_documents = documents
runner.base_row = base_row

if __name__ == '__main__':
    if '--model' not in sys.argv or sys.argv[sys.argv.index('--model') + 1] != 'mamba2':
        raise SystemExit('This launcher requires --model mamba2')
    runner.main()
