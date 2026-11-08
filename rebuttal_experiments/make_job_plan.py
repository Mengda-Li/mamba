"""Write a portable, explicit four-GPU queue; does not launch it."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from repo_paths import REPO_ROOT, ARTIFACT_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=ARTIFACT_ROOT / 'runs_v2')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--frozen-manifest', type=Path, required=True)
    p.add_argument('--structured-manifest', type=Path, required=True)
    p.add_argument('--modern-manifest', type=Path)
    p.add_argument('--model-root', type=Path, default=REPO_ROOT / 'models')
    p.add_argument('--output', type=Path, default=ARTIFACT_ROOT / 'run_plan_v2.json')
    a = p.parse_args()
    a.model_root = a.model_root.resolve()
    a.frozen_manifest = a.frozen_manifest.resolve()
    a.structured_manifest = a.structured_manifest.resolve()
    a.modern_manifest = a.modern_manifest.resolve() if a.modern_manifest else None
    base = Path(__file__).resolve().parent
    root = a.run_root.resolve()
    def job(name, gpu, script, argv):
        return dict(name=name, gpu=gpu, command=[a.python, '-u', str(base / script), *map(str, argv)])
    def cross(name, model, checkpoint, gpu, manifest, lo=0, hi=32, lengths='256,512,1024,2048', heads='all', extra=()):
        script = 'run_mamba_crosscut_bounded.py' if model == 'mamba2' else 'run_crosscut.py'
        return job(name, gpu, script, ['--model', model, '--checkpoint', a.model_root / checkpoint,
            '--manifest', manifest, '--output', root / 'crosscut' / model, '--device', f'cuda:{gpu}',
            '--svd-device', f'cuda:{gpu}', '--svd-driver', 'gesvd', '--doc-start', lo, '--doc-stop', hi,
            '--lengths', lengths, '--layers', '11' if model == 'gpt2' else '-1', '--heads', heads, *extra])
    queues=[]
    for gpu, lo, hi in ((0,0,16),(1,16,32)):
        queues.append([job(f'gpt2_functional_{gpu}', gpu, 'run_gpt2_functional.py', [
            '--manifest', a.frozen_manifest, '--checkpoint', a.model_root / 'gpt2',
            '--output', root / 'functional', '--device', f'cuda:{gpu}', '--svd-driver', 'gesvd',
            '--document-start', lo, '--document-stop', hi, '--lengths', '256,512,1024',
            '--layers', '0,5,11', '--heads', 'all', '--modes', 'ablate,sink-neutralized']),
            cross(f'gpt2_crosscut_{gpu}', 'gpt2', 'gpt2', gpu, a.frozen_manifest, lo, hi,
                  lengths='256,512,1024', extra=['--model-dtype','float32','--alibi-slopes','0.01,0.1'])])
    modern=[cross('retnet_crosscut', 'retnet', 'retnet-1.3B-100B', 2, a.structured_manifest)]
    if a.modern_manifest:
        for name, ckpt in (('bloom','bloom-1b1'),('pythia','pythia-1.4b')):
            modern.append(cross(name+'_crosscut', name, ckpt, 2, a.modern_manifest,
                                extra=['--model-dtype','float32']))
    queues += [modern, [cross('mamba_crosscut', 'mamba2', 'Mamba-Codestral-7B-v0.1', 3,
                             a.structured_manifest, heads='0,16,32,48,64,80,96,112', extra=['--include-with-d'])]]
    plan=dict(root=str(root), working_directory=str(REPO_ROOT), queues=queues,
        analysis_command=[a.python, '-u', str(base/'analyze_rebuttal.py'), 'analyze',
            '--functional-dir', str(root/'functional'), '--crosscut-dir', str(root/'crosscut'),
            '--output-dir', str(root/'analysis')],
        note='Corrected QR/entry-energy protocol; explicit paths supplied by caller; no jobs launched by this generator')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), queues=len(queues))))


if __name__ == '__main__':
    main()
