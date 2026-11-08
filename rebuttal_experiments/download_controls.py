"""Fetch only public inference files for the optional modern softmax controls."""
from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import json
from pathlib import Path


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', required=True)
    args = parser.parse_args()
    root = Path(args.model_root)
    api = HfApi(token=False)
    for repo, name in [('bigscience/bloom-1b1', 'bloom-1b1'),
                       ('EleutherAI/pythia-1.4b', 'pythia-1.4b')]:
        print(json.dumps({'event': 'resolving', 'repo': repo}), flush=True)
        info = api.model_info(repo, token=False)
        revision = info.sha
        files = api.list_repo_files(repo, revision=revision, token=False)
        weights = ['*.safetensors', '*.safetensors.index.json'] if any(
            p.endswith('.safetensors') for p in files
        ) else ['pytorch_model*.bin', 'pytorch_model*.bin.index.json']
        allow = ['config.json', 'generation_config.json', 'tokenizer*',
                 'vocab.json', 'merges.txt', 'special_tokens_map.json',
                 'added_tokens.json', 'README.md', *weights]
        destination = root / name
        snapshot_download(repo, revision=revision, local_dir=destination,
                          allow_patterns=allow, max_workers=4, token=False)
        (destination / 'rebuttal_download_provenance.json').write_text(
            json.dumps({'repo': repo, 'revision': revision,
                        'allow_patterns': allow}, indent=2) + '\n')
        print(json.dumps({'event': 'complete', 'repo': repo,
                          'revision': revision, 'path': str(destination)}), flush=True)


if __name__ == '__main__':
    main()
