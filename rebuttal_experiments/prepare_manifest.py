"""Retokenize the submitted document set without selecting favorable new documents."""
from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import hashlib
import json
import sys
from pathlib import Path

LEGACY = REPO_ROOT
sys.path.insert(0, str(LEGACY))


def main() -> None:
    from experiment_common import load_wikitext_documents, prefix_record, read_jsonl
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-manifest', required=True)
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--checkpoint', action='append', required=True, help='label=local_path')
    parser.add_argument('--max-length', action='append', default=[], help='label=integer')
    parser.add_argument('--lengths', default='64,128,256,384,512,768,1024,2048')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    frozen = read_jsonl(args.frozen_manifest)
    document_ids = {r['document_id'] for r in frozen}
    documents = [d for d in load_wikitext_documents(args.parquet)
                 if d.document_id in document_ids]
    if {d.document_id for d in documents} != document_ids:
        raise RuntimeError('Original frozen document IDs do not match dataset reconstruction')
    reference = {(r['checkpoint'], r['document_id'], int(r['length'])): r
                 for r in frozen if r['input_condition'] == 'native-prefix'}
    caps = {n: int(v) for n, v in (x.split('=', 1) for x in args.max_length)}
    lengths = sorted({int(x) for x in args.lengths.split(',')})
    output, skipped, provenance = [], [], {}
    for label, checkpoint in (x.split('=', 1) for x in args.checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        cfg = Path(checkpoint) / 'config.json'
        provenance[label] = {'checkpoint': checkpoint, 'config_sha256':
                             hashlib.sha256(cfg.read_bytes()).hexdigest()}
        for doc in documents:
            count = len(tokenizer(doc.text, add_special_tokens=False, verbose=False)['input_ids'])
            for length in lengths:
                if length > caps.get(label, max(lengths)):
                    continue
                if count < length:
                    skipped.append({'checkpoint': label, 'document_id': doc.document_id,
                                    'length': length, 'available_tokens': count})
                    continue
                record = prefix_record(doc, tokenizer, label, length, 'native-prefix')
                old = reference.get((label, doc.document_id, length))
                if old and old['token_ids'] != record['token_ids']:
                    raise RuntimeError(f'Tokenization changed for {label}/{doc.document_id}/{length}')
                output.append(record)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in output)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(payload)
    temporary.replace(target)
    report = {'schema_version': 1, 'document_count': len(documents), 'records': len(output),
              'frozen_manifest_sha256': hashlib.sha256(Path(args.frozen_manifest).read_bytes()).hexdigest(),
              'parquet_sha256': hashlib.sha256(Path(args.parquet).read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'checkpoints': provenance, 'lengths': lengths, 'max_length': caps,
              'skipped_short_documents': skipped,
              'document_selection': 'exactly submitted document IDs; no replacement for short documents'}
    target.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'records': len(output), 'documents': len(documents),
                      'skipped': len(skipped), 'output': str(target)}), flush=True)


if __name__ == '__main__':
    main()
