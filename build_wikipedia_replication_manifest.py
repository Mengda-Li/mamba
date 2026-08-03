"""Build a deterministic English Wikipedia manifest for post-plan replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from experiment_common import PRIMARY_LENGTHS, write_jsonl


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
TOKENIZER_PATHS = {
    "mamba2": os.environ.get(
        "MAMBA2_MODEL_PATH",
        str(MODEL_ROOT / "Mamba-Codestral-7B-v0.1"),
    ),
    "retnet": os.environ.get("RETNET_MODEL_PATH", str(MODEL_ROOT / "retnet-1.3B-100B")),
}
DEFAULT_WIKIPEDIA_PARQUET = os.environ.get(
    "WIKIPEDIA_PARQUET",
    str(DATA_ROOT / "wikipedia" / "20231101.en" / "train-00000-of-00041.parquet"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=DEFAULT_WIKIPEDIA_PARQUET)
    parser.add_argument("--mamba2-checkpoint", default=TOKENIZER_PATHS["mamba2"])
    parser.add_argument("--retnet-checkpoint", default=TOKENIZER_PATHS["retnet"])
    parser.add_argument("--out-dir", default="experiments_2027_wikipedia_replication")
    parser.add_argument("--documents", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20270803)
    args = parser.parse_args()

    tokenizer_paths = {
        "mamba2": args.mamba2_checkpoint,
        "retnet": args.retnet_checkpoint,
    }
    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for name, path in tokenizer_paths.items()
    }
    table = pq.read_table(args.parquet, columns=["id", "title", "text"])
    candidates = []
    for index, row in enumerate(table.to_pylist()):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        lengths = [
            len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])
            for tokenizer in tokenizers.values()
        ]
        if min(lengths) >= max(PRIMARY_LENGTHS) + 32:
            candidates.append((index, str(row["id"]), str(row.get("title") or ""), text))
        if len(candidates) >= 512:
            break
    if len(candidates) < args.documents:
        raise RuntimeError(f"Only {len(candidates)} qualifying Wikipedia articles found")

    selected = random.Random(args.seed).sample(candidates, args.documents)
    selected.sort(key=lambda item: item[0])
    records = []
    for article_index, source_id, title, text in selected:
        digest = hashlib.sha256(source_id.encode()).hexdigest()[:12]
        document_id = f"wikipedia-en-{source_id}-{digest}"
        for model, tokenizer in tokenizers.items():
            ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
            for length in PRIMARY_LENGTHS:
                used = [int(value) for value in ids[:length]]
                decoded = tokenizer.decode(used, skip_special_tokens=False)
                records.append(
                    {
                        "checkpoint": model,
                        "document_id": document_id,
                        "title": title,
                        "article_index": article_index,
                        "source_row_start": article_index,
                        "source_row_end": article_index + 1,
                        "sample_offset": 0,
                        "input_condition": "native-prefix",
                        "length": length,
                        "token_ids": used,
                        "decoded_text": decoded,
                        "text_sha256": hashlib.sha256(decoded.encode()).hexdigest(),
                    }
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / "sample_manifest.jsonl")
    metadata = {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "source_parquet": args.parquet,
        "tokenizer_paths": tokenizer_paths,
        "seed": args.seed,
        "document_count": args.documents,
        "selection": "Seeded sample from the first 512 articles with >=1056 tokens under both tokenizers.",
        "scope": "Post-plan cross-corpus replication of final-layer WP1 and WP5.",
    }
    (out_dir / "replication_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} records for {args.documents} Wikipedia articles")


if __name__ == "__main__":
    main()
