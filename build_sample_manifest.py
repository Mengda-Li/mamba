"""Build the document-disjoint nested-prefix manifest used by all experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from experiment_common import (
    CHECKPOINT_REVISIONS,
    PRIMARY_LENGTHS,
    choose_documents,
    load_wikitext_documents,
    prefix_record,
    write_jsonl,
)


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
DEFAULT_PATHS = {
    "gpt2": os.environ.get("GPT2_MODEL_PATH", str(MODEL_ROOT / "gpt2")),
    "mamba2": os.environ.get(
        "MAMBA2_MODEL_PATH",
        str(MODEL_ROOT / "Mamba-Codestral-7B-v0.1"),
    ),
    "retnet": os.environ.get("RETNET_MODEL_PATH", str(MODEL_ROOT / "retnet-1.3B-100B")),
    "based": os.environ.get("BASED_MODEL_PATH", str(MODEL_ROOT / "based-360m")),
}
DEFAULT_WIKITEXT_PARQUET = os.environ.get(
    "WIKITEXT_PARQUET",
    str(DATA_ROOT / "wikitext" / "wikitext-103-raw-v1" / "validation-00000-of-00001.parquet"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default=DEFAULT_WIKITEXT_PARQUET,
    )
    parser.add_argument("--gpt2-checkpoint", default=DEFAULT_PATHS["gpt2"])
    parser.add_argument("--mamba2-checkpoint", default=DEFAULT_PATHS["mamba2"])
    parser.add_argument("--retnet-checkpoint", default=DEFAULT_PATHS["retnet"])
    parser.add_argument("--based-checkpoint", default=DEFAULT_PATHS["based"])
    parser.add_argument("--out-dir", default="experiments_2027")
    parser.add_argument("--documents", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20270802)
    args = parser.parse_args()

    checkpoint_paths = {
        "gpt2": args.gpt2_checkpoint,
        "mamba2": args.mamba2_checkpoint,
        "retnet": args.retnet_checkpoint,
        "based": args.based_checkpoint,
    }
    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for name, path in checkpoint_paths.items()
        if name != "based"
    }
    documents = choose_documents(
        load_wikitext_documents(args.parquet),
        tokenizers,
        count=args.documents,
        minimum_tokens=max(PRIMARY_LENGTHS) + 32,
        seed=args.seed,
    )
    records = []
    for document in documents:
        for model in ("gpt2", "based", "mamba2", "retnet"):
            tokenizer = tokenizers["gpt2"] if model == "based" else tokenizers[model]
            conditions = ("native-prefix", "explicit-boundary") if model == "gpt2" else ("native-prefix",)
            for condition in conditions:
                for length in PRIMARY_LENGTHS:
                    records.append(prefix_record(document, tokenizer, model, length, condition))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / "sample_manifest.jsonl")
    checkpoint_manifest = {
        "revisions": CHECKPOINT_REVISIONS,
        "paths": checkpoint_paths,
        "source_parquet": args.parquet,
        "sample_seed": args.seed,
        "document_count": len(documents),
        "article_indices": [document.article_index for document in documents],
        "selection_rule": (
            "Uniform seeded sample without replacement among WikiText-103 validation articles "
            "having at least 1056 tokens under GPT-2, RetNet, and Mamba tokenizers."
        ),
    }
    with (out_dir / "checkpoint_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(checkpoint_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {len(records)} records for {len(documents)} documents")


if __name__ == "__main__":
    main()
