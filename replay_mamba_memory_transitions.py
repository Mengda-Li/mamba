"""Replay frozen Mamba inputs and persist transition arrays for exact audits.

This utility is intentionally separate from the fingerprinted production
runner. It recreates only the native and deterministically shuffled transition
sequences needed by ``audit_structured_memory_svd.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from experiment_common import CHECKPOINT_REVISIONS, HEADS, LAYERS, read_jsonl


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
MODEL_PATH = os.environ.get(
    "MAMBA2_MODEL_PATH",
    str(MODEL_ROOT / "Mamba-Codestral-7B-v0.1"),
)
VARIANTS = ("native", "transition_shuffled")


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def selected_for_exact_audit(document_id: str, layer: int, head: int, length: int) -> bool:
    if length <= 512:
        return True
    if length != 1024:
        return False
    payload = {
        "checkpoint": "mamba2",
        "document_id": document_id,
        "layer": layer,
        "head_or_channel": head,
    }
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return int(digest[:8], 16) % 10 == 0


def sample_token_sha256(sample: dict) -> str:
    tokens = [int(token) for token in sample["token_ids"]]
    return hashlib.sha256(canonical_json(tokens).encode()).hexdigest()


def stable_observation_id(sample: dict, layer: int, head: int, variant: str) -> str:
    identity = {
        "checkpoint": "mamba2",
        "checkpoint_revision": CHECKPOINT_REVISIONS["mamba2"],
        "corpus": "wikitext-103-validation",
        "document_id": sample["document_id"],
        "sample_offset": sample["sample_offset"],
        "input_condition": sample["input_condition"],
        "length": sample["length"],
        "sample_token_sha256": sample_token_sha256(sample),
        "layer": layer,
        "head_or_channel": head,
        "operator_variant": variant,
        "schema_version": 3,
    }
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:32]


def load_mamba(device: str, checkpoint: str = MODEL_PATH):
    from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

    model = Mamba2ForCausalLM.from_pretrained(
        checkpoint,
        dtype="auto",
        device_map=device,
        local_files_only=True,
    ).eval()
    for block in model.backbone.layers:
        block.mixer.chunk_size = 64
    return model


def selected_samples(manifest_path: Path) -> list[dict]:
    rows = [
        row
        for row in read_jsonl(manifest_path)
        if row["checkpoint"] == "mamba2"
        and row["input_condition"] == "native-prefix"
        and (row["length"] <= 512 or row["length"] == 1024)
    ]
    return sorted(rows, key=lambda row: (row["document_id"], row["length"]))


def expected_keys(sample: dict, layer: int) -> set[str]:
    return {
        f"transition-{stable_observation_id(sample, layer, head, variant)}"
        for head in HEADS["mamba2"]
        if selected_for_exact_audit(sample["document_id"], layer, head, sample["length"])
        for variant in VARIANTS
    }


def archive_path(output: Path, sample: dict, layer: int) -> Path:
    identity = canonical_json(
        {
            "document_id": sample["document_id"],
            "length": sample["length"],
            "layer": layer,
        }
    )
    unit = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return output / "structured_shards" / "mamba2" / "transitions" / f"{unit}.npz"


def valid_archive(path: Path, keys: set[str], metric_ids: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != keys:
                return False
            return all(
                key.removeprefix("transition-") in metric_ids
                and archive[key].ndim == 1
                and np.isfinite(archive[key]).all()
                for key in archive.files
            )
    except Exception:
        return False


def metric_observation_ids(results_dir: Path) -> set[str]:
    identifiers: set[str] = set()
    for path in sorted((results_dir / "structured_shards" / "mamba2" / "metrics").glob("*.parquet")):
        identifiers.update(pd.read_parquet(path, columns=["observation_id"])["observation_id"].astype(str))
    return identifiers


def write_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def pending_layers(
    sample: dict,
    output: Path,
    metric_ids: set[str],
    layers=LAYERS["mamba2"],
) -> list[tuple[int, set[str], Path]]:
    pending: list[tuple[int, set[str], Path]] = []
    for layer in layers:
        keys = expected_keys(sample, layer)
        if not keys:
            continue
        path = archive_path(output, sample, layer)
        if valid_archive(path, keys, metric_ids):
            continue
        if not all(key.removeprefix("transition-") in metric_ids for key in keys):
            raise RuntimeError(
                f"Cannot replay before the production work unit is complete: "
                f"{sample['document_id']} length={sample['length']} layer={layer}"
            )
        pending.append((layer, keys, path))
    return pending


def pending_units(
    samples: list[dict],
    output: Path,
    metric_ids: set[str],
    layers=LAYERS["mamba2"],
) -> list[tuple[str, int]]:
    units: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for sample in samples:
        unit = (str(sample["document_id"]), int(sample["length"]))
        if unit in seen:
            raise RuntimeError(f"Duplicate Mamba replay work unit in manifest: {unit}")
        seen.add(unit)
        if pending_layers(sample, output, metric_ids, layers):
            units.append(unit)
    return units


def select_sample(samples: list[dict], document_id: str, length: int) -> dict:
    matches = [
        sample
        for sample in samples
        if str(sample["document_id"]) == document_id and int(sample["length"]) == length
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Mamba replay sample for document={document_id!r}, "
            f"length={length}; found {len(matches)}"
        )
    return matches[0]


def worker_command(
    python_executable: str,
    script_path: Path,
    manifest: Path,
    output: Path,
    device: str,
    unit: tuple[str, int],
    checkpoint: str = MODEL_PATH,
) -> list[str]:
    document_id, length = unit
    return [
        python_executable,
        str(script_path),
        "--manifest",
        str(manifest),
        "--out-dir",
        str(output),
        "--device",
        device,
        "--checkpoint",
        checkpoint,
        "--worker",
        "--document-id",
        document_id,
        "--length",
        str(length),
    ]


def run_isolated_units(
    units: list[tuple[str, int]],
    python_executable: str,
    script_path: Path,
    manifest: Path,
    output: Path,
    device: str,
    checkpoint: str = MODEL_PATH,
    runner=subprocess.run,
) -> None:
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for index, unit in enumerate(units, start=1):
        document_id, length = unit
        print(
            f"[{index}/{len(units)}] starting isolated replay "
            f"{document_id} L={length}",
            flush=True,
        )
        runner(
            worker_command(
                python_executable,
                script_path,
                manifest,
                output,
                device,
                unit,
                checkpoint,
            ),
            check=True,
            env=environment,
        )


def replay_sample(
    sample: dict,
    pending: list[tuple[int, set[str], Path]],
    device: str,
    checkpoint: str = MODEL_PATH,
) -> None:
    from compare_attention_ranks_wikitext import capture_module_input
    from extract_mamba_matrices import extract_mamba2_matrices

    model = load_mamba(device, checkpoint)
    selected_heads = list(HEADS["mamba2"])
    ids = torch.tensor([sample["token_ids"]], device=model.get_input_embeddings().weight.device)
    encoded = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    for index, (layer, keys, path) in enumerate(pending, start=1):
        mixer = model.backbone.layers[layer].mixer
        hidden, mask = capture_module_input(model, mixer, encoded)
        with torch.inference_mode():
            matrices = extract_mamba2_matrices(
                mixer,
                hidden,
                attention_mask=mask,
                selected_heads=selected_heads,
            )
        arrays: dict[str, np.ndarray] = {}
        for local_head, head in enumerate(selected_heads):
            if not selected_for_exact_audit(sample["document_id"], layer, head, sample["length"]):
                continue
            native = matrices.a[0, local_head]
            generator = torch.Generator(device=native.device).manual_seed(
                20270802 + layer * 1000 + head * 10 + sample["article_index"]
            )
            shuffled = native.clone()
            if sample["length"] > 1:
                permutation = torch.randperm(
                    sample["length"] - 1,
                    generator=generator,
                    device=native.device,
                ) + 1
                shuffled[1:] = native[permutation]
            for variant, transitions in (("native", native), ("transition_shuffled", shuffled)):
                observation = stable_observation_id(sample, layer, head, variant)
                arrays[f"transition-{observation}"] = transitions.detach().float().cpu().numpy()
        if set(arrays) != keys:
            raise RuntimeError(f"Replay key mismatch for {path}: expected={len(keys)}, observed={len(arrays)}")
        write_archive(path, arrays)
        print(
            f"[{index}/{len(pending)}] {sample['document_id']} "
            f"L={sample['length']} layer={layer}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments_2027/sample_manifest.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments_2027"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=MODEL_PATH)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--document-id", help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    samples = selected_samples(args.manifest)
    metric_ids = metric_observation_ids(args.out_dir)
    if not metric_ids:
        raise RuntimeError("No completed Mamba structured metric shards are available.")

    if args.worker:
        if args.document_id is None or args.length is None:
            parser.error("--worker requires --document-id and --length")
        sample = select_sample(samples, args.document_id, args.length)
        pending = pending_layers(sample, args.out_dir, metric_ids)
        if not pending:
            print(
                f"Replay archive already complete for {args.document_id} L={args.length}.",
                flush=True,
            )
            return
        replay_sample(sample, pending, args.device, args.checkpoint)
        return

    if args.document_id is not None or args.length is not None:
        parser.error("--document-id and --length are internal worker arguments")
    units = pending_units(samples, args.out_dir, metric_ids)
    if not units:
        print("All required Mamba transition replay archives already exist.")
        return
    run_isolated_units(
        units,
        sys.executable,
        Path(__file__).resolve(),
        args.manifest,
        args.out_dir,
        args.device,
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
