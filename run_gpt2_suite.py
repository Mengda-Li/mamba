"""Manifest-driven, resumable GPT-2 runner for WP1, WP2, WP3, and WP5.

Every sample is committed as an independent metrics JSONL, compressed array
NPZ, and completion marker.  The marker is written last and binds the sample,
configuration, implementation, file hashes, and array references.  Resuming
therefore never appends duplicate rows and never reads/recompresses a growing
global spectrum archive.

The scalar ``gpt2_metrics.jsonl`` index is a rebuildable view over valid sample
shards.  Each row names its ``array_archive`` and uses stable ``*_ref`` keys to
address arrays within that archive.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
CHECKPOINT_NAME = "gpt2"
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
CHECKPOINT_PATH = os.environ.get("GPT2_MODEL_PATH", str(MODEL_ROOT / "gpt2"))
WP1_LAYERS = (0, 5, 11)
ALL_LAYERS = tuple(range(12))
ALL_HEADS = tuple(range(12))
PREREGISTERED_HEADS = (0, 3, 6, 9)
DEFAULT_LENGTHS = (64, 128, 256, 384, 512, 768, 1024)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any, length: int = 16) -> str:
    return sha256_bytes(canonical_json(value).encode("ascii"))[:length]


def implementation_fingerprints() -> dict[str, str]:
    here = Path(__file__).resolve()
    sources = (here, here.with_name("gpt2_experiments.py"))
    return {path.name: sha256_file(path) for path in sources}


def implementation_id(source_hashes: Mapping[str, str]) -> str:
    return fingerprint(dict(source_hashes))


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in ("numpy", "torch", "transformers", "accelerate"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def protocol_config(
    *,
    checkpoint_revision: str,
    manifest_sha256: str,
    source_hashes: Mapping[str, str],
    environment_versions: Mapping[str, str],
    model_device_map: str,
    analysis_device: str,
    svd_device: str,
    svd_dtype: str,
    checkpoint_path: str = CHECKPOINT_PATH,
) -> dict[str, Any]:
    """Return only result-affecting settings, not work-shard CLI options."""

    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": CHECKPOINT_NAME,
        "checkpoint_path": checkpoint_path,
        "checkpoint_revision": checkpoint_revision,
        "model_device_map": model_device_map,
        "manifest_sha256": manifest_sha256,
        "source_sha256": dict(source_hashes),
        "environment_versions": dict(environment_versions),
        "coverage": {
            "all_layers": list(ALL_LAYERS),
            "all_heads": list(ALL_HEADS),
            "wp1_layers": list(WP1_LAYERS),
            "preregistered_heads": list(PREREGISTERED_HEADS),
            "intensive_diagnostic_heads": list(PREREGISTERED_HEADS),
            "non_wp1_native_spectrum_heads": list(PREREGISTERED_HEADS),
            "wp1_all_heads_full_spectrum_at_all_lengths": True,
            "all_layer_head_pairs_have_exact_cheap_wp2_diagnostics": True,
        },
        "analysis": {
            "analysis_device": analysis_device,
            "analysis_dtype": "float32",
            "svd_device": svd_device,
            "svd_dtype": svd_dtype,
            "store_full_spectra": True,
            "store_variant_spectra": False,
            "compute_nuclear_sink_residual": True,
            "compute_score_localization": True,
            "compute_qk_geometry": True,
            "compute_truncations": True,
            "compute_counterfactuals": True,
            "compute_matched_masks": True,
            "compute_mask_gain": True,
            "alignment_ranks": [1, 2, 4, 8, 16],
            "truncation_ranks": [1, 2, 4, 8, 16, 32],
            "decay_gammas": [0.5, 0.9, 0.99],
            "margin_c_values": [0.5, 0.75, 1.0],
            "fixed_margin_C": 1.0,
            "scale_directions": 8,
            "array_dtype": "float32",
        },
    }


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not result:
        raise argparse.ArgumentTypeError("Expected a nonempty comma-separated integer list.")
    return result


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=True) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=4 * 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_payload(sample: Mapping[str, Any], checkpoint_revision: str) -> dict[str, Any]:
    return {
        "checkpoint": sample["checkpoint"],
        "checkpoint_revision": checkpoint_revision,
        "document_id": sample["document_id"],
        "article_index": int(sample["article_index"]),
        "sample_offset": int(sample["sample_offset"]),
        "input_condition": sample["input_condition"],
        "length": int(sample["length"]),
        "token_ids": [int(value) for value in sample["token_ids"]],
        "text_sha256": sample.get("text_sha256"),
    }


def sample_fingerprint(sample: Mapping[str, Any], checkpoint_revision: str) -> str:
    return fingerprint(sample_payload(sample, checkpoint_revision), length=24)


def safe_component(value: Any, maximum: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (text or "sample")[:maximum]


def sample_id(sample: Mapping[str, Any], checkpoint_revision: str) -> str:
    condition = safe_component(sample["input_condition"], 24)
    document = safe_component(sample["document_id"], 64)
    length = int(sample["length"])
    digest = sample_fingerprint(sample, checkpoint_revision)[:12]
    return f"{document}__{condition}__L{length:04d}__{digest}"


def row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "sample_id",
        "work_package",
        "layer",
        "head_or_channel",
        "operator_variant",
        "truncation_rank",
        "late_spike_position_0based",
    )
    return {key: row.get(key) for key in keys}


def add_row_ids(rows: Sequence[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = fingerprint(row_identity(row), length=24)
        if row_id in seen:
            raise RuntimeError(f"Duplicate row identity within sample: {row_identity(row)}")
        seen.add(row_id)
        row["row_id"] = row_id


def _walk_array_refs(value: Any, refs: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).endswith("_ref") and isinstance(child, str):
                refs.add(child)
            else:
                _walk_array_refs(child, refs)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _walk_array_refs(child, refs)
    elif isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return
        _walk_array_refs(decoded, refs)


def collect_array_refs(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    refs: set[str] = set()
    _walk_array_refs(rows, refs)
    return refs


def dense_head_policy(length: int) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    """Return native-spectrum heads and additional dense-diagnostic heads."""

    del length  # The preregistered coverage is identical at every primary length.
    full = {
        layer: (ALL_HEADS if layer in WP1_LAYERS else PREREGISTERED_HEADS)
        for layer in ALL_LAYERS
    }
    intensive = {
        layer: (PREREGISTERED_HEADS if layer in WP1_LAYERS else ())
        for layer in ALL_LAYERS
    }
    return full, intensive


def validate_analysis_coverage(rows: Sequence[Mapping[str, Any]], length: int) -> None:
    native = {
        (int(row["layer"]), int(row["head_or_channel"])): row
        for row in rows
        if row.get("operator_variant") == "native_softmax"
    }
    expected = {(layer, head) for layer in ALL_LAYERS for head in ALL_HEADS}
    if set(native) != expected:
        missing = sorted(expected.difference(native))
        extra = sorted(set(native).difference(expected))
        raise RuntimeError(f"Native GPT-2 coverage mismatch: missing={missing[:5]} extra={extra[:5]}")
    cheap_required = (
        "sink_S",
        "sink_D",
        "sink_D_over_S",
        "sink_rho2",
        "margin_median",
        "collision_scaled_median",
    )
    full_policy, intensive_policy = dense_head_policy(length)
    for (layer, head), row in native.items():
        missing_metrics = [name for name in cheap_required if name not in row]
        if missing_metrics:
            raise RuntimeError(f"Layer {layer} head {head} lacks cheap WP2 metrics {missing_metrics}")
        should_have_spectrum = head in full_policy[layer]
        if bool(row.get("full_operator_spectrum")) != should_have_spectrum:
            raise RuntimeError(f"Unexpected full-spectrum policy at layer {layer} head {head}")
        if should_have_spectrum:
            for name in ("srank", "erank_1", "erank_2", "spectrum_ref"):
                if name not in row:
                    raise RuntimeError(f"Layer {layer} head {head} lacks WP1 spectrum field {name}")
        should_be_intensive = head in intensive_policy[layer]
        if bool(row.get("intensive_dense_diagnostics")) != should_be_intensive:
            raise RuntimeError(f"Unexpected intensive policy at layer {layer} head {head}")
    for layer in WP1_LAYERS:
        if any("spectrum_ref" not in native[(layer, head)] for head in ALL_HEADS):
            raise RuntimeError(f"WP1 layer {layer} does not have all-head full spectra at length {length}")


def shard_paths(run_root: Path, sample_name: str) -> tuple[Path, Path, Path, Path]:
    return (
        run_root / "metrics" / f"{sample_name}.jsonl",
        run_root / "arrays" / f"{sample_name}.npz",
        run_root / "completed" / f"{sample_name}.json",
        run_root / "locks" / f"{sample_name}.lock",
    )


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def valid_completion_marker(
    marker_path: Path,
    *,
    expected_config_id: str,
    expected_implementation_id: str,
    expected_sample_id: str | None = None,
    expected_sample_fingerprint: str | None = None,
    verify_hashes: bool = True,
) -> tuple[bool, str]:
    if not marker_path.exists():
        return False, "marker_missing"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"marker_unreadable:{error}"
    checks = {
        "schema_version": SCHEMA_VERSION,
        "config_id": expected_config_id,
        "implementation_id": expected_implementation_id,
    }
    if expected_sample_id is not None:
        checks["sample_id"] = expected_sample_id
    if expected_sample_fingerprint is not None:
        checks["sample_fingerprint"] = expected_sample_fingerprint
    for key, expected in checks.items():
        if marker.get(key) != expected:
            return False, f"{key}_mismatch"
    for kind in ("metrics", "arrays"):
        path_text = marker.get(f"{kind}_path")
        if not isinstance(path_text, str):
            return False, f"{kind}_path_missing"
        path = marker_path.parents[1] / path_text
        if not path.is_file():
            return False, f"{kind}_missing"
        if verify_hashes and sha256_file(path) != marker.get(f"{kind}_sha256"):
            return False, f"{kind}_hash_mismatch"
    return True, "ok"


def write_sample_shard(
    *,
    run_root: Path,
    output: Any,
    sample: Mapping[str, Any],
    sample_name: str,
    sample_hash: str,
    config_id: str,
    impl_id: str,
) -> dict[str, Any]:
    metrics_path, arrays_path, marker_path, _lock_path = shard_paths(run_root, sample_name)
    rows = output.rows
    arrays = output.arrays
    validate_analysis_coverage(rows, int(sample["length"]))
    refs = collect_array_refs(rows)
    missing_refs = refs.difference(arrays)
    if missing_refs:
        raise RuntimeError(f"Rows reference missing arrays: {sorted(missing_refs)[:5]}")
    unreferenced = set(arrays).difference(refs)
    if unreferenced:
        raise RuntimeError(f"Analysis produced unreferenced arrays: {sorted(unreferenced)[:5]}")

    relative_archive = arrays_path.relative_to(run_root.parent.parent).as_posix()
    for row in rows:
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "config_id": config_id,
                "implementation_id": impl_id,
                "sample_id": sample_name,
                "sample_fingerprint": sample_hash,
                "array_archive": relative_archive,
                "wp1_rank_scaling_observation": (
                    row.get("operator_variant") == "native_softmax"
                    and int(row["layer"]) in WP1_LAYERS
                ),
                "wp2_sink_observation": row.get("operator_variant") == "native_softmax",
            },
        )
    add_row_ids(rows)
    rows.sort(key=row_sort_key)

    atomic_npz(arrays_path, arrays)
    arrays_sha256 = sha256_file(arrays_path)
    for row in rows:
        row["array_archive_sha256"] = arrays_sha256
    row_count = atomic_jsonl(metrics_path, rows)
    metrics_sha256 = sha256_file(metrics_path)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "config_id": config_id,
        "implementation_id": impl_id,
        "sample_id": sample_name,
        "sample_fingerprint": sample_hash,
        "document_id": sample["document_id"],
        "input_condition": sample["input_condition"],
        "length": int(sample["length"]),
        "metrics_path": metrics_path.relative_to(run_root).as_posix(),
        "arrays_path": arrays_path.relative_to(run_root).as_posix(),
        "metrics_sha256": metrics_sha256,
        "arrays_sha256": arrays_sha256,
        "row_count": row_count,
        "array_count": len(arrays),
        "array_keys_sha256": fingerprint(sorted(arrays), length=64),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(marker_path, marker)
    return marker


def row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("document_id", "")),
        str(row.get("input_condition", "")),
        int(row.get("length", 0)),
        int(row.get("layer", -1)),
        str(row.get("head_or_channel", "")),
        str(row.get("work_package", "")),
        str(row.get("operator_variant", "")),
        int(row.get("truncation_rank") or -1),
        str(row.get("row_id", "")),
    )


def rebuild_metrics_index(
    *,
    run_root: Path,
    out_dir: Path,
    config_id: str,
    impl_id: str,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    # The compatibility index is shared by all config-specific run roots.
    lock_path = out_dir / ".gpt2_metrics_index.lock"
    with exclusive_lock(lock_path):
        valid_markers: list[tuple[Path, dict[str, Any]]] = []
        marker_count = invalid_count = 0
        for marker_path in sorted((run_root / "completed").glob("*.json")):
            valid, _reason = valid_completion_marker(
                marker_path,
                expected_config_id=config_id,
                expected_implementation_id=impl_id,
                verify_hashes=verify_hashes,
            )
            if not valid:
                invalid_count += 1
                continue
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            valid_markers.append((marker_path, marker))
            marker_count += 1

        index_path = run_root / "gpt2_metrics.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(index_path)
        row_hash_by_id: dict[str, str] = {}
        deduplicated_rows = 0
        try:
            with temporary.open("w", encoding="utf-8") as output_handle:
                for _marker_path, marker in valid_markers:
                    metrics_path = run_root / marker["metrics_path"]
                    shard_rows = read_jsonl(metrics_path)
                    if len(shard_rows) != int(marker["row_count"]):
                        raise RuntimeError(f"Row count mismatch in {metrics_path}")
                    for row in shard_rows:
                        row_id = row.get("row_id")
                        if not isinstance(row_id, str):
                            raise RuntimeError(f"Missing row_id in {metrics_path}")
                        line = json.dumps(
                            row,
                            sort_keys=True,
                            ensure_ascii=True,
                            allow_nan=True,
                        )
                        row_hash = sha256_bytes(line.encode("utf-8"))
                        previous_hash = row_hash_by_id.get(row_id)
                        if previous_hash is not None:
                            if previous_hash != row_hash:
                                raise RuntimeError(f"Conflicting duplicate row_id {row_id}")
                            continue
                        row_hash_by_id[row_id] = row_hash
                        output_handle.write(line + "\n")
                        deduplicated_rows += 1
                output_handle.flush()
                os.fsync(output_handle.fileno())
            os.replace(temporary, index_path)
            _fsync_parent(index_path)
        finally:
            temporary.unlink(missing_ok=True)
        compatibility_path = out_dir / "gpt2_metrics.jsonl"
        atomic_copy(index_path, compatibility_path)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "config_id": config_id,
            "implementation_id": impl_id,
            "valid_sample_shards": marker_count,
            "invalid_sample_markers": invalid_count,
            "deduplicated_metric_rows": deduplicated_rows,
            "authoritative_metrics_path": index_path.relative_to(out_dir).as_posix(),
            "compatibility_metrics_path": compatibility_path.relative_to(out_dir).as_posix(),
            "spectra_layout": "per-sample NPZ archives referenced by each metrics row",
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(run_root / "index_summary.json", summary)
        atomic_json(
            out_dir / "gpt2_metrics_manifest.json",
            {
                **summary,
                "run_root": run_root.relative_to(out_dir).as_posix(),
            },
        )
        return summary


def select_samples(
    manifest: Sequence[Mapping[str, Any]],
    *,
    checkpoint_revision: str,
    document_start: int,
    document_stop: int | None,
    lengths: Sequence[int],
    max_samples: int | None,
) -> list[dict[str, Any]]:
    gpt2_rows = [dict(row) for row in manifest if row.get("checkpoint") == CHECKPOINT_NAME]
    documents = sorted({str(row["document_id"]) for row in gpt2_rows})[
        document_start:document_stop
    ]
    allowed_documents = set(documents)
    allowed_lengths = {int(value) for value in lengths}
    selected = [
        row
        for row in gpt2_rows
        if str(row["document_id"]) in allowed_documents and int(row["length"]) in allowed_lengths
    ]
    selected.sort(key=lambda row: (row["document_id"], row["input_condition"], int(row["length"])))
    unique: dict[str, dict[str, Any]] = {}
    for row in selected:
        if len(row["token_ids"]) != int(row["length"]):
            raise ValueError(f"Manifest token count mismatch for {row['document_id']}")
        name = sample_id(row, checkpoint_revision)
        if name in unique and canonical_json(unique[name]) != canonical_json(row):
            raise RuntimeError(f"Conflicting manifest records map to sample id {name}")
        unique[name] = row
    output = list(unique.values())
    if max_samples is not None:
        output = output[:max_samples]
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="experiments_2027/sample_manifest.jsonl")
    parser.add_argument("--out-dir", default="experiments_2027")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--document-start", type=int, default=0)
    parser.add_argument("--document-stop", type=int)
    parser.add_argument("--lengths", type=parse_int_list, default=DEFAULT_LENGTHS)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--analysis-device", default="cpu")
    parser.add_argument(
        "--svd-device",
        default="cuda",
        help="Dense operator SVD device; the analysis module retries failed CUDA SVDs on CPU.",
    )
    parser.add_argument("--svd-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--rebuild-index-only", action="store_true")
    parser.add_argument("--run-id", help="Existing config id to aggregate with --rebuild-index-only.")
    parser.add_argument("--skip-index", action="store_true", help="Leave only authoritative shards.")
    parser.add_argument(
        "--fast-resume",
        action="store_true",
        help="Trust marker metadata and file existence without rehashing completed shards.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_index_only and args.run_id:
        run_root = out_dir / "gpt2_runs" / safe_component(args.run_id)
        config_path = run_root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        summary = rebuild_metrics_index(
            run_root=run_root,
            out_dir=out_dir,
            config_id=stored["config_id"],
            impl_id=stored["implementation_id"],
            verify_hashes=not args.fast_resume,
        )
        print(canonical_json(summary), flush=True)
        return

    manifest_path = Path(args.manifest).resolve()
    manifest = read_jsonl(manifest_path)
    from experiment_common import CHECKPOINT_REVISIONS

    checkpoint_revision = CHECKPOINT_REVISIONS[CHECKPOINT_NAME]
    source_hashes = implementation_fingerprints()
    impl_id = implementation_id(source_hashes)
    config = protocol_config(
        checkpoint_revision=checkpoint_revision,
        manifest_sha256=sha256_file(manifest_path),
        source_hashes=source_hashes,
        environment_versions=runtime_versions(),
        model_device_map=args.device_map,
        analysis_device=args.analysis_device,
        svd_device=args.svd_device,
        svd_dtype=args.svd_dtype,
        checkpoint_path=args.checkpoint,
    )
    config_id = fingerprint(config)
    run_root = out_dir / "gpt2_runs" / config_id
    stored_config = {**config, "config_id": config_id, "implementation_id": impl_id}
    config_path = run_root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(stored_config):
            raise RuntimeError(f"Configuration collision at {config_path}")
    else:
        atomic_json(config_path, stored_config)
    atomic_json(
        out_dir / "gpt2_active_run.json",
        {
            "config_id": config_id,
            "implementation_id": impl_id,
            "run_root": run_root.relative_to(out_dir).as_posix(),
        },
    )

    if args.rebuild_index_only:
        summary = rebuild_metrics_index(
            run_root=run_root,
            out_dir=out_dir,
            config_id=config_id,
            impl_id=impl_id,
            verify_hashes=not args.fast_resume,
        )
        print(canonical_json(summary), flush=True)
        return

    samples = select_samples(
        manifest,
        checkpoint_revision=checkpoint_revision,
        document_start=args.document_start,
        document_stop=args.document_stop,
        lengths=args.lengths,
        max_samples=args.max_samples,
    )
    pending: list[dict[str, Any]] = []
    skipped = 0
    for sample in samples:
        name = sample_id(sample, checkpoint_revision)
        sample_hash = sample_fingerprint(sample, checkpoint_revision)
        _metrics_path, _arrays_path, marker_path, _lock_path = shard_paths(run_root, name)
        complete, _reason = valid_completion_marker(
            marker_path,
            expected_config_id=config_id,
            expected_implementation_id=impl_id,
            expected_sample_id=name,
            expected_sample_fingerprint=sample_hash,
            verify_hashes=not args.fast_resume,
        )
        if complete:
            skipped += 1
        else:
            pending.append(sample)
    print(
        f"GPT-2 config={config_id} selected={len(samples)} pending={len(pending)} skipped={skipped}",
        flush=True,
    )

    model = tokenizer = None
    if pending:
        import numpy as np
        import torch

        from gpt2_experiments import AnalysisConfig, analyze_capture, capture_all_layers, load_gpt2

        model, tokenizer = load_gpt2(
            args.checkpoint,
            device_map=args.device_map,
            local_files_only=True,
        )
        if len(model.transformer.h) != len(ALL_LAYERS):
            raise RuntimeError(f"Expected 12 GPT-2 layers, found {len(model.transformer.h)}")
        if any(block.attn.num_heads != len(ALL_HEADS) for block in model.transformer.h):
            raise RuntimeError("Expected 12 attention heads in every GPT-2 layer")
        analysis_config = AnalysisConfig(**config["analysis"])

        for index, sample in enumerate(pending, start=1):
            name = sample_id(sample, checkpoint_revision)
            sample_hash = sample_fingerprint(sample, checkpoint_revision)
            _metrics_path, _arrays_path, marker_path, lock_path = shard_paths(run_root, name)
            with exclusive_lock(lock_path):
                complete, _reason = valid_completion_marker(
                    marker_path,
                    expected_config_id=config_id,
                    expected_implementation_id=impl_id,
                    expected_sample_id=name,
                    expected_sample_fingerprint=sample_hash,
                    verify_hashes=not args.fast_resume,
                )
                if complete:
                    print(f"[{index}/{len(pending)}] {name} completed by another worker", flush=True)
                    continue

                input_ids = torch.from_numpy(np.asarray(sample["token_ids"], dtype=np.int64)[None])
                if sample["input_condition"] == "explicit-boundary":
                    if tokenizer.eos_token_id is None or int(input_ids[0, 0]) != tokenizer.eos_token_id:
                        raise RuntimeError(f"Explicit-boundary sample {name} lacks GPT-2 boundary token")
                bundle = capture_all_layers(
                    model,
                    input_ids,
                    layer_indices=ALL_LAYERS,
                    capture_native_core=False,
                    offload_to_cpu=True,
                )
                metadata = {
                    "checkpoint": CHECKPOINT_NAME,
                    "checkpoint_path": args.checkpoint,
                    "checkpoint_revision": checkpoint_revision,
                    "corpus": "wikitext-103-validation",
                    "document_id": sample["document_id"],
                    "article_index": int(sample["article_index"]),
                    "sample_offset": int(sample["sample_offset"]),
                    "input_condition": sample["input_condition"],
                    "length": int(sample["length"]),
                    "text_sha256": sample.get("text_sha256"),
                }
                spectrum_heads, intensive_heads = dense_head_policy(int(sample["length"]))
                output = analyze_capture(
                    model,
                    bundle,
                    config=analysis_config,
                    metadata=metadata,
                    heads_by_layer={layer: ALL_HEADS for layer in ALL_LAYERS},
                    full_spectrum_heads_by_layer=spectrum_heads,
                    intensive_heads_by_layer=intensive_heads,
                )
                marker = write_sample_shard(
                    run_root=run_root,
                    output=output,
                    sample=sample,
                    sample_name=name,
                    sample_hash=sample_hash,
                    config_id=config_id,
                    impl_id=impl_id,
                )
                print(
                    f"[{index}/{len(pending)}] {name} rows={marker['row_count']} "
                    f"arrays={marker['array_count']}",
                    flush=True,
                )
                del bundle, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not args.skip_index:
        summary = rebuild_metrics_index(
            run_root=run_root,
            out_dir=out_dir,
            config_id=config_id,
            impl_id=impl_id,
            verify_hashes=not args.fast_resume,
        )
        print(canonical_json(summary), flush=True)


if __name__ == "__main__":
    main()
