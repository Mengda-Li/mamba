#!/usr/bin/env python3
"""Exact full-mixer and past-to-future diagnostics; no functional interventions.

Use a separate output directory for each model/configuration. Existing
``sample_manifest.jsonl`` records are accepted; shorter nested prefixes can be
derived from a document's longest stored native-prefix record. Per-document,
length, layer shards are atomic and independently resumable. No dense matrices
or document text are written. Run --self-test before checkpoint experiments.

Cross-cut convention (zero-based Python): H=M[t+gap:L, :t]. A window block is
M[t+gap:t+gap+w, t-w:t], emitted only when both complete windows fit. Thus gap=0
excludes the diagonal already. Entropy-ESS uses unsquared singular values;
erank_2 uses squared singular values. Zero-matrix entropy ranks are DEFINED AS
ZERO by convention, and zero-denominator gains/fractions are null.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

import torch
from spectral_qc import exact_svdvals, entry_energy_qc

SCHEMA_VERSION = 2
ROOT = REPO_ROOT
EXPERIMENT_CODE = REPO_ROOT
sys.path.insert(0, str(EXPERIMENT_CODE))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def parse_csv(text: str, cast=int) -> list:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def indices(text: str, total: int) -> list[int]:
    if text == "all":
        return list(range(total))
    values = [x % total if -total <= x < 0 else x for x in parse_csv(text)]
    if not values or any(x < 0 or x >= total for x in values):
        raise ValueError(f"Invalid indices {text!r} for size {total}")
    return sorted(set(values))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def spectral_metrics(matrix: torch.Tensor, args) -> dict:
    """Exact dense SVD; FP64 accumulation of computed singular values."""
    work = matrix.detach().to(device=args.svd_device, dtype=getattr(torch, args.svd_dtype)).contiguous()
    if not bool(torch.isfinite(work).all()):
        raise ValueError("Nonfinite matrix: do not silently drop singular values")
    shape = list(work.shape)
    if min(shape) < 1:
        raise ValueError(f"Empty operator block {shape}")
    # Values only, one head/block at a time; do not use batched Jacobi SVD.
    driver = getattr(args, "svd_driver", "gesvd")
    singular = exact_svdvals(work, driver=driver).double().cpu()
    qc = entry_energy_qc(work, singular, driver=driver,
                         tolerance=getattr(args, "svd_energy_tolerance", 1e-5))
    square = singular.square()
    total = float(square.sum())
    top = float(singular[0])
    nonzero = top > 0.0
    result = {
        "shape": shape, "rank_dimension": min(shape), "svd_dtype": args.svd_dtype,
        "svd_method": "torch.linalg.svdvals_exact", "zero_matrix": not nonzero,
        **qc,
        "zero_entropy_convention": "0 (entropy undefined for zero spectrum)",
        "operator_norm": top, "frobenius_norm": qc["entry_frobenius_norm"],
        "nuclear_norm": float(singular.sum()), "srank": 0.0,
        "erank_1": 0.0, "erank_2": 0.0,
        "top16_singular_values": singular[:16].tolist(),
        "tolerance_ess": {format(t, ".12g"): int((singular > t).sum()) for t in args.abs_tolerances},
        "relative_tolerance_rank": {
            format(t, ".12g"): int((singular > t * top).sum()) if nonzero else 0
            for t in args.relative_tolerances
        },
    }
    if nonzero:
        p1, p2 = singular / singular.sum(), square / square.sum()
        result.update({
            "srank": qc["entry_squared_energy"] / top**2,
            "erank_1": float((-(p1[p1 > 0] * p1[p1 > 0].log()).sum()).exp()),
            "erank_2": float((-(p2[p2 > 0] * p2[p2 > 0].log()).sum()).exp()),
        })
    for name in ("srank", "erank_1", "erank_2"):
        result[name + "_fraction"] = result[name] / min(shape)
    return result


def value_metrics(block: torch.Tensor, value: torch.Tensor | None, col_start: int,
                  col_stop: int, *, value_definition: str) -> dict:
    """Core value path only; excludes output projection, gate, norm, residual."""
    if value is None:
        return {"value_available": False, "value_definition": value_definition}
    source = value[col_start:col_stop].to(device=block.device, dtype=block.dtype)
    output = block @ source
    source_norm = float(torch.linalg.vector_norm(source.double()))
    output_norm = float(torch.linalg.vector_norm(output.double()))
    return {
        "value_available": True, "value_definition": value_definition,
        "value_input_frobenius_norm": source_norm,
        "value_output_frobenius_norm": output_norm,
        "value_output_to_input_gain": safe_ratio(output_norm, source_norm),
        "value_input_is_zero": source_norm == 0.0,
        "value_path_scope": "single head core; excludes output projection, gate, norm, residual",
    }


def analyze_operator(matrix: torch.Tensor, value: torch.Tensor | None, base: dict,
                     args, *, value_definition: str) -> list[dict]:
    matrix = matrix.to(device=args.svd_device, dtype=getattr(torch, args.svd_dtype))
    length = matrix.shape[0]
    if tuple(matrix.shape) != (length, length):
        raise ValueError("Expected square complete mixer")
    full = spectral_metrics(matrix, args)
    full_energy = full["entry_squared_energy"]
    positions = torch.arange(length, device=matrix.device)
    lag = positions[:, None] - positions[None, :]
    square = matrix.double().square()
    common = {**base, "status": "ok", "full_frobenius_norm": full["frobenius_norm"],
              "full_operator_is_zero": full["zero_matrix"]}
    full_row = {
        **common, "region": "full", "cut_fraction": None, "cut_index": None, "gap": None,
        "row_start": 0, "row_stop": length, "col_start": 0, "col_stop": length,
        **full, "frobenius_energy_fraction_full": safe_ratio(full_energy, full_energy),
        "diagonal_energy_fraction": safe_ratio(float(square.diag().sum()), full_energy),
        "strict_past_energy_fraction": safe_ratio(float(square[lag > 0].sum()), full_energy),
        "future_energy_fraction": safe_ratio(float(square[lag < 0].sum()), full_energy),
        "lag_energy_fractions": {
            str(g): safe_ratio(float(square[lag > g].sum()), full_energy) for g in args.gaps
        },
        **value_metrics(matrix, value, 0, length, value_definition=value_definition),
    }
    rows = [full_row]
    for fraction in args.cut_fractions:
        cut = int(math.floor(fraction * length))
        if not 0 < cut < length:
            continue
        for gap in args.gaps:
            start = cut + gap
            if start >= length:
                continue
            ranges = [("crosscut", start, length, 0, cut)]
            if cut >= args.window and start + args.window <= length:
                ranges.append(("window", start, start + args.window, cut - args.window, cut))
            for region, rs, re, cs, ce in ranges:
                block = matrix[rs:re, cs:ce]
                metrics = spectral_metrics(block, args)
                rows.append({
                    **common, "region": region, "cut_fraction": fraction,
                    "cut_index": cut, "gap": gap, "window": args.window if region == "window" else None,
                    "row_start": rs, "row_stop": re, "col_start": cs, "col_stop": ce,
                    **metrics,
                    "frobenius_energy_fraction_full": safe_ratio(metrics["entry_squared_energy"], full_energy),
                    **value_metrics(block, value, cs, ce, value_definition=value_definition),
                })
    return rows


def read_documents(args) -> list[dict]:
    longest: dict[str, dict] = {}
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        family = row.get("checkpoint", row.get("model"))
        if family != args.model or row.get("input_condition", "native-prefix") != args.input_condition:
            continue
        document = str(row["document_id"])
        if document not in longest or len(row["token_ids"]) > len(longest[document]["token_ids"]):
            longest[document] = row
    if not longest:
        raise ValueError(f"No {args.model}/{args.input_condition} records in manifest")
    documents = [{**record, "document_index": index} for index, record in enumerate(longest.values())]
    return documents[args.doc_start:args.doc_stop]


def provenance(args) -> tuple[dict, str]:
    config = vars(args).copy()
    for key in ("doc_start", "doc_stop", "output", "device", "svd_device", "overwrite_stale", "self_test", "consolidate_only"):
        config.pop(key, None)
    sources = [Path(__file__), Path(__file__).with_name("spectral_qc.py"),
               Path(__file__).with_name("repo_paths.py"), EXPERIMENT_CODE / "gpt2_experiments.py",
               EXPERIMENT_CODE / "extract_mamba_matrices.py",
               EXPERIMENT_CODE / "compare_attention_ranks_wikitext.py"]
    checkpoint = Path(args.checkpoint)
    checkpoint_files = {}
    for pattern in ("config.json", "generation_config.json", "*.index.json"):
        for path in sorted(checkpoint.glob(pattern)):
            checkpoint_files[path.name] = file_sha(path)
    weights = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
               for pattern in ("*.safetensors", "pytorch_model*.bin") for p in sorted(checkpoint.glob(pattern))]
    metadata = {
        "schema_version": SCHEMA_VERSION, "config": config,
        "manifest_sha256": file_sha(Path(args.manifest)),
        "source_sha256": {p.name: file_sha(p) for p in sources if p.exists()},
        "checkpoint_config_sha256": checkpoint_files, "checkpoint_weight_inventory": weights,
        "checkpoint_weight_hash_note": "weight inventory is provenance, not a full cryptographic hash of all weights",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "block_convention": "H=M[t+gap:L,:t]; window=M[t+gap:t+gap+w,t-w:t]",
        "normalization": "none before SVD; erank_1 uses sigma, erank_2 uses sigma^2",
        "zero_convention": "all ranks zero for zero operator; undefined gain/fraction is null",
    }
    fingerprint = digest(metadata)
    metadata["run_fingerprint"] = fingerprint
    return metadata, fingerprint


def shard_path(args, sample: dict, length: int, layer: int) -> Path:
    doc_hash = digest(str(sample["document_id"]))[:16]
    return Path(args.output) / "shards" / f"{args.model}.{doc_hash}.L{length}.layer{layer}.jsonl"


def is_complete(path: Path, fingerprint: str, args) -> bool:
    marker = path.with_suffix(".complete.json")
    if not path.exists() or not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("run_fingerprint") != fingerprint:
        if args.overwrite_stale:
            return False
        raise ValueError(f"Stale shard {path}; use a fresh output directory or --overwrite-stale")
    return record.get("sha256") == file_sha(path)


def save_shard(path: Path, rows: list[dict], fingerprint: str) -> None:
    atomic_jsonl(path, rows)
    atomic_json(path.with_suffix(".complete.json"), {
        "run_fingerprint": fingerprint, "sha256": file_sha(path), "rows": len(rows), "status": "complete",
    })


def consolidate(args, fingerprint: str) -> int:
    rows = []
    for path in sorted((Path(args.output) / "shards").glob(f"{args.model}.*.jsonl")):
        if is_complete(path, fingerprint, args):
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    atomic_jsonl(Path(args.output) / "metrics.jsonl", rows)
    return len(rows)


def base_row(args, sample: dict, length: int, layer: int, head: int, fingerprint: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "run_fingerprint": fingerprint,
        "model": args.model, "checkpoint": args.model, "checkpoint_path": args.checkpoint,
        "document_id": sample["document_id"], "document_index": sample["document_index"],
        "corpus": sample.get("corpus", "manifest-defined"), "input_condition": args.input_condition,
        "length": length, "layer": layer, "head": head, "head_or_channel": head,
        "token_ids_sha256": digest(sample["token_ids"][:length]), "sample_offset": sample.get("sample_offset", 0),
    }


def run_gpt2(args, samples, fingerprint):
    from gpt2_experiments import capture_all_layers, causal_softmax_from_scores, load_gpt2, score_denominator
    model, _ = load_gpt2(args.checkpoint, device_map=args.device, dtype=args.model_dtype)
    layers = indices(args.layers, len(model.transformer.h))
    context_limit = getattr(model.config, "max_position_embeddings", model.config.n_positions)
    for sample in samples:
        for length in args.lengths:
            if length > len(sample["token_ids"]):
                print(f"skip short document {sample['document_id']} L={length}", flush=True)
                continue
            if length > context_limit:
                raise ValueError(f"GPT2 length {length} exceeds native context limit {context_limit}")
            pending = [layer for layer in layers if not is_complete(shard_path(args, sample, length, layer), fingerprint, args)]
            if not pending:
                continue
            ids = torch.tensor([sample["token_ids"][:length]], dtype=torch.long)
            bundle = capture_all_layers(model, ids, layer_indices=pending, capture_native_core=True)
            for layer in pending:
                capture, attention = bundle.layers[layer], model.transformer.h[layer].attn
                heads = indices(args.heads, attention.num_heads)
                rows = []
                for head in heads:
                    q, k, v = [x[0, head].to(device=args.svd_device, dtype=torch.float32)
                               for x in (capture.query, capture.key, capture.value)]
                    score = q @ k.T / score_denominator(attention, layer)
                    matrix = causal_softmax_from_scores(score)
                    native_core = capture.native_core[0, head].to(device=matrix.device, dtype=matrix.dtype)
                    err = float(torch.linalg.vector_norm((matrix @ v - native_core).double()))
                    ref = float(torch.linalg.vector_norm(native_core.double()))
                    base = {**base_row(args, sample, length, layer, head, fingerprint),
                            "native_core_reconstruction_relative_error": safe_ratio(err, ref),
                            "model_dtype": str(capture.query.dtype), "operator_build_dtype": "float32"}
                    variants = [("native", matrix)]
                    if args.alibi_slopes:
                        positions = torch.arange(length, device=score.device)
                        lag = positions[:, None] - positions[None, :]
                        variants.extend((f"alibi_m{m:g}", causal_softmax_from_scores(score - m * lag))
                                        for m in args.alibi_slopes)
                    for name, operator in variants:
                        rows.extend(analyze_operator(operator, v, {**base, "operator_variant": name,
                            "operator_definition": "causal row softmax of fixed native scaled QK logits" +
                            (" minus m*(i-j); operator counterfactual, not an ALiBi-trained model" if name != "native" else "")},
                            args, value_definition="native projected V"))
                path = shard_path(args, sample, length, layer)
                save_shard(path, rows, fingerprint)
                print(f"saved {path.name} ({len(rows)} rows)", flush=True)
            del bundle


def run_modern_softmax(args, samples, fingerprint):
    """Native eager BLOOM/Pythia attention weights; V diagnostics unavailable.

    Return the base model output to avoid allocating vocabulary logits. The
    public attentions API avoids assuming version-specific attention hooks.
    Only requested layers are retained, immediately offloaded to CPU.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=args.model_dtype, device_map=args.device,
        local_files_only=True, attn_implementation="eager",
    ).eval()
    total_layers = getattr(model.config, "num_hidden_layers", None) or model.config.n_layer
    layers = indices(args.layers, total_layers)
    context_limit = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "seq_length", 2048)
    for sample in samples:
        for length in args.lengths:
            if length > len(sample["token_ids"]):
                print(f"skip short document {sample['document_id']} L={length}", flush=True)
                continue
            if length > context_limit and not args.allow_context_extrapolation:
                raise ValueError(f"{args.model} L={length} exceeds context limit {context_limit}; "
                                 "explicit --allow-context-extrapolation is required")
            pending = [layer for layer in layers if not is_complete(shard_path(args, sample, length, layer), fingerprint, args)]
            if not pending:
                continue
            ids = torch.tensor([sample["token_ids"][:length]], dtype=torch.long,
                               device=model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                output = model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                          output_attentions=True, use_cache=False, return_dict=True)
            if output.attentions is None or any(output.attentions[layer] is None for layer in pending):
                raise RuntimeError("No attention tensors from eager model; do not substitute synthetic weights")
            captures = {layer: output.attentions[layer].detach().cpu() for layer in pending}
            del output
            for layer in pending:
                weights = captures.pop(layer)
                heads = indices(args.heads, weights.shape[1])
                rows = []
                for head in heads:
                    base = {
                        **base_row(args, sample, length, layer, head, fingerprint),
                        "operator_variant": "native", "model_dtype": str(weights.dtype),
                        "operator_build_dtype": str(weights.dtype),
                        "operator_definition": "native eager attention weights returned by model.attentions",
                        "position_method": "ALiBi" if args.model == "bloom" else "RoPE",
                        "native_context_limit": int(context_limit), "context_extrapolation": length > context_limit,
                    }
                    rows.extend(analyze_operator(weights[0, head], None, base, args,
                                                 value_definition="not captured for generic modern baseline"))
                path = shard_path(args, sample, length, layer)
                save_shard(path, rows, fingerprint)
                print(f"saved {path.name} ({len(rows)} rows)", flush=True)


def run_structured(args, samples, fingerprint):
    from compare_attention_ranks_wikitext import capture_module_input
    if args.model == "retnet":
        import fla.models  # registration is intentionally isolated from Mamba process
        from fla.models import RetNetForCausalLM
        model = RetNetForCausalLM.from_pretrained(args.checkpoint, dtype=args.model_dtype,
                                                 device_map=args.device, local_files_only=True).eval()
        blocks = model.model.layers
    else:
        from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
        from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
        config = Mamba2Config.from_pretrained(args.checkpoint, local_files_only=True)
        model = Mamba2ForCausalLM.from_pretrained(args.checkpoint, config=config, dtype=args.model_dtype,
                                                device_map=args.device, local_files_only=True).eval()
        blocks = model.backbone.layers
        for block in blocks:
            block.mixer.chunk_size = args.mamba_chunk_size
    layers = indices(args.layers, len(blocks))
    for sample in samples:
        for length in args.lengths:
            if length > len(sample["token_ids"]):
                print(f"skip short document {sample['document_id']} L={length}", flush=True)
                continue
            for layer in layers:
                path = shard_path(args, sample, length, layer)
                if is_complete(path, fingerprint, args):
                    continue
                module = blocks[layer].attn if args.model == "retnet" else blocks[layer].mixer
                heads = indices(args.heads, module.num_heads)
                ids = torch.tensor([sample["token_ids"][:length]], dtype=torch.long)
                hidden, mask = capture_module_input(model, module, {"input_ids": ids, "attention_mask": torch.ones_like(ids)})
                with torch.inference_mode():
                    rows = (retnet_rows(args, sample, length, layer, heads, hidden, module, fingerprint)
                            if args.model == "retnet" else
                            mamba_rows(args, sample, length, layer, heads, hidden, mask, module, fingerprint))
                save_shard(path, rows, fingerprint)
                print(f"saved {path.name} ({len(rows)} rows)", flush=True)
                del hidden, rows
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def retnet_rows(args, sample, length, layer, heads, hidden, module, fingerprint):
    kv_heads = getattr(module, "num_kv_heads", module.num_heads)
    q = module.q_proj(hidden).view(1, length, module.num_heads, module.head_k_dim)
    k = module.k_proj(hidden).view(1, length, kv_heads, module.head_k_dim)
    v = module.v_proj(hidden).view(1, length, kv_heads, module.head_v_dim)
    q, k = module.rotary(q, k, seqlen_offset=0, max_seqlen=max(length, 2048))
    if kv_heads != module.num_heads:
        k = k.repeat_interleave(module.num_heads // kv_heads, dim=2)
        v = v.repeat_interleave(module.num_heads // kv_heads, dim=2)
    positions = torch.arange(length, device=args.svd_device)
    lag = positions[:, None] - positions[None, :]
    causal = lag >= 0
    rows = []
    for head in heads:
        query, key, value = [x[0, :, head].float().to(args.svd_device) for x in (q, k, v)]
        score = query @ key.T / math.sqrt(module.head_k_dim)
        gamma = 1.0 - 2.0**(-5.0 - head)
        native = score * torch.exp(math.log(gamma) * lag.clamp_min(0)) * causal
        for name, matrix in (("native", native), ("all_one", score * causal)):
            base = {**base_row(args, sample, length, layer, head, fingerprint),
                    "operator_variant": name, "gamma": gamma,
                    "model_dtype": str(hidden.dtype), "operator_build_dtype": "float32",
                    "operator_definition": "rotary QK/sqrt(d_k) times causal " + ("geometric decay" if name == "native" else "all-one mask"),
                    "state_size_per_head_channel": module.head_k_dim}
            rows.extend(analyze_operator(matrix, value, base, args, value_definition="native projected V; before RetNet gate/norm/o_proj"))
    return rows


def mamba_rows(args, sample, length, layer, heads, hidden, mask, module, fingerprint):
    from extract_mamba_matrices import extract_mamba2_matrices
    # Extract one selected head at a time to bound dense L^2 memory on a 3090.
    # V extraction matches validate_reconstruction.py; no gate/norm/out_proj.
    groups_state = module.n_groups * module.ssm_state_size
    projected = module.in_proj(hidden)
    d_mlp = (projected.shape[-1] - 2 * module.intermediate_size - 2 * groups_state - module.num_heads) // 2
    _, _, _, hidden_bc, _ = projected.split([d_mlp, d_mlp, module.intermediate_size, module.conv_dim, module.num_heads], -1)
    convolved = module.act(module.conv1d(hidden_bc.transpose(1, 2))[..., :length].transpose(1, 2))
    x, _, _ = convolved.split([module.intermediate_size, groups_state, groups_state], -1)
    values = x.view(1, length, module.num_heads, module.head_dim)
    rows = []
    for head in heads:
        matrices = extract_mamba2_matrices(module, hidden, attention_mask=mask, selected_heads=[head])
        native = matrices.m[0, 0].to(args.svd_device)
        all_one = torch.tril(matrices.qk[0, 0]).to(args.svd_device)
        dt = matrices.value_dt[0, 0].to(args.svd_device)
        value = values[0, :, head].float().to(args.svd_device)
        direct = float(module.D[head])
        variants = [("native", native, value * dt[:, None], "dt-scaled native V; canonical M"),
                    ("all_one", all_one, value * dt[:, None], "dt-scaled native V; canonical M"),
                    ("native_dt", native * dt[None, :], value, "pre-discretization native V; M diag(dt)"),
                    ("all_one_dt", all_one * dt[None, :], value, "pre-discretization native V; M diag(dt)")]
        if args.include_with_d:
            identity = torch.eye(length, device=native.device, dtype=native.dtype)
            variants.extend((name + "_with_d", matrix + direct * identity, val, definition + "; plus D feedthrough")
                            for name, matrix, val, definition in list(variants) if name.endswith("_dt"))
        for name, matrix, val, definition in variants:
            base = {**base_row(args, sample, length, layer, head, fingerprint),
                    "operator_variant": name, "operator_definition": definition,
                    "D": direct, "model_dtype": str(hidden.dtype), "operator_build_dtype": "float32",
                    "state_size_per_head_channel": module.ssm_state_size,
                    "value_dt_min": float(dt.min()), "value_dt_max": float(dt.max())}
            rows.extend(analyze_operator(matrix, val, base, args, value_definition=definition))
        del matrices, variants
    return rows


def self_test() -> None:
    from types import SimpleNamespace
    args = SimpleNamespace(svd_device="cpu", svd_dtype="float64", svd_driver="gesvd",
                           svd_energy_tolerance=1e-5, abs_tolerances=[1e-1, 1e-3],
                           relative_tolerances=[1e-2], gaps=[0, 4], cut_fractions=[0.5], window=8)
    zero = spectral_metrics(torch.zeros(16, 16), args)
    assert zero["zero_matrix"] and zero["erank_1"] == zero["srank"] == 0
    assert zero["entry_squared_energy"] == zero["svd_squared_energy"] == 0
    assert zero["svd_frobenius_relative_error"] == 0 and zero["svd_energy_qc_pass"]
    identity = torch.eye(32, dtype=torch.float64)
    rows = analyze_operator(identity, torch.ones(32, 2), {}, args, value_definition="test")
    assert abs(rows[0]["srank"] - 32) < 1e-10
    assert rows[0]["entry_squared_energy"] == 32
    assert abs(rows[0]["frobenius_norm"]**2 - 32) < 1e-12
    assert rows[0]["diagonal_energy_fraction"] == 1
    assert rows[0]["strict_past_energy_fraction"] == rows[0]["future_energy_fraction"] == 0
    try:
        entry_energy_qc(identity, torch.full((32,), 2.0), driver="gesvd", tolerance=1e-5)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Material SVD/entry-energy mismatch was not rejected")
    assert all(row["zero_matrix"] for row in rows[1:])
    assert all(row["value_output_frobenius_norm"] == 0 for row in rows[1:])
    positions = torch.arange(32)
    lag = positions[:, None] - positions[None, :]
    gamma = 0.7
    geometric = gamma**lag.clamp_min(0).double() * (lag >= 0)
    metrics = analyze_operator(geometric, None, {}, args, value_definition="unavailable")
    assert metrics[0]["srank"] >= 32 * ((1-gamma)/(1+gamma))**2
    for row in metrics[1:]:
        assert abs(row["erank_1"] - 1) < 1e-7
        assert not row["value_available"]
    null = value_metrics(torch.eye(2), torch.zeros(2, 1), 0, 2, value_definition="test")
    assert null["value_output_to_input_gain"] is None
    print("self-test passed: zero conventions, identity, geometric cross-cut rank, null gain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("gpt2", "retnet", "mamba2", "bloom", "pythia"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--svd-device", default="cuda:0")
    parser.add_argument("--svd-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--svd-driver", choices=("gesvd", "gesvdj", "default"), default="gesvd")
    parser.add_argument("--svd-energy-tolerance", type=float, default=1e-5)
    parser.add_argument("--model-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-stop", type=int, default=None)
    parser.add_argument("--lengths", default="64,128,256,512,1024")
    parser.add_argument("--layers", default="0,-1")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--cut-fractions", default=".25,.5,.75")
    parser.add_argument("--gaps", default="0,32")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--abs-tolerances", default="1e-1,1e-3,1e-5")
    parser.add_argument("--relative-tolerances", default="1e-2,1e-3")
    parser.add_argument("--input-condition", default="native-prefix")
    parser.add_argument("--include-with-d", action="store_true")
    parser.add_argument("--alibi-slopes", default="", help="GPT2 fixed-logit controls only; not trained ALiBi checkpoints")
    parser.add_argument("--mamba-chunk-size", type=int, default=64)
    parser.add_argument("--allow-context-extrapolation", action="store_true",
                        help="Only BLOOM/Pythia; record explicitly when exceeding checkpoint context length")
    parser.add_argument("--overwrite-stale", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for key in ("model", "checkpoint", "manifest", "output"):
        if not getattr(args, key):
            parser.error(f"--{key} is required unless --self-test")
    for key, cast in (("lengths", int), ("gaps", int), ("cut_fractions", float),
                      ("abs_tolerances", float), ("relative_tolerances", float), ("alibi_slopes", float)):
        setattr(args, key, parse_csv(getattr(args, key), cast))
    if any(x <= 0 for x in args.lengths) or args.window <= 0:
        parser.error("lengths/window must be positive")
    if any(x < 0 for x in args.gaps + args.abs_tolerances + args.relative_tolerances + args.alibi_slopes):
        parser.error("gaps, tolerances and ALiBi slopes must be nonnegative")
    if not args.cut_fractions or any(not 0 < x < 1 for x in args.cut_fractions):
        parser.error("cut fractions must lie strictly between 0 and 1")
    if args.doc_start < 0 or (args.doc_stop is not None and args.doc_stop <= args.doc_start):
        parser.error("invalid document slice")
    if args.model != "gpt2" and args.alibi_slopes:
        parser.error("ALiBi fixed-logit controls are only implemented for GPT2")
    torch.manual_seed(20261108)
    # Disable TF32 so exact diagnostic spectra are based on ordinary FP32/64 matmuls.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata, fingerprint = provenance(args)
    atomic_json(Path(args.output) / f"run_metadata.{fingerprint[:16]}.json", metadata)
    if not args.consolidate_only:
        samples = read_documents(args)
        runner = (run_gpt2 if args.model == "gpt2" else
                  run_modern_softmax if args.model in ("bloom", "pythia") else run_structured)
        runner(args, samples, fingerprint)
    count = consolidate(args, fingerprint)
    print(json.dumps({"status": "complete", "rows": count, "run_fingerprint": fingerprint,
                      "metrics": str(Path(args.output) / "metrics.jsonl")}), flush=True)


if __name__ == "__main__":
    main()
