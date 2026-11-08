"""Offline, strictly checked official Mamba2-1.3B -> Transformers 4.57.1 adapter.

Only embedding's singular/plural key changes. Stored tensors are not rescaled,
transposed, cast, or otherwise modified. Validation runs FP32 to avoid the
native-fused/HF difference in when FP32 residuals are rounded to FP16/BF16.

Required validation: real-prefix mixer/core/serial-recurrence checks at
lengths16/32/64, layers0/middle/final, plus a complete16-token serial-reference
model. This verifies implemented equations and weight mapping, NOT equivalence
to the unavailable official fused CUDA/Triton kernels. A finite held-out NLL
is reported as a diagnostic, not an equivalence test or benchmark result.

Example:
 python prepare_mamba13_hf.py --source /path/mamba2-1.3b \
   --output /path/mamba2-1.3b-hf --tokenizer /path/pythia-1.4b \
   --manifest /path/sample_manifest.jsonl --device cuda:3

ADAPTER_VALIDATED.json is written last, only after all numeric gates pass.
"""
from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
from typing import Any


PRIMARY_SOURCES = {
    "native_mixer": "https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py",
    "native_gate_norm": "https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py",
    "native_backbone": "https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py",
    "native_block": "https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py",
    "native_tokenizer": "https://github.com/state-spaces/mamba/blob/main/benchmarks/benchmark_generation_mamba_simple.py",
    "hf_runtime": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/mamba2/modeling_mamba2.py",
    "official_config": "https://huggingface.co/state-spaces/mamba2-1.3b/blob/main/config.json",
}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def emit(event: str, **values):
    print(json.dumps(dict(event=event, **values), sort_keys=True, allow_nan=False), flush=True)


def infer_config(native, state, tokenizer, chunk_size):
    from transformers import Mamba2Config
    if native.get("d_intermediate", 0) or native.get("attn_layer_idx"):
        raise ValueError("Only pure Mamba2 without MLP/attention blocks is supported")
    if not native.get("rms_norm", False) or not native.get("residual_in_fp32", False):
        raise ValueError("Adapter requires RMSNorm and FP32 residuals")
    ssm = dict(native.get("ssm_cfg", {}))
    if ssm.pop("layer", None) != "Mamba2":
        raise ValueError("Expected native ssm_cfg.layer=Mamba2")
    allowed = {"d_state", "d_conv", "expand", "headdim", "ngroups", "norm_before_gate", "rmsnorm",
               "D_has_hdim", "bias", "conv_bias", "d_ssm", "dt_limit", "chunk_size", "use_mem_eff_path"}
    if set(ssm) - allowed:
        raise ValueError(f"Unsupported native SSM options: {set(ssm) - allowed}")
    if ssm.get("norm_before_gate", False) or not ssm.get("rmsnorm", True) or ssm.get("D_has_hdim", False):
        raise ValueError("HF gate-first RMSNorm/per-head D does not support this native setting")
    if ssm.get("ngroups", 1) != 1 or ssm.get("d_ssm") is not None:
        raise ValueError("Only full-state Mamba2 with one normalization group is supported")
    embedding = state["backbone.embeddings.weight"]
    vocab, width = embedding.shape
    layers = sorted({int(m.group(1)) for k in state if (m := re.match(r"backbone.layers\.(\d+)\.", k))})
    if layers != list(range(native["n_layer"])) or width != native["d_model"]:
        raise ValueError("Native model dimensions/layer indices do not match weights")
    inner = state["backbone.layers.0.mixer.out_proj.weight"].shape[1]
    heads = state["backbone.layers.0.mixer.D"].numel()
    head_dim = inner // heads
    conv = state["backbone.layers.0.mixer.conv1d.weight"]
    state_size = (conv.shape[0] - inner) // 2
    expand = inner // width
    expected = {"d_state": state_size, "d_conv": conv.shape[-1], "expand": expand, "headdim": head_dim}
    defaults = {"d_state": 128, "d_conv": 4, "expand": 2, "headdim": 64}
    for key, value in expected.items():
        if ssm.get(key, defaults[key]) != value:
            raise ValueError(f"Native/default {key} does not match checkpoint: {value}")
    padded_vocab = ((native["vocab_size"] + native.get("pad_vocab_size_multiple", 1) - 1)
                    // native.get("pad_vocab_size_multiple", 1)) * native.get("pad_vocab_size_multiple", 1)
    if vocab != padded_vocab or inner != heads * head_dim or inner != width * expand:
        raise ValueError("Checkpoint vocabulary/head expansion dimensions are inconsistent")
    vocabulary = tokenizer.get_vocab()
    if (tokenizer.__class__.__name__ != "GPTNeoXTokenizerFast" or not vocabulary
            or min(vocabulary.values()) < 0 or max(vocabulary.values()) >= native["vocab_size"]):
        raise ValueError(f"Expected GPT-NeoX tokenizer with IDs inside native vocabulary; got {tokenizer.__class__.__name__}")
    if ssm.get("bias", False) or not ssm.get("conv_bias", True):
        raise ValueError("Unsupported native linear/conv bias setting")
    # Both tied tensors are retained with identical values. No training is done;
    # avoiding HF automatic retie keeps strict state-dict verification explicit.
    config = Mamba2Config(hidden_size=width, num_hidden_layers=len(layers), vocab_size=vocab,
        state_size=state_size, num_heads=heads, head_dim=head_dim, expand=expand,
        n_groups=1, conv_kernel=conv.shape[-1], layer_norm_epsilon=1e-5,
        hidden_act="silu", use_bias=False, use_conv_bias=True, rms_norm=True,
        residual_in_fp32=True, chunk_size=chunk_size, tie_word_embeddings=False,
        time_step_limit=tuple(ssm.get("dt_limit", (0.0, float("inf")))),
        use_cache=False, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    config.architectures = ["Mamba2ForCausalLM"]
    config.torch_dtype = "float32"
    config.native_checkpoint = "state-spaces/mamba2-1.3b"
    config.native_tie_embeddings = bool(native.get("tie_embeddings", False))
    config.validation_required_dtype = "float32"
    return config


def reference_mixer(mixer, hidden, *, return_core=False):
    """Independent serial scalar-state equations; uses only parameter tensors."""
    import torch
    import torch.nn.functional as F
    batch, length, _ = hidden.shape
    projected = F.linear(hidden, mixer.in_proj.weight, mixer.in_proj.bias)
    inner, groups, n, h, p = mixer.intermediate_size, mixer.n_groups, mixer.ssm_state_size, mixer.num_heads, mixer.head_dim
    if groups != 1:
        raise ValueError("Reference only covers official single-group Mamba2")
    gate, xbc, raw_dt = projected.split((inner, inner + 2 * n, h), dim=-1)
    convolved = F.conv1d(xbc.transpose(1, 2), mixer.conv1d.weight, mixer.conv1d.bias,
                        padding=mixer.conv_kernel_size - 1, groups=inner + 2 * n)[..., :length]
    x, B, C = F.silu(convolved.transpose(1, 2)).split((inner, n, n), dim=-1)
    x = x.reshape(batch, length, h, p).float()
    B, C = B.float(), C.float()
    dt = F.softplus(raw_dt.float() + mixer.dt_bias.float())
    dt = torch.clamp(dt, mixer.time_step_limit[0], mixer.time_step_limit[1])
    a = torch.exp(dt * (-torch.exp(mixer.A_log.float()))[None, None, :])
    state = torch.zeros(batch, h, p, n, dtype=torch.float32, device=hidden.device)
    core = []
    for t in range(length):
        update = dt[:, t, :, None, None] * x[:, t, :, :, None] * B[:, t, None, None, :]
        state = state * a[:, t, :, None, None] + update
        core.append((state * C[:, t, None, None, :]).sum(-1))
    core = torch.stack(core, dim=1)
    y = (core + mixer.D.float()[None, None, :, None] * x).reshape(batch, length, inner)
    gated = y * F.silu(gate.float())
    normalized = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + mixer.layer_norm_epsilon)
    normalized = normalized * mixer.norm.weight.float()
    output = F.linear(normalized.to(hidden.dtype), mixer.out_proj.weight, mixer.out_proj.bias)
    if return_core:
        return output, dict(core=core, x=x, B=B, C=C, dt=dt, a=a, gate=gate, pre_norm=y, normalized=normalized)
    return output


def reference_full_model(model, ids):
    import torch
    import torch.nn.functional as F
    def rms(value, weight, eps):
        work = value.float()
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + eps)) * weight.float()
    hidden = F.embedding(ids, model.backbone.embeddings.weight)
    residual = None
    for block in model.backbone.layers:
        residual = hidden if residual is None else hidden + residual
        normalized = rms(residual, block.norm.weight, block.norm.variance_epsilon)
        hidden = reference_mixer(block.mixer, normalized)
    hidden = rms(hidden + residual, model.backbone.norm_f.weight, model.backbone.norm_f.variance_epsilon)
    return F.linear(hidden, model.lm_head.weight).float()


def validation_ids(manifest: Path, tokenizer, needed=64):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("input_condition") == "native-prefix" and r.get("decoded_text")]
    # Retokenize frozen submitted text. Never reuse Codestral/GPT-2 token IDs.
    for row in sorted(rows, key=lambda r: (str(r["document_id"]), -int(r["length"]))):
        ids = tokenizer(row["decoded_text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= needed:
            return ids[:needed], dict(document_id=row["document_id"],
                source_checkpoint=row.get("checkpoint"), source_length=row["length"],
                source_text_sha256=hashlib.sha256(row["decoded_text"].encode()).hexdigest(),
                token_ids_sha256=hashlib.sha256(json.dumps(ids[:needed]).encode()).hexdigest(),
                selection="first sorted frozen document with sufficient retokenized decoded text")
    raise ValueError("Manifest contains no sufficiently long decoded_text for offline validation")


def validate(model, token_ids, config, args):
    import torch
    import torch.nn.functional as F
    checks = []
    def compare(name, reference, value, tolerance):
        error = float(torch.linalg.vector_norm((value - reference).double()) /
                      torch.linalg.vector_norm(reference.double()).clamp_min(1e-30))
        result = dict(name=name, shape=list(value.shape), relative_error=error,
                      max_abs_error=float((value - reference).abs().max()), tolerance=tolerance)
        checks.append(result)
        if not bool(torch.isfinite(value).all()) or error > tolerance:
            raise RuntimeError(f"Reference validation failed: {result}")
    layers = (0, config.num_hidden_layers // 2 - 1, config.num_hidden_layers - 1)
    nlls = []
    with torch.inference_mode():
        for length in (16, 32, 64):
            captured = {}
            handles = []
            def hook(layer):
                def capture(_module, inputs):
                    captured[layer] = inputs[0].detach().clone()
                return capture
            for layer in layers:
                handles.append(model.backbone.layers[layer].mixer.register_forward_pre_hook(hook(layer)))
            ids = torch.tensor([token_ids[:length]], device=args.device, dtype=torch.long)
            try:
                native = model(input_ids=ids, use_cache=False, return_dict=True).logits
            finally:
                for handle in handles:
                    handle.remove()
            loss = F.cross_entropy(native[:, :-1].reshape(-1, config.vocab_size), ids[:, 1:].reshape(-1))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite validation-prefix NLL")
            nlls.append(dict(length=length, next_token_nll=float(loss), token_count=length - 1,
                             interpretation="plausibility diagnostic only; not a benchmark or equivalence proof"))
            for layer in layers:
                mixer, hidden = model.backbone.layers[layer].mixer, captured[layer]
                reference, detail = reference_mixer(mixer, hidden, return_core=True)
                actual = mixer.torch_forward(hidden, cache_params=None)
                compare(f"layer_{layer}_length_{length}_serial_mixer_vs_HF", reference, actual, args.mixer_tolerance)
                # Canonical M= L*(C B^T), acting on discretized values dt*x.
                for head in (0, config.num_heads // 2, config.num_heads - 1):
                    q, k = detail["C"][0], detail["B"][0]
                    # Compute log-transition directly, never log(exp(log_a)):
                    # native exp may underflow to zero even for finite log_a.
                    log_a = detail["dt"][0, :, head] * (-torch.exp(mixer.A_log.float()[head]))
                    if not bool(torch.isfinite(log_a).all()):
                        raise FloatingPointError("Nonfinite native log-transition")
                    cumulative = log_a.cumsum(0)
                    L = torch.exp(cumulative[:, None] - cumulative[None, :]).tril()
                    M = L * (q @ k.T)
                    mapped = M @ (detail["dt"][0, :, head, None] * detail["x"][0, :, head])
                    compare(f"layer_{layer}_head_{head}_length_{length}_canonical_core_vs_recurrence",
                            detail["core"][0, :, head], mapped, args.core_tolerance)
        ids = torch.tensor([token_ids[:16]], device=args.device, dtype=torch.long)
        hf_logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
        reference_logits = reference_full_model(model, ids)
        compare("full_48_layer_16_token_serial_reference_logits_vs_HF", reference_logits, hf_logits, args.full_tolerance)
        top1_equal = float((reference_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean())
    return dict(status="passed", checks=checks, prefix_nll_diagnostics=nlls,
        full_reference_top1_agreement=top1_equal, runtime_dtype="float32", layers=list(layers),
        lengths=[16, 32, 64], full_reference_length=16,
        equivalence_scope="strict weight/key/shape mapping and independently implemented serial PyTorch equations",
        not_verified="official fused CUDA/Triton kernel numerical parity; mamba_ssm is not installed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--head-chunk-size", type=int, default=8)
    parser.add_argument("--mixer-tolerance", type=float, default=2e-4)
    parser.add_argument("--core-tolerance", type=float, default=2e-4)
    parser.add_argument("--full-tolerance", type=float, default=5e-4)
    parser.add_argument("--source-snapshots", type=Path, help="Optional directory of native source snapshots to hash")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.source.resolve() in args.output.resolve().parents:
        raise ValueError("Output must be a separate sibling/new directory, never inside the source checkpoint")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite output; use a new output directory or inspect previous failure")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoTokenizer, Mamba2ForCausalLM
    from transformers.models.mamba2 import modeling_mamba2
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Adapter audited only for Transformers4.57.1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261108)
    native_config_path, weights = args.source / "config.json", args.source / "pytorch_model.bin"
    native = json.loads(native_config_path.read_text())
    source_weights_hash = file_hash(weights)
    emit("loading_raw_state", source=str(weights), sha256=source_weights_hash)
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a plain tensor state_dict")
    if "backbone.embedding.weight" not in state or "backbone.embeddings.weight" in state:
        raise ValueError("Unexpected native embedding key layout")
    converted = {(k.replace("backbone.embedding.weight", "backbone.embeddings.weight")
                  if k == "backbone.embedding.weight" else k): v for k, v in state.items()}
    if not native.get("tie_embeddings") or not torch.equal(converted["backbone.embeddings.weight"], converted["lm_head.weight"]):
        raise ValueError("Expected equal tied native input/output embedding tensors")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    config = infer_config(native, converted, tokenizer, args.chunk_size)
    with torch.device("meta"):
        model = Mamba2ForCausalLM(config)
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(converted)), sorted(set(converted) - set(expected))
    bad_shapes = {k: {"expected": list(expected[k].shape), "actual": list(converted[k].shape)}
                  for k in set(expected) & set(converted) if expected[k].shape != converted[k].shape}
    if missing or unexpected or bad_shapes:
        raise ValueError(f"Strict state mismatch: missing={missing}, unexpected={unexpected}, shapes={bad_shapes}")
    incompatible = model.load_state_dict(converted, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict loading mismatch: {incompatible}")
    token_ids, sample = validation_ids(args.manifest, tokenizer)
    args.output.mkdir(parents=True)
    provenance = dict(source=str(args.source.resolve()), source_config=native,
        source_config_sha256=file_hash(native_config_path), source_weights_sha256=source_weights_hash,
        adapter_sha256=file_hash(Path(__file__)), transformers_version=transformers.__version__,
        installed_HF_modeling_sha256=file_hash(Path(inspect.getsourcefile(modeling_mamba2))),
        primary_source_urls=PRIMARY_SOURCES,
        optional_native_source_snapshot_hashes={p.name: file_hash(p) for p in sorted(args.source_snapshots.glob("*")) if p.is_file()}
            if args.source_snapshots else {},
        manifest_sha256=file_hash(args.manifest), validation_sample=sample,
        state_dict_key_count=len(converted), strict_missing_keys=[], strict_unexpected_keys=[], strict_shape_mismatches={},
        key_mapping={"backbone.embedding.weight": "backbone.embeddings.weight"},
        tensor_transformations="none; stored tensors/dtypes retained", validated_runtime_dtype="float32",
        native_tie_embeddings=True, HF_tie_word_embeddings=False,
        tie_note="Both input/output tensors explicitly retained and verified equal; inference only, no training",
        supported_native_semantics={"norm_before_gate": False, "gate_norm_group_count": 1,
            "gated_norm_epsilon": 1e-5, "D_has_hdim": False, "residual_in_fp32": True},
        tokenizer_class=tokenizer.__class__.__name__, tokenizer_source=str(args.tokenizer.resolve()),
        tokenizer_base_vocab_size=tokenizer.vocab_size, tokenizer_total_size=len(tokenizer),
        tokenizer_max_id=max(tokenizer.get_vocab().values()),
        tokenizer_files_sha256={}, runtime_chunk_size=args.chunk_size,
        scope_note="FP32 equations/reference parity; no claim of native fused-kernel numerical equivalence")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"):
        path = args.tokenizer / name
        if path.exists():
            shutil.copy2(path, args.output / name)
            provenance["tokenizer_files_sha256"][name] = file_hash(path)
    config.save_pretrained(args.output)
    temporary_weights = args.output / "pytorch_model.bin.tmp"
    torch.save(converted, temporary_weights)
    os.replace(temporary_weights, args.output / "pytorch_model.bin")
    provenance["converted_weights_sha256"] = file_hash(args.output / "pytorch_model.bin")
    provenance["converted_config_sha256"] = file_hash(args.output / "config.json")
    write_json(args.output / "conversion_provenance.json", provenance)
    emit("strict_conversion_passed", keys=len(converted), output=str(args.output))
    try:
        from mamba_memory_patch_exact import install_patch
        patch = install_patch(head_chunk_size=args.head_chunk_size)
        provenance["HF_memory_patch"] = patch
        write_json(args.output / "conversion_provenance.json", provenance)
        model = model.to(device=args.device, dtype=torch.float32).eval()
        report = validate(model, token_ids, config, args)
        report["validation_token_ids"] = token_ids
        write_json(args.output / "validation_report.json", report)
        marker = dict(status="passed", validated_dtype="float32", checkpoint=str(args.output.resolve()),
            source_weights_sha256=source_weights_hash,
            conversion_provenance_sha256=file_hash(args.output / "conversion_provenance.json"),
            validation_report_sha256=file_hash(args.output / "validation_report.json"),
            completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(args.output / "ADAPTER_VALIDATED.json", marker)
        emit("adapter_validated", **marker)
    except Exception as error:
        write_json(args.output / "validation_report.json", dict(status="failed", error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
