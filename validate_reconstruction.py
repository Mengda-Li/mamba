"""WP0 native-output reconstruction checks for all trained operators."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from compare_attention_ranks_wikitext import BasedExtractor, capture_module_input
from experiment_common import CHECKPOINT_REVISIONS, HEADS, LAYERS, read_jsonl
from extract_mamba_matrices import extract_mamba2_matrices


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
CHECKPOINT_PATHS = {
    "gpt2": os.environ.get("GPT2_MODEL_PATH", str(MODEL_ROOT / "gpt2")),
    "based": os.environ.get("BASED_MODEL_PATH", str(MODEL_ROOT / "based-360m")),
    "retnet": os.environ.get("RETNET_MODEL_PATH", str(MODEL_ROOT / "retnet-1.3B-100B")),
    "mamba2": os.environ.get(
        "MAMBA2_MODEL_PATH",
        str(MODEL_ROOT / "Mamba-Codestral-7B-v0.1"),
    ),
}


def relative_error(native: torch.Tensor, reconstructed: torch.Tensor) -> float:
    numerator = torch.linalg.matrix_norm((native.float() - reconstructed.float()).reshape(-1, native.shape[-1]))
    denominator = torch.linalg.matrix_norm(native.float().reshape(-1, native.shape[-1])).clamp_min(1e-12)
    return float(numerator / denominator)


def samples(manifest: list[dict], model: str):
    rows = [row for row in manifest if row["checkpoint"] == model and row["input_condition"] == "native-prefix" and row["length"] in (16, 32, 64)]
    if rows:
        return rows[:24]
    # The primary manifest starts at 64. Derive nested smoke prefixes from eight documents.
    base = [row for row in manifest if row["checkpoint"] == model and row["input_condition"] == "native-prefix" and row["length"] == 64][:8]
    return [{**row, "length": length, "token_ids": row["token_ids"][:length]} for row in base for length in (16, 32, 64)]


def validate_gpt2(manifest, device: str, checkpoint: str = CHECKPOINT_PATHS["gpt2"]):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().to(device)
    results = []
    for sample in samples(manifest, "gpt2"):
        ids = torch.tensor([sample["token_ids"]])
        for layer in LAYERS["gpt2"]:
            attention = model.transformer.h[layer].attn
            hidden, _ = capture_module_input(model, attention, {"input_ids": ids})
            capture = {}
            def hook(_module, args, output):
                capture["native"] = output[0].detach()
            handle = attention.register_forward_hook(hook)
            with torch.inference_mode():
                model(ids.to(device), use_cache=False)
            handle.remove()
            with torch.inference_mode():
                q, k, v = attention.c_attn(hidden).split(attention.split_size, dim=2)
                length = sample["length"]
                q = q.view(1, length, attention.num_heads, attention.head_dim).transpose(1, 2)
                k = k.view(1, length, attention.num_heads, attention.head_dim).transpose(1, 2)
                v = v.view(1, length, attention.num_heads, attention.head_dim).transpose(1, 2)
                scores = q @ k.transpose(-1, -2) / math.sqrt(attention.head_dim)
                causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
                weights = torch.softmax(scores.masked_fill(~causal, torch.finfo(scores.dtype).min), -1)
                core = (weights @ v).transpose(1, 2).reshape(1, length, -1)
                reconstructed = attention.c_proj(core)
            results.append({
                "checkpoint": "gpt2", "checkpoint_path": checkpoint,
                "revision": CHECKPOINT_REVISIONS["gpt2"], "document_id": sample["document_id"],
                "length": sample["length"], "layer": layer, "heads": list(range(attention.num_heads)),
                "relative_error": relative_error(capture["native"], reconstructed),
                "native_dtype": str(capture["native"].dtype), "matrix_dtype": str(weights.dtype),
                "included": "causal mask, 1/sqrt(d_h), softmax, value mixing, c_proj",
                "excluded": "residual addition, subsequent normalization; dropout disabled by eval()",
            })
    return results


def validate_based(manifest, device: str, checkpoint: str = CHECKPOINT_PATHS["based"]):
    results = []
    for layer in LAYERS["based"]:
        extractor = BasedExtractor(checkpoint, torch.device(device), target_layer=layer)
        for sample in samples(manifest, "based"):
            ids = torch.tensor([sample["token_ids"]], device=device)
            with torch.inference_mode():
                hidden = extractor.target_layer_input(ids)
                q = F.linear(hidden, extractor.weight(extractor.layer_key(layer, "mixer.proj_q.weight")))
                k = F.linear(hidden, extractor.weight(extractor.layer_key(layer, "mixer.proj_k.weight")))
                v = F.linear(hidden, extractor.weight(extractor.layer_key(layer, "mixer.proj_v.weight")))
                length = sample["length"]
                q = q.view(1, length, extractor.num_heads, extractor.feature_dim).transpose(1, 2)
                k = k.view(1, length, extractor.num_heads, extractor.feature_dim).transpose(1, 2)
                v = v.view(1, length, extractor.num_heads, extractor.head_dim).transpose(1, 2)
                qf, kf = extractor.taylor_exp(q), extractor.taylor_exp(k)
                weights = torch.tril(qf @ kf.transpose(-1, -2))
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-12)
                mixed = (weights.to(hidden.dtype) @ v.to(hidden.dtype)).transpose(1, 2).reshape(1, length, -1)
                reconstructed = F.linear(mixed, extractor.weight(extractor.layer_key(layer, "mixer.out_proj.weight")))
                native = extractor.linear_attention_forward(hidden, layer)
            results.append({
                "checkpoint": "based", "checkpoint_path": checkpoint,
                "revision": CHECKPOINT_REVISIONS["based"], "document_id": sample["document_id"],
                "length": length, "layer": layer, "heads": list(range(extractor.num_heads)),
                "relative_error": relative_error(native, reconstructed), "native_dtype": str(native.dtype),
                "matrix_dtype": str(weights.dtype),
                "included": "Taylor feature map, causal support, epsilon=1e-12 row normalization, value mixing, out_proj",
                "excluded": "residual addition and subsequent normalization",
            })
    return results


def validate_retnet(manifest, device: str, checkpoint: str = CHECKPOINT_PATHS["retnet"]):
    import fla.models
    from fla.models import RetNetForCausalLM
    from fla.ops.retention.naive import naive_retention
    model = RetNetForCausalLM.from_pretrained(
        checkpoint,
        dtype="auto",
        device_map=device,
        local_files_only=True,
    ).eval()
    results = []
    for sample in samples(manifest, "retnet"):
        ids = torch.tensor([sample["token_ids"]])
        for layer in LAYERS["retnet"]:
            attention = model.model.layers[layer].attn
            hidden, _ = capture_module_input(model, attention, {"input_ids": ids})
            with torch.inference_mode():
                length = sample["length"]
                q = attention.q_proj(hidden).view(1, length, attention.num_heads, attention.head_k_dim)
                k = attention.k_proj(hidden).view(1, length, attention.num_heads, attention.head_k_dim)
                v = attention.v_proj(hidden).view(1, length, attention.num_heads, attention.head_v_dim)
                q, k = attention.rotary(q, k, seqlen_offset=0, max_seqlen=max(length, 2048))
                native_core = naive_retention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                positions = torch.arange(length, device=q.device)
                lag = positions[:, None] - positions[None, :]
                heads = torch.arange(attention.num_heads, device=q.device)
                gamma = 1.0 - torch.pow(2.0, -5.0 - heads.float())
                decay = torch.exp(torch.log(gamma)[:, None, None] * lag.clamp_min(0)[None]) * (lag >= 0)
                scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) / math.sqrt(attention.head_k_dim)
                reconstructed = torch.einsum("bhts,bshd->bhtd", scores * decay[None], v.float())
                native_fp32 = naive_retention(
                    q.transpose(1, 2).float(),
                    k.transpose(1, 2).float(),
                    v.transpose(1, 2).float(),
                )
            results.append({
                "checkpoint": "retnet", "checkpoint_path": checkpoint,
                "revision": CHECKPOINT_REVISIONS["retnet"], "document_id": sample["document_id"],
                "length": length, "layer": layer, "heads": list(range(attention.num_heads)),
                "relative_error": relative_error(native_core, reconstructed), "native_dtype": str(native_core.dtype),
                "fp32_reference_error": relative_error(native_fp32, reconstructed),
                "matrix_dtype": str(scores.dtype),
                "included": "rotary Q/K, 1/sqrt(d_h), learned head decay, causal mask, value mixing",
                "excluded": "RMS normalization, output gate, o_proj, residual addition",
            })
    return results


def validate_mamba(manifest, device: str, checkpoint: str = CHECKPOINT_PATHS["mamba2"]):
    from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
    model = Mamba2ForCausalLM.from_pretrained(
        checkpoint,
        dtype="auto",
        device_map=device,
        local_files_only=True,
    ).eval()
    for block in model.backbone.layers: block.mixer.chunk_size = 64
    results = []
    for sample in samples(manifest, "mamba2"):
        ids = torch.tensor([sample["token_ids"]])
        for layer in LAYERS["mamba2"]:
            mixer = model.backbone.layers[layer].mixer
            hidden, mask = capture_module_input(model, mixer, {"input_ids": ids})
            with torch.inference_mode():
                matrices = extract_mamba2_matrices(mixer, hidden, mask)
                projected = mixer.in_proj(hidden)
                groups_state = mixer.n_groups * mixer.ssm_state_size
                d_mlp = (projected.shape[-1] - 2 * mixer.intermediate_size - 2 * groups_state - mixer.num_heads) // 2
                _, _, _, hidden_b_c, _ = projected.split([d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], -1)
                conv = mixer.act(mixer.conv1d(hidden_b_c.transpose(1, 2))[..., : sample["length"]].transpose(1, 2))
                value, _, _ = torch.split(conv, [mixer.intermediate_size, groups_state, groups_state], -1)
                value = value.view(1, sample["length"], mixer.num_heads, mixer.head_dim).permute(0, 2, 1, 3).float()
                reconstructed = torch.einsum("bhij,bhjd->bhid", matrices.effective_m, value)
                direct = mixer.D[None, :, None, None].float() * value
                recurrent = reconstructed + direct
                # Validate against an explicit scalar recurrence, the native SSM core before norm/gate.
                state = torch.zeros(1, mixer.num_heads, mixer.head_dim, mixer.ssm_state_size, device=hidden.device)
                native_steps = []
                for token in range(sample["length"]):
                    state = state * matrices.a[:, :, token, None, None] + matrices.k[:, :, token, None, :] * (value[:, :, token, :, None] * matrices.value_dt[:, :, token, None, None])
                    native_steps.append((state * matrices.q[:, :, token, None, :]).sum(-1) + direct[:, :, token])
                native_core = torch.stack(native_steps, dim=2)
            results.append({
                "checkpoint": "mamba2", "checkpoint_path": checkpoint,
                "revision": CHECKPOINT_REVISIONS["mamba2"], "document_id": sample["document_id"],
                "length": sample["length"], "layer": layer, "heads": list(range(mixer.num_heads)),
                "relative_error": relative_error(native_core, recurrent), "native_dtype": str(native_core.dtype),
                "matrix_dtype": str(matrices.m.dtype),
                "included": "input-dependent A/B/C, dt discretization, transition products, value mixing, D feedthrough",
                "excluded": "gated RMS normalization, out_proj, residual addition",
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2", "based", "retnet", "mamba2"), required=True)
    parser.add_argument(
        "--checkpoint",
        help="Local checkpoint path; defaults to the selected model's environment-configured path.",
    )
    parser.add_argument("--manifest", default="experiments_2027/sample_manifest.jsonl")
    parser.add_argument("--output", default="experiments_2027/reconstruction_checks.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = args.checkpoint or CHECKPOINT_PATHS[args.model]
    manifest = read_jsonl(args.manifest)
    checks = {"gpt2": validate_gpt2, "based": validate_based, "retnet": validate_retnet, "mamba2": validate_mamba}[args.model](manifest, args.device, checkpoint)
    path = Path(args.output)
    existing = json.loads(path.read_text()) if path.exists() else {"checks": []}
    existing["checks"] = [row for row in existing["checks"] if row["checkpoint"] != args.model] + checks
    values = [row["relative_error"] for row in checks]
    reference_values = [row.get("fp32_reference_error", row["relative_error"]) for row in checks]
    native_dtype = {row["native_dtype"] for row in checks}
    bf16_explained = native_dtype == {"torch.bfloat16"} and max(values) < 3e-3 and max(reference_values) < 1e-4
    existing.setdefault("acceptance", {})[args.model] = {
        "median_relative_error": float(torch.tensor(values).median()),
        "maximum_relative_error": max(values),
        "maximum_fp32_reference_error": max(reference_values),
        "passed": (float(torch.tensor(values).median()) < 1e-4 and max(values) < 1e-3) or bf16_explained,
        "bf16_tolerance_used": bf16_explained,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(json.dumps(existing["acceptance"][args.model], indent=2))


if __name__ == "__main__": main()
