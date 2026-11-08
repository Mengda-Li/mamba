"""Frozen-manifest GPT-2 head relevance experiments; no network access.

Example (each GPU may use a disjoint document range and the same output):
  python run_gpt2_functional.py --manifest /path/sample_manifest.jsonl \
    --checkpoint /path/gpt2 --output /path/new_results --device cuda:0 \
    --lengths 256,512,1024 --document-start 0 --document-stop 8 \
    --modes ablate,sink-neutralized

Every intervention changes exactly one head's pre-output-projection value
mixture and recomputes the downstream model. The shared projection bias is
retained once. Native and intervened NLL use identical next-token targets.
Global rank-one SVD interventions are deliberately unsupported: a low-rank
approximation to a causal matrix generally leaks future-token information.

Authoritative output: config-<id>/metrics/<sample>.jsonl, committed by a
checksum-bound completed/<sample>.json marker. Failed validation never creates
a completed sample. Frozen token IDs are consumed directly; no tokenizer or
dataset download is performed. All operations are evaluated in FP32.
"""

from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any
from spectral_qc import helper_rank_metrics


REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SCHEMA_VERSION = 2


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any, size: int = 20) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:size]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@contextlib.contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def integers(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(v) for v in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not parsed or min(parsed) < 0:
        raise argparse.ArgumentTypeError("Expected nonnegative integer indices")
    return parsed


def select_samples(manifest: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    source = [s for s in manifest if s["checkpoint"] == "gpt2"]
    documents = sorted({s["document_id"] for s in source})
    selected = set(documents[args.document_start:args.document_stop])
    conditions = set(args.conditions.split(","))
    samples = [s for s in source if s["document_id"] in selected
               and s["input_condition"] in conditions and s["length"] in args.lengths]
    seen = set()
    for sample in samples:
        key = (sample["document_id"], sample["input_condition"], sample["length"])
        if key in seen:
            raise ValueError(f"Duplicate manifest sample {key}")
        seen.add(key)
        if len(sample["token_ids"]) != sample["length"] or sample["length"] < 2:
            raise ValueError(f"Invalid token count in {key}")
    expected = {(d, c, n) for d in selected for c in conditions for n in args.lengths}
    if seen != expected:
        raise ValueError(f"Manifest missing requested samples: {sorted(expected - seen)[:8]}")
    return sorted(samples, key=lambda s: (s["document_id"], s["input_condition"], s["length"]))


def sample_name(sample: dict[str, Any]) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", sample["document_id"])
    identity = {k: sample.get(k) for k in ("checkpoint", "document_id", "input_condition", "length", "token_ids")}
    return f"{safe}__{sample['input_condition']}__L{sample['length']:04d}__{fingerprint(identity, 12)}"


def completed(root: Path, name: str, config_id: str) -> bool:
    try:
        marker = json.loads((root / "completed" / f"{name}.json").read_text())
        if marker["config_id"] != config_id or marker["sample_id"] != name:
            return False
        for kind in ("metrics", "validation"):
            if file_hash(root / marker[f"{kind}_path"]) != marker[f"{kind}_sha256"]:
                return False
        return True
    except (OSError, KeyError, ValueError):
        return False


def nll(logits, input_ids) -> tuple[float, int]:
    import torch.nn.functional as F
    # FP32 log-softmax/cross-entropy, FP64 summation for document NLL.
    losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                             input_ids[:, 1:].reshape(-1), reduction="none")
    if not bool(torch.isfinite(losses).all()):
        raise FloatingPointError("Nonfinite next-token loss")
    return float(losses.double().sum()), losses.numel()


def relative_error(value, reference) -> float:
    return float(torch.linalg.vector_norm((value - reference).double()) /
                 torch.linalg.vector_norm(reference.double()).clamp_min(1e-30))


def capture_native(model, input_ids, layers) -> tuple[dict[int, dict[str, Any]], Any]:
    captures: dict[int, dict[str, Any]] = {layer: {} for layer in layers}
    handles = []

    def qkv_hook(layer, attn):
        def capture(_module, _inputs, output):
            q, k, v = output.split(attn.split_size, dim=-1)
            batch, tokens, _ = q.shape
            for name, tensor in (("query", q), ("key", k), ("value", v)):
                captures[layer][name] = tensor.view(batch, tokens, attn.num_heads, attn.head_dim).transpose(1, 2).detach().clone()
        return capture

    def core_hook(layer):
        def capture(_module, inputs):
            captures[layer]["native_core_flat"] = inputs[0].detach().clone()
        return capture

    def projection_hook(layer):
        def capture(_module, _inputs, output):
            captures[layer]["projected_output"] = output.detach().clone()
        return capture

    def residual_hook(layer):
        def capture(_module, inputs):
            captures[layer]["residual_input"] = inputs[0].detach().clone()
        return capture

    try:
        for layer in layers:
            block = model.transformer.h[layer]
            handles.extend((block.attn.c_attn.register_forward_hook(qkv_hook(layer, block.attn)),
                            block.attn.c_proj.register_forward_pre_hook(core_hook(layer)),
                            block.attn.c_proj.register_forward_hook(projection_hook(layer)),
                            block.register_forward_pre_hook(residual_hook(layer))))
        logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
    finally:
        for handle in handles:
            handle.remove()
    return captures, logits


def replacement_hook(reference, heads: int, head: int | None, replacement, tolerance: float, checks: list[float]):
    """Replace one concatenated head slice, before the shared biased c_proj."""
    def hook(_module, inputs):
        core = inputs[0]
        error = relative_error(core, reference)
        checks.append(error)
        if error > tolerance:
            raise RuntimeError(f"Target pre-projection input changed before intervention: {error}")
        modified = core.clone()
        dim = core.shape[-1] // heads
        if head is None:  # Exact no-op for every slice validates all head boundaries.
            for h in range(heads):
                modified[..., h * dim:(h + 1) * dim] = reference[..., h * dim:(h + 1) * dim]
        else:
            modified[..., head * dim:(head + 1) * dim] = replacement
        return (modified, *inputs[1:])
    return hook


def analyze_sample(model, helper, sample, args, config_id: str):
    ids = torch.tensor(sample["token_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
    length = ids.shape[1]
    if length > model.config.n_positions:
        raise ValueError(f"Length {length} exceeds GPT-2 position limit {model.config.n_positions}")
    captures, native_logits = capture_native(model, ids, args.layers)
    native_sum, token_count = nll(native_logits, ids)
    native_loss = native_sum / token_count
    checks: dict[str, Any] = {"native_nll_sum": native_sum, "loss_token_count": token_count, "layers": {}}
    matrices, scores_by_layer = {}, {}

    # All validation gates run before any functional rows are committed.
    for layer in args.layers:
        attn = model.transformer.h[layer].attn
        cap = captures[layer]
        heads, dim = int(attn.num_heads), int(attn.head_dim)
        working = helper.LayerCapture(query=cap["query"], key=cap["key"], value=cap["value"])
        reconstructed, matrix, scores = helper.reconstruct_native_attention(attn, working, layer)
        flat = reconstructed.transpose(1, 2).reshape(1, length, heads * dim)
        cap["reconstructed_core"] = reconstructed
        matrices[layer], scores_by_layer[layer] = matrix, scores
        core_error = relative_error(flat, cap["native_core_flat"])
        # Conv1D weight is [input_width, output_width]; shared bias added once.
        summed = torch.zeros_like(cap["projected_output"])
        for head in range(heads):
            native_core = cap["native_core_flat"][..., head * dim:(head + 1) * dim]
            summed += native_core @ attn.c_proj.weight[head * dim:(head + 1) * dim]
        if attn.c_proj.bias is not None:
            summed += attn.c_proj.bias
        projected_error = relative_error(summed, cap["projected_output"])
        checks["layers"][str(layer)] = {"core_reconstruction_relative_error": core_error,
            "projected_reconstruction_relative_error": projected_error}
        if max(core_error, projected_error) > args.reconstruction_tolerance:
            raise RuntimeError(f"Layer {layer} reconstruction failed: {checks['layers'][str(layer)]}")

    # One full no-op replay covers every selected layer and every head slice.
    noop_handles, noop_input_checks = [], []
    try:
        for layer in args.layers:
            attn = model.transformer.h[layer].attn
            noop_handles.append(attn.c_proj.register_forward_pre_hook(replacement_hook(
                captures[layer]["native_core_flat"], attn.num_heads, None, None,
                args.reconstruction_tolerance, noop_input_checks)))
        zero_logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
    finally:
        for handle in noop_handles:
            handle.remove()
    zero_sum, zero_count = nll(zero_logits, ids)
    zero_abs = float((zero_logits - native_logits).abs().max())
    zero_loss_error = abs(zero_sum / zero_count - native_loss)
    checks.update(zero_logits_max_abs_error=zero_abs, zero_loss_abs_error=zero_loss_error,
                  zero_target_input_relative_errors=noop_input_checks)
    if zero_count != token_count or zero_abs > args.zero_logits_tolerance or zero_loss_error > args.zero_loss_tolerance:
        raise RuntimeError(f"No-op forward validation failed: {checks}")
    del zero_logits, native_logits

    rows = []
    for layer in args.layers:
        attn, cap = model.transformer.h[layer].attn, captures[layer]
        heads, dim = int(attn.num_heads), int(attn.head_dim)
        selected_heads = tuple(range(heads)) if args.heads == "all" else integers(args.heads)
        if max(selected_heads) >= heads:
            raise ValueError(f"Head indices {selected_heads} exceed {heads} heads")
        residual_norm = float(torch.linalg.vector_norm(cap["residual_input"]))
        for head in selected_heads:
            matrix, value = matrices[layer][0, head], cap["value"][0, head]
            spectrum, rank = helper_rank_metrics(matrix, helper, svd_device=args.svd_device,
                driver=getattr(args, "svd_driver", "gesvd"),
                tolerance=getattr(args, "svd_energy_tolerance", 1e-5))
            native_core = cap["native_core_flat"][..., head * dim:(head + 1) * dim]
            contribution = native_core @ attn.c_proj.weight[head * dim:(head + 1) * dim]
            s = float(matrix[:, 0].square().sum())
            residual = matrix[:, 1:]
            d = float(residual.square().sum())
            contribution_norm = float(torch.linalg.vector_norm(contribution))
            meta = {k: sample[k] for k in ("document_id", "article_index", "sample_offset", "input_condition", "length")}
            meta.update(schema_version=SCHEMA_VERSION, config_id=config_id, checkpoint="gpt2",
                checkpoint_revision=args.revision, layer=layer, head_or_channel=head,
                operator_variant="native_softmax", native_loss=native_loss, native_nll_sum=native_sum,
                loss_token_count=token_count, native_srank=rank["srank"], native_erank_1=rank["erank_1"],
                native_erank_2=rank["erank_2"], native_operator_norm=rank["operator_norm"],
                native_svd_frobenius_norm=rank["svd_frobenius_norm"],
                native_svd_frobenius_relative_error=rank["svd_frobenius_relative_error"],
                native_svd_energy_qc_pass=rank["svd_energy_qc_pass"],
                svd_driver=rank["svd_driver_requested"],
                native_operator_fro=rank["frobenius_norm"], native_best_rank1_error=rank["best_rank1_error"],
                native_singular_values=spectrum.double().cpu().tolist(), sink_S=s, sink_D=d,
                sink_D_over_S=d / s, sink_rho2=d / (s + d),
                av_fro=float(torch.linalg.vector_norm(native_core)), contribution_fro=contribution_norm,
                contribution_token_variance_mean=float(contribution.var(dim=1, unbiased=False).mean()),
                residual_input_fro=residual_norm, relative_contribution_fro=contribution_norm / max(residual_norm, 1e-30),
                projected_output_fro=float(torch.linalg.vector_norm(cap["projected_output"])),
                zero_loss_abs_error=zero_loss_error, zero_logits_max_abs_error=zero_abs,
                **checks["layers"][str(layer)])

            for mode in args.modes.split(","):
                if mode == "ablate":
                    replacement = torch.zeros_like(native_core)
                    changed_rank = None  # Zero operator ranks are defined as zero.
                    intervention_ranks = {"srank": 0.0, "erank_1": 0.0, "erank_2": 0.0}
                elif mode == "sink-neutralized":
                    neutral_scores = scores_by_layer[layer][0, head].clone()
                    neutral_scores[1:, 0] = helper._rowwise_nonsink_median(neutral_scores)[1:]
                    changed = helper.causal_softmax_from_scores(neutral_scores)
                    replacement = (changed @ value).unsqueeze(0)
                    _, changed_rank = helper_rank_metrics(changed, helper, svd_device=args.svd_device,
                        driver=getattr(args, "svd_driver", "gesvd"),
                        tolerance=getattr(args, "svd_energy_tolerance", 1e-5))
                    intervention_ranks = changed_rank
                else:
                    raise ValueError(f"Unsupported intervention {mode}")
                incoming_checks = []
                handle = attn.c_proj.register_forward_pre_hook(replacement_hook(
                    cap["native_core_flat"], heads, head, replacement,
                    args.reconstruction_tolerance, incoming_checks))
                started = time.monotonic()
                try:
                    changed_logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
                    changed_sum, changed_count = nll(changed_logits, ids)
                finally:
                    handle.remove()
                if changed_count != token_count or len(incoming_checks) != 1:
                    raise RuntimeError("Intervention target/loss coverage mismatch")
                row = dict(meta)
                row.update(intervention=mode, intervention_loss=changed_sum / changed_count,
                    intervention_nll_sum=changed_sum, delta_loss=changed_sum / changed_count - native_loss,
                    intervention_core_relative_change=relative_error(replacement, native_core),
                    intervention_contribution_delta_fro=float(torch.linalg.vector_norm(
                        (replacement - native_core) @ attn.c_proj.weight[head * dim:(head + 1) * dim])),
                    intervention_srank=intervention_ranks["srank"],
                    intervention_erank_1=intervention_ranks["erank_1"],
                    intervention_erank_2=intervention_ranks["erank_2"],
                    intervention_svd_frobenius_norm=changed_rank["svd_frobenius_norm"] if changed_rank else 0.0,
                    intervention_svd_frobenius_relative_error=changed_rank["svd_frobenius_relative_error"] if changed_rank else 0.0,
                    intervention_svd_energy_qc_pass=changed_rank["svd_energy_qc_pass"] if changed_rank else True,
                    intervention_target_input_relative_error=incoming_checks[0],
                    intervention_seconds=time.monotonic() - started,
                    intervention_boundary="single_head_before_c_proj_shared_bias_retained",
                    downstream_recomputed=True, causality_preserved=True, status="ok")
                rows.append(row)
                del changed_logits
    expected_rows = sum((model.transformer.h[l].attn.num_heads if args.heads == "all" else len(integers(args.heads)))
                        for l in args.layers) * len(args.modes.split(","))
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} functional rows, got {len(rows)}")
    return rows, checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--svd-device", help="Defaults to the model device")
    parser.add_argument("--svd-driver", choices=("gesvd", "gesvdj", "default"), default="gesvd")
    parser.add_argument("--svd-energy-tolerance", type=float, default=1e-5)
    parser.add_argument("--layers", type=integers, default=(0, 5, 11))
    parser.add_argument("--heads", default="all")
    parser.add_argument("--lengths", type=integers, default=(256, 512, 1024))
    parser.add_argument("--document-start", type=int, default=0)
    parser.add_argument("--document-stop", type=int)
    parser.add_argument("--conditions", default="native-prefix")
    parser.add_argument("--modes", default="ablate", help="Comma-separated: ablate,sink-neutralized")
    parser.add_argument("--reconstruction-tolerance", type=float, default=2e-5)
    parser.add_argument("--zero-logits-tolerance", type=float, default=2e-5)
    parser.add_argument("--zero-loss-tolerance", type=float, default=1e-6)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.document_start < 0 or (args.document_stop is not None and args.document_stop <= args.document_start):
        raise ValueError("Require 0 <= document-start < document-stop")
    if not set(args.modes.split(",")) <= {"ablate", "sink-neutralized"}:
        raise ValueError("Modes must be ablate and/or sink-neutralized; global rank1 is noncausal")
    if len(set(args.modes.split(","))) != len(args.modes.split(",")):
        raise ValueError("Duplicate intervention modes")
    if not set(args.conditions.split(",")) <= {"native-prefix", "explicit-boundary"}:
        raise ValueError("Invalid input condition")
    args.svd_device = args.svd_device or args.device
    samples = select_samples(read_jsonl(args.manifest), args)
    if not samples:
        raise ValueError("No samples selected")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory missing: {args.checkpoint}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    global torch
    import torch
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(args.helpers.resolve()))
    import gpt2_experiments as helper
    torch.manual_seed(20261108)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "numpy"):
        versions[package] = importlib.metadata.version(package)
    checkpoint_manifest = args.checkpoint_manifest or args.manifest.parent / "checkpoint_manifest.json"
    manifest_provenance = None
    if checkpoint_manifest.exists():
        manifest_provenance = json.loads(checkpoint_manifest.read_text())
        declared = manifest_provenance.get("revisions", {}).get("gpt2")
        if declared and declared != args.revision:
            raise ValueError(f"Frozen checkpoint revision mismatch: {declared} != {args.revision}")
    weights = sorted({*args.checkpoint.glob("*.safetensors"), *args.checkpoint.glob("pytorch_model*.bin")})
    if not weights:
        raise FileNotFoundError("No local GPT-2 weights found")
    checkpoint_files = {p.name: file_hash(p) for p in [args.checkpoint / "config.json", *weights]}
    config = dict(schema_version=SCHEMA_VERSION, checkpoint="gpt2", checkpoint_revision=args.revision,
        revision_source="frozen_checkpoint_manifest" if manifest_provenance else "declared_cli_plus_weight_hashes",
        checkpoint_path=str(args.checkpoint.resolve()), checkpoint_files_sha256=checkpoint_files,
        manifest_sha256=file_hash(args.manifest),
        source_sha256={"runner": file_hash(Path(__file__)), "gpt2_experiments": file_hash(Path(helper.__file__)),
                       "spectral_qc": file_hash(Path(__file__).with_name("spectral_qc.py")),
                       "repo_paths": file_hash(Path(__file__).with_name("repo_paths.py"))},
        versions=versions, dtype="float32", layers=list(args.layers), heads=args.heads, modes=args.modes.split(","),
        device_type=torch.device(args.device).type, svd_device_type=torch.device(args.svd_device).type,
        svd_driver=args.svd_driver, svd_energy_tolerance=args.svd_energy_tolerance,
        tf32=False, dropout=False, reconstruction_tolerance=args.reconstruction_tolerance,
        zero_logits_tolerance=args.zero_logits_tolerance, zero_loss_tolerance=args.zero_loss_tolerance,
        protocol="same_manifest_tokens_next_token_nll_single_head_pre_c_proj_full_downstream_recompute")
    config_id = fingerprint(config)
    root = args.output.resolve() / f"config-{config_id}"
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".config.lock"):
        if (root / "config.json").exists() and json.loads((root / "config.json").read_text()) != config:
            raise RuntimeError("Config fingerprint collision")
        atomic_json(root / "config.json", config)
    invocation = {"argv": sys.argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(), "pid": os.getpid(), "samples_selected": len(samples),
        "device": args.device, "config_id": config_id,
        "checkpoint_manifest": manifest_provenance,
        "gpu": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None}
    atomic_json(root / "invocations" / f"{uuid.uuid4().hex}.json", invocation)
    model = AutoModelForCausalLM.from_pretrained(str(args.checkpoint), local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="eager").to(args.device).eval()
    if model.config.model_type != "gpt2":
        raise ValueError("Only GPT-2 checkpoints are supported")
    if max(args.layers) >= len(model.transformer.h):
        raise ValueError("Selected layer exceeds checkpoint depth")
    if any(m.training for m in model.modules()):
        raise RuntimeError("Evaluation mode required")
    print(canonical({"event": "start", "config_id": config_id, "samples": len(samples), "root": str(root)}), flush=True)
    failures = 0
    for sample in samples:
        name = sample_name(sample)
        with lock(root / "locks" / f"{name}.lock"):
            if completed(root, name, config_id):
                print(canonical({"event": "resume_skip", "sample_id": name}), flush=True)
                continue
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    rows, validation = analyze_sample(model, helper, sample, args, config_id)
                for row in rows:
                    row["sample_id"] = name
                metrics = root / "metrics" / f"{name}.jsonl"
                validation_path = root / "validation" / f"{name}.json"
                atomic_text(metrics, "".join(canonical(r) + "\n" for r in rows))
                atomic_json(validation_path, validation)
                marker = dict(config_id=config_id, sample_id=name, row_count=len(rows),
                    document_id=sample["document_id"], input_condition=sample["input_condition"], length=sample["length"],
                    metrics_path=str(metrics.relative_to(root)), metrics_sha256=file_hash(metrics),
                    validation_path=str(validation_path.relative_to(root)), validation_sha256=file_hash(validation_path),
                    seconds=time.monotonic() - started, completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                atomic_json(root / "completed" / f"{name}.json", marker)
                print(canonical({"event": "completed", **marker}), flush=True)
            except Exception as error:
                failures += 1
                atomic_json(root / "errors" / f"{name}__{time.time_ns()}.json", dict(
                    config_id=config_id, sample_id=name, error_type=type(error).__name__, error=str(error),
                    traceback=traceback.format_exc(), seconds=time.monotonic() - started))
                print(canonical({"event": "error", "sample_id": name, "error": str(error)}), flush=True)
                if not args.continue_on_error:
                    raise
    print(canonical({"event": "finished", "config_id": config_id, "failures": failures}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
