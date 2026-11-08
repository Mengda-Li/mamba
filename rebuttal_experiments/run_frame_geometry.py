"""Direct token-frame geometry/eigen-angle diagnostics on frozen documents.

Full square SVD-U eigen-angles are measured through length1024 by default.
Longer contexts retain thin active-frame diagnostics; algorithmic nullspace
completion is always labeled. Canonical signs and one seeded randomized-sign
sensitivity are both reported. Null comparisons are descriptive Monte Carlo
references, not iid-angle p-values or a proof of the Haar joint law.
"""
from __future__ import annotations
from repo_paths import REPO_ROOT
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def seed_for(seed, *keys):
    return (seed + int(digest(keys)[:12], 16)) % (2**63 - 1)


def gaussian(shape, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float64).to(device)


def haar_frame(length, rank, seed, device):
    q, r = torch.linalg.qr(gaussian((length, rank), seed, device), mode="reduced")
    signs = r.diag().sign()
    signs[signs == 0] = 1
    return q * signs


def canonical_signs(u):
    pivots = u.abs().argmax(dim=0)
    signs = u[pivots, torch.arange(u.shape[1], device=u.device)].sign()
    signs[signs == 0] = 1
    return u * signs


def svd(factor, full):
    kwargs = {"driver": "gesvd"} if factor.device.type == "cuda" else {}
    # If L<=d, reduced SVD already returns square U; avoid a huge dxd V.
    return torch.linalg.svd(factor, full_matrices=full and factor.shape[0] > factor.shape[1], **kwargs)


def frame_metrics(u):
    length, rank = u.shape
    leverage = u.square().sum(1)
    normalized = leverage * length / rank
    return dict(coherence=float(normalized.max()), leverage_scaled_second=float(normalized.square().mean()),
        leverage_scaled_variance=float(normalized.var(unbiased=False)),
        entry_scaled_second=float(u.square().mean()) * length,
        entry_scaled_fourth=float(u.pow(4).mean()) * length**2,
        max_scaled_abs_entry=float(u.abs().max()) * math.sqrt(length),
        orthogonality_error=float(torch.linalg.vector_norm(u.T @ u - torch.eye(rank, device=u.device, dtype=u.dtype))))


def angle_metrics(u, args):
    eigen = torch.linalg.eigvals(u.to(args.eig_device))
    angles = torch.remainder(torch.angle(eigen).double().cpu(), 2 * math.pi)
    ordered = (angles / (2 * math.pi)).sort().values
    n = ordered.numel()
    i = torch.arange(1, n + 1, dtype=torch.float64)
    plus, minus = float((i / n - ordered).max()), float((ordered - (i - 1) / n).max())
    cvm = float(((ordered - (2 * i - 1) / (2 * n)).square()).sum()) + 1 / (12 * n)
    result = dict(ks=max(plus, minus), circular_kuiper=plus + minus, cvm=cvm,
        circular_watson_u2=max(0.0, cvm - n * float((ordered.mean() - .5).square())),
        eigenvalue_modulus_max_error=float((eigen.abs() - 1).abs().max()),
        histogram_counts=torch.histc(angles, bins=args.angle_bins, min=0, max=2 * math.pi).to(torch.int64).tolist())
    hist = torch.tensor(result["histogram_counts"], dtype=torch.float64) + 1e-10
    hist /= hist.sum()
    result["histogram_kl_uniform"] = float((hist * (hist * args.angle_bins).log()).sum())
    result["angular_w1_midpoint_grid"] = float((ordered - (i - .5) / n).abs().mean()) * 2 * math.pi
    for k in (1, 2, 4):
        result[f"fourier_moment_abs_{k}"] = float(torch.exp(1j * k * angles).mean().abs())
    return result


def both_angle_signs(u, seed, args):
    canonical = canonical_signs(u)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (u.shape[1],), generator=generator) * 2 - 1).to(u.device)
    return {"solver_native": angle_metrics(u, args),
            "canonical": angle_metrics(canonical, args),
            "random_sign_sensitivity": angle_metrics(canonical * signs, args)}


def summarize_null(actual, draws):
    result = {}
    for metric, value in actual.items():
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            values = torch.tensor([d[metric] for d in draws], dtype=torch.float64)
            result[metric] = dict(null_mean=float(values.mean()), null_q05=float(values.quantile(.05)),
                null_q95=float(values.quantile(.95)), empirical_percentile=float((values <= value).double().mean()))
    return result


def thin_null(length, rank, args, cache):
    key = f"thin_L{length}_r{rank}_n{args.null_replicates}"
    if key not in cache:
        cache[key] = [frame_metrics(canonical_signs(haar_frame(length, rank,
            seed_for(args.seed, key, i), args.analysis_device))) for i in range(args.null_replicates)]
    return cache[key]


def full_null(length, width, args, cache):
    key = f"full_L{length}_d{width}_n{args.eigen_null_replicates}"
    if key not in cache:
        rank = min(length, width)
        result = {}
        for family in ("isotropic_gaussian_factor", "haar_active_distinct_spectrum_factor"):
            draws = []
            for i in range(args.eigen_null_replicates):
                seed = seed_for(args.seed, key, family, i)
                if family == "isotropic_gaussian_factor":
                    factor = gaussian((length, width), seed, args.analysis_device)
                else:
                    left = haar_frame(length, rank, seed, args.analysis_device)
                    right = haar_frame(width, rank, seed + 1, args.analysis_device)
                    singular = torch.linspace(2, 1, rank, device=args.analysis_device, dtype=torch.float64)
                    factor = (left * singular) @ right.T
                u, _, _ = svd(factor, True)
                draws.append(both_angle_signs(u, seed_for(args.seed, key, "signs", i), args))
            result[family] = draws
        cache[key] = result
    return cache[key]


def analyze_factor(factor, base, args, cache, eigen_enabled=True):
    work = factor.detach().to(device=args.analysis_device, dtype=torch.float64)
    length, width = work.shape
    full = eigen_enabled and length <= args.eigen_max_length
    native_u, singular, vh = svd(work, full)
    u = canonical_signs(native_u)
    threshold = float(singular[0]) * max(work.shape) * torch.finfo(work.dtype).eps if singular.numel() else 0
    rank = int((singular > threshold).sum())
    sq = singular.square()
    total = float(sq.sum())
    probs = singular / singular.sum() if total else singular
    probs2 = sq / sq.sum() if total else sq
    spectral = dict(stable_rank=total / float(singular[0])**2 if total else 0,
        erank_1=float((-(probs[probs > 0] * probs[probs > 0].log()).sum()).exp()) if total else 0,
        erank_2=float((-(probs2[probs2 > 0] * probs2[probs2 > 0].log()).sum()).exp()) if total else 0,
        r95=int(torch.searchsorted(sq.cumsum(0) / total, .95)) + 1 if total else 0,
        numerical_rank=rank, numerical_rank_threshold=threshold,
        energy_retained_at_fixed_rank={str(r): float(sq[:r].sum()) / total if total else 0 for r in args.ranks})
    common = {**base, "length": length, "factor_width": width, "factor_spectrum": spectral,
        "captured_factor_dtype": str(factor.dtype),
        "structural_completion_columns": max(0, length - min(length, width)),
        "numerical_nullspace_columns": length - rank,
        "completion_rule": "torch.linalg.svd(full_matrices=True), gesvd CUDA / native LAPACK CPU; no randomized completion",
        "completion_dominates": max(0, length - min(length, width)) > length / 2,
        "analysis_dtype": "float64", "status": "ok"}
    rows = []
    for r in args.ranks:
        if r > rank:
            rows.append({**common, "diagnostic": "thin_frame", "active_rank": r, "status": "insufficient_numerical_rank"})
            continue
        metrics = frame_metrics(u[:, :r])
        rows.append({**common, "diagnostic": "thin_frame", "active_rank": r, "metrics": metrics,
                     "haar_stiefel_null": summarize_null(metrics, thin_null(length, r, args, cache)),
                     "null_replicates": args.null_replicates})
    if full:
        measured = both_angle_signs(native_u, seed_for(args.seed, base, "sign_sensitivity"), args)
        nulls = full_null(length, width, args, cache)
        comparisons = {family: {sign: summarize_null(measured[sign], [d[sign] for d in draws])
                        for sign in measured} for family, draws in nulls.items()}
        rows.append({**common, "diagnostic": "square_u_eigenangles", "angle_metrics": measured,
            "processed_null_comparisons": comparisons, "null_replicates_per_family": args.eigen_null_replicates,
            "angle_histogram_interval": [0, 2 * math.pi], "angle_histogram_bins": args.angle_bins,
            "interpretation": "orthogonal SVD factor including labeled completion; sign-sensitive descriptive statistics, no iid p-values"})
    else:
        rows.append({**common, "diagnostic": "square_u_eigenangles", "status": "not_requested_above_eigen_max_length"})
    return rows, u[:, :min(rank, max(args.ranks))], canonical_signs(vh.T[:, :min(rank, max(args.ranks))])


def pair_metrics(q, k, args, cache, base):
    rows = []
    for rank in args.ranks:
        if min(q.shape[1], k.shape[1]) < rank:
            continue
        overlap = q[:, :rank].T @ k[:, :rank]
        kwargs = {"driver": "gesvd"} if overlap.device.type == "cuda" else {}
        values = torch.linalg.svdvals(overlap, **kwargs).clamp(0, 1)
        metrics = dict(mean_squared_canonical_correlation=float(values.square().mean()),
            min_canonical_correlation=float(values.min()), max_canonical_correlation=float(values.max()),
            canonical_angles_degrees=(values.acos() * 180 / math.pi).cpu().tolist(),
            signed_basis_diagonal_abs_mean=float(overlap.diag().abs().mean()),
            ordered_squared_overlap_mean=float(overlap.diag().square().mean()),
            ordered_squared_overlaps=overlap.diag().square().cpu().tolist(),
            off_diagonal_overlap_energy=float((overlap - torch.diag(overlap.diag())).square().sum()) / rank,
            procrustes_error=math.sqrt(max(0, 2 - 2 * float(values.mean()))))
        key = f"pair_L{q.shape[0]}_r{rank}_n{args.null_replicates}"
        if key not in cache:
            draws = []
            for i in range(args.null_replicates):
                seed = seed_for(args.seed, key, i)
                x = haar_frame(q.shape[0], rank, seed, args.analysis_device)
                y = haar_frame(q.shape[0], rank, seed + 1, args.analysis_device)
                s = torch.linalg.svdvals(x.T @ y, **kwargs).clamp(0, 1)
                draws.append(dict(mean_squared_canonical_correlation=float(s.square().mean()),
                    min_canonical_correlation=float(s.min()), max_canonical_correlation=float(s.max()),
                    signed_basis_diagonal_abs_mean=float((x.T @ y).diag().abs().mean()),
                    ordered_squared_overlap_mean=float((x.T @ y).diag().square().mean()),
                    off_diagonal_overlap_energy=float(((x.T @ y) - torch.diag((x.T @ y).diag())).square().sum()) / rank,
                    procrustes_error=math.sqrt(max(0, 2 - 2 * float(s.mean())))))
            cache[key] = draws
        rows.append({**base, "diagnostic": "qk_subspace_and_basis_alignment", "active_rank": rank,
            "metrics": metrics, "independent_haar_null": summarize_null(metrics, cache[key]),
            "status": "ok", "null_replicates": args.null_replicates})
    return rows


def self_test():
    """Offline diagnostic sanity checks: localized frames and basis/span split."""
    global torch
    import torch
    from types import SimpleNamespace
    args = SimpleNamespace(seed=20261108, analysis_device="cpu", eig_device="cpu", ranks=[4, 8],
        eigen_max_length=64, null_replicates=4, eigen_null_replicates=2, angle_bins=50)
    uniform = haar_frame(128, 8, 17, "cpu")
    localized = torch.eye(128, dtype=torch.float64)[:, :8]
    assert frame_metrics(localized)["coherence"] > frame_metrics(uniform)["coherence"]
    assert frame_metrics(localized)["entry_scaled_fourth"] > frame_metrics(uniform)["entry_scaled_fourth"]
    cache = {}
    rows, left, right = analyze_factor(gaussian((64, 16), 19, "cpu"), {"branch": "test"}, args, cache)
    angle = [r for r in rows if r["diagnostic"] == "square_u_eigenangles"][0]
    assert angle["structural_completion_columns"] == 48 and angle["completion_dominates"]
    assert set(angle["angle_metrics"]) == {"solver_native", "canonical", "random_sign_sensitivity"}
    assert all(sum(v["histogram_counts"]) == 64 for v in angle["angle_metrics"].values())
    rotation = haar_frame(8, 8, 23, "cpu")
    pairs = pair_metrics(uniform, uniform @ rotation, args, cache, {})
    same_span = next(r for r in pairs if r["active_rank"] == 8)["metrics"]
    assert same_span["mean_squared_canonical_correlation"] > 1 - 1e-12
    assert same_span["ordered_squared_overlap_mean"] < .8
    print(json.dumps(dict(status="passed", tests=["Haar vs localized leverage/fourth moments",
        "explicit null-completion label", "all three sign conventions", "same span need not share ordered basis"])))


def select_samples(args):
    longest = {}
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("checkpoint", row.get("model")) != args.manifest_label or row.get("input_condition", "native-prefix") != "native-prefix":
            continue
        key = str(row["document_id"])
        if key not in longest or len(row["token_ids"]) > len(longest[key]["token_ids"]):
            longest[key] = row
    return [longest[k] for k in sorted(longest)[args.doc_start:args.doc_stop]]


def load_model(args):
    if args.model == "gpt2":
        from gpt2_experiments import load_gpt2
        model, _ = load_gpt2(str(args.checkpoint), device_map=args.device, dtype="float32")
        return model, model.transformer.h
    if args.model == "retnet":
        from fla.models import RetNetForCausalLM
        model = RetNetForCausalLM.from_pretrained(str(args.checkpoint), dtype=args.model_dtype,
            device_map=args.device, local_files_only=True).eval()
        return model, model.model.layers
    from transformers import Mamba2ForCausalLM
    if args.model == "mamba13":
        marker = args.checkpoint / "ADAPTER_VALIDATED.json"
        result = json.loads(marker.read_text())
        if result.get("status") != "passed" or file_sha(args.checkpoint / "validation_report.json") != result["validation_report_sha256"]:
            raise RuntimeError("Mamba13 adapter validation is missing/changed")
        args.model_dtype = "float32"
    from mamba_memory_patch_exact import install_patch
    args.patch_provenance = install_patch()
    model = Mamba2ForCausalLM.from_pretrained(str(args.checkpoint), dtype=args.model_dtype,
        device_map=args.device, local_files_only=True).eval()
    for block in model.backbone.layers:
        block.mixer.chunk_size = args.mamba_chunk_size
    return model, model.backbone.layers


def capture(model, blocks, ids, layers, args):
    if args.model == "gpt2":
        from gpt2_experiments import capture_all_layers
        inputs, handles = {}, []
        def hook(layer):
            def save(_module, values):
                inputs[layer] = values[0].detach().cpu()
            return save
        try:
            for layer in layers:
                handles.append(blocks[layer].attn.c_attn.register_forward_pre_hook(hook(layer)))
            bundle = capture_all_layers(model, ids.cpu(), layer_indices=layers)
        finally:
            for handle in handles:
                handle.remove()
        return {layer: dict(Q=bundle.layers[layer].query[0], K=bundle.layers[layer].key[0],
                            X=inputs[layer][0]) for layer in layers}
    captured, handles = {}, []
    def hook(layer):
        def save(_module, inputs, kwargs):
            hidden = inputs[0] if inputs else kwargs["hidden_states"]
            captured[layer] = hidden.detach().cpu()
        return save
    try:
        for layer in layers:
            module = blocks[layer].attn if args.model == "retnet" else blocks[layer].mixer
            handles.append(module.register_forward_pre_hook(hook(layer), with_kwargs=True))
        model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for layer, hidden in captured.items():
        module = blocks[layer].attn if args.model == "retnet" else blocks[layer].mixer
        hidden = hidden.to(device=args.device, dtype=next(module.parameters()).dtype)
        length = hidden.shape[1]
        if args.model == "retnet":
            q = module.q_proj(hidden).view(1, length, module.num_heads, module.head_k_dim)
            kv_heads = getattr(module, "num_kv_heads", module.num_heads)
            k = module.k_proj(hidden).view(1, length, kv_heads, module.head_k_dim)
            pre_q, pre_k = q.detach().clone(), k.detach().clone()
            q, k = module.rotary(q, k, seqlen_offset=0, max_seqlen=max(length, 2048))
            if kv_heads != module.num_heads:
                k = k.repeat_interleave(module.num_heads // kv_heads, dim=2)
                pre_k = pre_k.repeat_interleave(module.num_heads // kv_heads, dim=2)
            q, k = q[0].transpose(0, 1), k[0].transpose(0, 1)
        else:
            projected = module.in_proj(hidden)
            group_state = module.n_groups * module.ssm_state_size
            d_mlp = (projected.shape[-1] - 2 * module.intermediate_size - 2 * group_state - module.num_heads) // 2
            _, _, _, xbc, _ = projected.split([d_mlp, d_mlp, module.intermediate_size, module.conv_dim, module.num_heads], -1)
            xbc = module.act(module.conv1d(xbc.transpose(1, 2))[..., :length].transpose(1, 2))
            _, b, c = xbc.split([module.intermediate_size, group_state, group_state], -1)
            repeat = module.num_heads // module.n_groups
            q = c.view(1, length, module.n_groups, module.ssm_state_size).repeat_interleave(repeat, dim=2)[0].transpose(0, 1)
            k = b.view(1, length, module.n_groups, module.ssm_state_size).repeat_interleave(repeat, dim=2)[0].transpose(0, 1)
        result[layer] = dict(Q=q.detach().cpu(), K=k.detach().cpu(), X=hidden[0].detach().cpu(),
                             n_groups=getattr(module, "n_groups", None))
        if args.model == "retnet":
            result[layer].update(Q_pre_rotary=pre_q[0].transpose(0, 1).cpu(),
                                 K_pre_rotary=pre_k[0].transpose(0, 1).cpu())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("gpt2", "retnet", "mamba2", "mamba13"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-label")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--analysis-device")
    p.add_argument("--eig-device", default="cpu")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--doc-start", type=int, default=0)
    p.add_argument("--doc-stop", type=int, default=8)
    p.add_argument("--lengths", default="64,128,256,512,1024,2048")
    p.add_argument("--layers", default="0,-1")
    p.add_argument("--heads", default="0,mid")
    p.add_argument("--ranks", default="4,8,16")
    p.add_argument("--eigen-max-length", type=int, default=1024)
    p.add_argument("--null-replicates", type=int, default=16)
    p.add_argument("--eigen-null-replicates", type=int, default=4)
    p.add_argument("--angle-bins", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261108)
    p.add_argument("--mamba-chunk-size", type=int, default=64)
    args = p.parse_args()
    args.manifest_label = args.manifest_label or args.model
    if args.model == "mamba13":
        args.model_dtype = "float32"
    args.analysis_device = args.analysis_device or args.device
    args.lengths = sorted({int(v) for v in args.lengths.split(",")})
    args.ranks = sorted({int(v) for v in args.ranks.split(",")})
    if args.null_replicates < 2 or args.eigen_null_replicates < 2 or min(args.ranks + args.lengths) < 1:
        p.error("Require positive dimensions and at least2 null replicates")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    global torch
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("doc_start", "doc_stop", "output", "device", "analysis_device")}
    config.update(manifest_sha256=file_sha(args.manifest), source_sha256=file_sha(Path(__file__)),
        checkpoint_config_sha256=file_sha(args.checkpoint / "config.json"), torch_version=torch.__version__,
        source_helper_hashes={name: file_sha(REPO_ROOT / name) for name in ("gpt2_experiments.py", "extract_mamba_matrices.py")},
        sign_convention="primary solver-native signs; max-absolute-pivot-positive and one seeded random-sign sensitivity also reported",
        null_completion="identical full-SVD algorithm for real and shape-matched null factors",
        null_statistics="descriptive empirical percentiles; dependent eigenangles, no iid p-values")
    config["checkpoint_weight_inventory"] = [{"name": path.name, "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns} for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in sorted(args.checkpoint.glob(pattern))]
    for name in ("rebuttal_download_provenance.json", "ADAPTER_VALIDATED.json", "conversion_provenance.json"):
        path = args.checkpoint / name
        if path.exists():
            config[name + "_sha256"] = file_sha(path)
    run_id = digest(config)[:20]
    root = args.output / ("config-" + run_id)
    atomic_json(root / "protocol.json", config)
    cache_path = root / "null_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    samples = select_samples(args)
    if not samples:
        raise ValueError("No frozen documents selected")
    model, blocks = load_model(args)
    layers = sorted({int(v) % len(blocks) for v in args.layers.split(",")})
    if hasattr(args, "patch_provenance"):
        atomic_json(root / "model_runtime_patch.json", args.patch_provenance)
    with torch.inference_mode():
        for sample in samples:
            for length in args.lengths:
                if length > len(sample["token_ids"]) or (args.model == "gpt2" and length > model.config.n_positions):
                    atomic_json(root / "skipped" / f"{digest(sample['document_id'])[:12]}_L{length}.json",
                        dict(document_id=sample["document_id"], length=length, status="insufficient_tokens_or_native_context_limit"))
                    continue
                pending = []
                for layer in layers:
                    name = f"{digest(sample['document_id'])[:12]}_L{length}_layer{layer}"
                    marker = root / "completed" / (name + ".json")
                    path = root / "shards" / (name + ".jsonl")
                    good = False
                    if marker.exists() and path.exists():
                        stored = json.loads(marker.read_text())
                        good = stored.get("sha256") == file_sha(path) and stored.get("run_id") == run_id
                    if not good:
                        pending.append(layer)
                if not pending:
                    continue
                ids = torch.tensor([sample["token_ids"][:length]], dtype=torch.long, device=args.device)
                captures = capture(model, blocks, ids, pending, args)
                for layer in pending:
                    started = time.monotonic()
                    name = f"{digest(sample['document_id'])[:12]}_L{length}_layer{layer}"
                    try:
                        factors = captures[layer]
                        total_heads = factors["Q"].shape[0]
                        heads = sorted({total_heads // 2 if h == "mid" else int(h) for h in args.heads.split(",")})
                        rows = []
                        shared = dict(run_id=run_id, model=args.model, checkpoint=str(args.checkpoint),
                            document_id=sample["document_id"], input_condition="native-prefix", length=length,
                            token_ids_sha256=digest(sample["token_ids"][:length]), layer=layer)
                        x_rows, x_frame, _ = analyze_factor(factors["X"], {**shared, "branch": "X", "head": None,
                            "factor_definition": "actual normalized mixer input immediately before projections"}, args, cache)
                        rows.extend(x_rows)
                        for head in heads:
                            base = dict(run_id=run_id, model=args.model, checkpoint=str(args.checkpoint),
                                document_id=sample["document_id"], input_condition="native-prefix", length=length,
                                token_ids_sha256=digest(sample["token_ids"][:length]), layer=layer, head=head,
                                factor_definition="C/B branches after causal convolution" if args.model.startswith("mamba") else "native Q/K after rotary if applicable")
                            if factors.get("n_groups"):
                                base["factor_group"] = head // (total_heads // factors["n_groups"])
                                base["shared_with_heads_in_group"] = total_heads // factors["n_groups"]
                            frames, right_frames = {}, {}
                            for branch in ("Q", "K"):
                                measured, frame, right = analyze_factor(factors[branch][head], {**base, "branch": branch}, args, cache)
                                rows.extend(measured)
                                frames[branch] = frame
                                right_frames[branch] = right
                            rows.extend(pair_metrics(frames["Q"], frames["K"], args, cache, {**base, "pair": "Q_K", "space": "token_left"}))
                            rows.extend(pair_metrics(right_frames["Q"], right_frames["K"], args, cache,
                                {**base, "pair": "V_Q_V_K", "space": "feature_right"}))
                            for branch in ("Q", "K"):
                                rows.extend(pair_metrics(x_frame, frames[branch], args, cache,
                                    {**base, "pair": "X_" + branch, "space": "token_left"}))
                            if args.model == "retnet":
                                pre_frames, pre_right = {}, {}
                                for branch in ("Q", "K"):
                                    measured, pre_frames[branch], pre_right[branch] = analyze_factor(
                                        factors[branch + "_pre_rotary"][head], {**base, "branch": branch + "_pre_rotary"},
                                        args, cache, eigen_enabled=False)
                                    rows.extend(measured)
                                rows.extend(pair_metrics(pre_frames["Q"], pre_frames["K"], args, cache,
                                    {**base, "pair": "Q_K_pre_rotary", "space": "token_left"}))
                                rows.extend(pair_metrics(pre_right["Q"], pre_right["K"], args, cache,
                                    {**base, "pair": "V_Q_V_K_pre_rotary", "space": "feature_right"}))
                        atomic_json(cache_path, cache)
                        path = root / "shards" / (name + ".jsonl")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temp = path.with_suffix(f".{os.getpid()}.tmp")
                        temp.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
                        os.replace(temp, path)
                        marker = dict(run_id=run_id, document_id=sample["document_id"], length=length, layer=layer,
                                      rows=len(rows), sha256=file_sha(path), seconds=time.monotonic() - started)
                        atomic_json(root / "completed" / (name + ".json"), marker)
                        print(json.dumps(dict(event="completed", **marker)), flush=True)
                    except Exception as error:
                        atomic_json(root / "errors" / (name + ".json"), dict(error=str(error), traceback=traceback.format_exc()))
                        raise


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
