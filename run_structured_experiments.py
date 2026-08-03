"""Run WP1/WP3/WP4/WP5 for Based, RetNet, and Mamba-2."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import experiment_common as experiment_common_module
from compare_attention_ranks_wikitext import BasedExtractor, capture_module_input
from experiment_common import (
    CHECKPOINT_REVISIONS,
    HEADS,
    LAYERS,
    factor_metrics,
    geometric_memory_metrics,
    memory_metrics,
    rank_metrics_from_singular_values,
    read_jsonl,
    singular_metrics,
)
from experiment_common import explicit_semiseparable_mask
from extract_mamba_matrices import extract_mamba2_matrices


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
PATHS = {
    "based": os.environ.get("BASED_MODEL_PATH", str(MODEL_ROOT / "based-360m")),
    "retnet": os.environ.get("RETNET_MODEL_PATH", str(MODEL_ROOT / "retnet-1.3B-100B")),
    "mamba2": os.environ.get(
        "MAMBA2_MODEL_PATH",
        str(MODEL_ROOT / "Mamba-Codestral-7B-v0.1"),
    ),
}
CORPUS = os.environ.get("EXPERIMENT_CORPUS", "wikitext-103-validation")
_GEOMETRIC_MEMORY_CACHE: dict[tuple[float, int, bool], dict] = {}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def compact_metrics(matrix: torch.Tensor) -> tuple[dict, np.ndarray]:
    result = singular_metrics(matrix, return_spectrum=True)
    spectrum = result.pop("singular_values")
    result["top32_singular_values"] = json.dumps(result["top32_singular_values"])
    result["top32_energy_fractions"] = json.dumps(result["top32_energy_fractions"])
    return result, spectrum


SCHEMA_VERSION = 3
ALIGNMENT_RANKS = (1, 2, 4, 8, 16)
TRUNCATION_RANKS = (1, 2, 4, 8, 16, 32)


def _implementation_sha256() -> str:
    source = Path(__file__).read_text(encoding="utf-8")
    normalized = "\n".join(
        line for line in source.splitlines() if not line.startswith("IMPLEMENTATION_SHA256 = ")
    )
    common = Path(experiment_common_module.__file__).read_text(encoding="utf-8")
    dependencies = (Path(__file__).with_name("compare_attention_ranks_wikitext.py"), Path(__file__).with_name("extract_mamba_matrices.py"))
    dependency_text = "\n".join(path.read_text(encoding="utf-8") for path in dependencies)
    return hashlib.sha256((normalized + "\n" + common + "\n" + dependency_text).encode()).hexdigest()


IMPLEMENTATION_SHA256 = _implementation_sha256()


def _json(value) -> str:
    return json.dumps(value, default=json_value, sort_keys=True, separators=(",", ":"))


def _score_singular_values(sq, vhq, vhk, sk) -> torch.Tensor:
    """Exact nonzero singular values of Q K^T from a feature-sized core."""
    core = torch.diag(sq) @ (vhq @ vhk.T) @ torch.diag(sk)
    return torch.linalg.svdvals(core)


def geometry_metrics(q: torch.Tensor, k: torch.Tensor) -> dict:
    q = q.float()
    k = k.float()
    uq, sq, vhq = torch.linalg.svd(q, full_matrices=False)
    uk, sk, vhk = torch.linalg.svd(k, full_matrices=False)
    dimension = min(q.shape[1], k.shape[1], q.shape[0], k.shape[0])
    score_singular = _score_singular_values(sq, vhq, vhk, sk)
    alignments = {}
    scale_diagnostics = []
    for r in ALIGNMENT_RANKS:
        if r > dimension:
            continue
        token_overlap = uq[:, :r].T @ uk[:, :r]
        projection_overlap = vhq[:r] @ vhk[:r].T
        canonical = torch.linalg.svdvals(token_overlap)
        projection_singular = torch.linalg.svdvals(projection_overlap)
        rotation = torch.linalg.svd(token_overlap, full_matrices=False)
        aligned_k = uk[:, :r] @ (rotation.Vh.T @ rotation.U.T)
        scaled_entries = (math.sqrt(q.shape[1]) * projection_overlap).flatten()
        qq_probabilities = (torch.arange(scaled_entries.numel(), dtype=torch.float64) + 0.5) / scaled_entries.numel()
        alignments[str(r)] = {
            "rank": r,
            "token_canonical_correlations": canonical.detach().cpu().tolist(),
            "token_alignment_score": float(token_overlap.square().sum() / r),
            "independent_subspace_reference": float(r / q.shape[0]),
            "common_frame_reference": 1.0,
            "token_procrustes_distance": float(torch.linalg.matrix_norm(uq[:, :r] - aligned_k) / math.sqrt(r)),
            "projection_scaled_entries": scaled_entries.detach().cpu().tolist(),
            "projection_overlap_singular_values": projection_singular.detach().cpu().tolist(),
            "projection_scaled_mean": float(scaled_entries.mean()),
            "projection_scaled_variance": float(scaled_entries.var(unbiased=False)),
            "projection_scaled_max_abs": float(scaled_entries.abs().max()),
            "projection_normal_qq_probabilities": qq_probabilities.tolist(),
            "projection_normal_qq_theoretical": torch.distributions.Normal(0.0, 1.0).icdf(qq_probabilities).tolist(),
            "projection_normal_qq_empirical": torch.sort(scaled_entries.double().cpu()).values.tolist(),
        }
    leading = min(4, sq.numel(), sk.numel())
    for a in range(leading):
        for b in range(leading):
            scale_diagnostics.append({
                "q_direction": a + 1,
                "k_direction": b + 1,
                "normalized_product": float(sq[a] * sk[b] / (q.shape[0] * math.sqrt(q.shape[1]))),
            })
    return {
        "geometry_by_rank": _json(alignments),
        "q_rank_metrics": _json(factor_metrics(q)),
        "k_rank_metrics": _json(factor_metrics(k)),
        "score_rank_metrics": _json(rank_metrics_from_singular_values(score_singular, dimension=dimension)),
        "scale_diagnostics": _json(scale_diagnostics),
        "scale_sigma1": float(sq[0] * sk[0] / (q.shape[0] * math.sqrt(q.shape[1]))),
    }


def low_rank_reconstruction_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    native: torch.Tensor,
    build_matrix,
) -> list[dict]:
    q = q.float()
    k = k.float()
    uq, sq, vhq = torch.linalg.svd(q, full_matrices=False)
    uk, sk, vhk = torch.linalg.svd(k, full_matrices=False)
    native_metrics = singular_metrics(native)
    rows = []
    maximum = min(q.shape[0], q.shape[1], k.shape[0], k.shape[1])
    q_total = sq.square().sum().clamp_min(1e-30)
    k_total = sk.square().sum().clamp_min(1e-30)
    score_singular = _score_singular_values(sq, vhq, vhk, sk)
    score_total = score_singular.square().sum().clamp_min(1e-30)
    ranks = sorted({rank for rank in (*TRUNCATION_RANKS, maximum) if rank <= maximum})
    for rank in ranks:
        qr = (uq[:, :rank] * sq[:rank]) @ vhq[:rank]
        kr = (uk[:, :rank] * sk[:rank]) @ vhk[:rank]
        approximation = build_matrix(qr, kr)
        metrics = singular_metrics(approximation)
        truncated_score_singular = _score_singular_values(sq[:rank], vhq[:rank], vhk[:rank], sk[:rank])
        error = torch.linalg.matrix_norm(native.float() - approximation.float()) / torch.linalg.matrix_norm(native.float()).clamp_min(1e-12)
        row = {
            "truncation_rank": rank,
            "is_full_rank": rank == maximum,
            "q_retained_energy": float(sq[:rank].square().sum() / q_total),
            "k_retained_energy": float(sk[:rank].square().sum() / k_total),
            "factor_retained_energy_min": float(min(sq[:rank].square().sum() / q_total, sk[:rank].square().sum() / k_total)),
            "score_relative_squared_energy": float(truncated_score_singular.square().sum() / score_total),
            "relative_mixer_frobenius_error": float(error),
            "leading_singular_value_relative_error": abs(metrics["operator_norm"] - native_metrics["operator_norm"]) / max(abs(native_metrics["operator_norm"]), 1e-12),
        }
        for metric in ("operator_norm", "srank", "erank_1", "erank_2"):
            row[f"relative_{metric}_error"] = abs(metrics[metric] - native_metrics[metric]) / max(abs(native_metrics[metric]), 1e-12)
        rows.append(row)
    return rows


def stable_observation_id(base: dict, variant: str) -> str:
    identity = {key: base[key] for key in ("checkpoint", "checkpoint_revision", "corpus", "document_id", "sample_offset", "input_condition", "length", "sample_token_sha256", "layer", "head_or_channel")}
    identity.update({"operator_variant": variant, "schema_version": SCHEMA_VERSION})
    return hashlib.sha256(_json(identity).encode()).hexdigest()[:32]


def operator_as_mask_metrics(matrix: torch.Tensor, family: str) -> dict:
    """Memory-like summaries when normalization itself is the operator."""
    nonnegative = matrix.detach().float()
    if not torch.isfinite(nonnegative).all() or bool((nonnegative < -1e-7).any()):
        raise ValueError(f"{family} must be finite and nonnegative to serve as an effective mask.")
    nonnegative = nonnegative.clamp_min(0.0)
    return {
        "mask_family": family,
        "mask_definition": "normalized_operator",
        "mask_R": float(nonnegative.sum(-1).max()),
        "mask_C": float(nonnegative.sum(-2).max()),
        "mask_kappa_from_operator": True,
        "block_products": {},
        "effective_half_life_summary": {
            "not_applicable": True,
            "reason": "normalization is not a scalar transition sequence",
        },
        "effective_half_life_min": None,
        "effective_half_life_median": None,
        "effective_half_life_q95": None,
        "effective_half_life_max": None,
        "effective_half_life_censored_fraction": None,
    }


def transition_memory_metrics(
    transitions: torch.Tensor,
    family: str,
    *,
    validate_exact: bool,
) -> dict:
    result = memory_metrics(transitions, validate_exact=validate_exact)
    result.update({"mask_family": family, "mask_definition": "scalar_semiseparable_transitions"})
    return result


def geometric_variant_memory(gamma: float, length: int, *, validate_exact: bool) -> dict:
    key = (float(gamma), int(length), bool(validate_exact))
    if key not in _GEOMETRIC_MEMORY_CACHE:
        _GEOMETRIC_MEMORY_CACHE[key] = geometric_memory_metrics(gamma, length, validate_exact=validate_exact)
    result = copy.deepcopy(_GEOMETRIC_MEMORY_CACHE[key])
    result["mask_definition"] = "geometric_scalar_transitions"
    return result


def sample_token_sha256(sample: dict) -> str:
    return hashlib.sha256(_json([int(token) for token in sample["token_ids"]]).encode()).hexdigest()


def should_validate_memory(base: dict, *, enabled: bool = True) -> bool:
    """Preregistered exact validation: all <=512 and fixed 10% at 1024."""
    if not enabled:
        return False
    if base["length"] <= 512:
        return True
    if base["length"] != 1024:
        return False
    payload = _json({key: base[key] for key in ("checkpoint", "document_id", "layer", "head_or_channel")})
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % 10 == 0


def variant_rows(
    base: dict,
    variants: dict[str, torch.Tensor],
    variant_memory: dict[str, dict],
    spectra: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for variant, matrix in variants.items():
        metrics, singular = compact_metrics(matrix)
        observation_id = stable_observation_id(base, variant)
        spectrum_id = f"spectrum-{observation_id}"
        spectra[spectrum_id] = singular
        memory = variant_memory.get(variant)
        if memory is None:
            raise KeyError(f"Missing mask diagnostics for operator variant {variant!r}.")
        memory = dict(memory)
        if memory.pop("mask_kappa_from_operator", False):
            memory["mask_kappa"] = metrics["operator_norm"]
        for key in ("block_products", "effective_half_life_summary"):
            if key in memory:
                memory[key] = _json(memory[key])
        row = {
            **base,
            "schema_version": SCHEMA_VERSION,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "observation_id": observation_id,
            "operator_variant": variant,
            "spectrum_id": spectrum_id,
            **memory,
            **metrics,
        }
        log_length = max(math.log(base["length"]), 1e-12)
        gain = row["mask_kappa"]
        row["theorem_normalized_srank"] = row["srank"] * gain**2 * log_length**2 / base["length"]
        row["theorem_normalized_erank_2"] = row["erank_2"] * gain**2 * log_length**2 / base["length"]
        row["theorem_normalized_erank_1"] = row["erank_1"] * gain * log_length / base["length"]
        rows.append(row)
    return rows


def write_outputs(rows: list[dict], spectra: dict[str, np.ndarray], output: Path, model: str) -> None:
    if not rows:
        return
    identifiers = {row["observation_id"] for row in rows}
    if len(identifiers) != len(rows):
        raise RuntimeError("Duplicate observation ids within a work unit.")
    expected_spectra = {row["spectrum_id"] for row in rows}
    if expected_spectra != set(spectra):
        raise RuntimeError("Metric spectrum references do not match the work-unit archive.")
    variants = {row["operator_variant"] for row in rows}
    if model == "based":
        expected_variants = {"native_raw", "native_normalized", "identity"}
    elif model == "retnet":
        expected_variants = {"native", "all_one", "identity", "decay_0.5", "decay_0.9", "decay_0.99", "causal_softmax"}
    else:
        expected_variants = {"native", "all_one", "identity", "constant_decay", "transition_shuffled"}
    if variants != expected_variants:
        raise RuntimeError(f"Incomplete {model} operator variants: {sorted(variants)}")
    identity_fields = {(row["document_id"], row["length"], row["layer"]) for row in rows}
    if len(identity_fields) != 1:
        raise RuntimeError("A structured work unit must contain exactly one document/length/layer.")
    expected_rows = len(HEADS[model]) * len(expected_variants)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in a {model} work unit, got {len(rows)}.")
    unit = hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest()[:24]
    metrics_dir = output / "structured_shards" / model / "metrics"
    spectra_dir = output / "structured_shards" / model / "spectra"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{unit}.parquet"
    spectrum_path = spectra_dir / f"{unit}.npz"
    if metrics_path.exists() != spectrum_path.exists():
        metrics_path.unlink(missing_ok=True)
        spectrum_path.unlink(missing_ok=True)
    if metrics_path.exists() and spectrum_path.exists():
        existing = pd.read_parquet(
            metrics_path, columns=["observation_id", "spectrum_id", "implementation_sha256"]
        )
        with np.load(spectrum_path) as archive:
            spectrum_ids = set(archive.files)
        if (
            len(existing) == len(identifiers)
            and not existing["observation_id"].duplicated().any()
            and set(existing["observation_id"]) == identifiers
            and set(existing["spectrum_id"]) == spectrum_ids == set(spectra)
            and set(existing["implementation_sha256"]) == {IMPLEMENTATION_SHA256}
        ):
            return
        raise RuntimeError(f"Existing work unit {unit} does not match deterministic contents.")
    tmp_metrics = metrics_path.with_suffix(f".tmp-{os.getpid()}.parquet")
    tmp_spectra = spectrum_path.with_suffix(f".tmp-{os.getpid()}.npz")
    pd.DataFrame(rows).sort_values("observation_id").to_parquet(tmp_metrics, index=False)
    with tmp_spectra.open("wb") as handle:
        np.savez_compressed(handle, **spectra)
    os.replace(tmp_spectra, spectrum_path)
    os.replace(tmp_metrics, metrics_path)


def flush(rows: list[dict], spectra: dict[str, np.ndarray], output: Path, model: str) -> None:
    write_outputs(rows, spectra, output, model)
    rows.clear()
    spectra.clear()


def work_unit_complete(
    output: Path,
    model: str,
    sample: dict,
    layer: int,
    variants: tuple[str, ...],
    heads: tuple[int, ...],
) -> bool:
    seed = {
        "checkpoint": model,
        "checkpoint_revision": CHECKPOINT_REVISIONS[model],
        "corpus": CORPUS,
        "document_id": sample["document_id"],
        "sample_offset": sample["sample_offset"],
        "input_condition": sample["input_condition"],
        "length": sample["length"],
        "sample_token_sha256": sample_token_sha256(sample),
        "layer": layer,
    }
    identifiers = {
        stable_observation_id({**seed, "head_or_channel": head}, variant)
        for head in heads
        for variant in variants
    }
    unit = hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest()[:24]
    metrics_path = output / "structured_shards" / model / "metrics" / f"{unit}.parquet"
    spectra_path = output / "structured_shards" / model / "spectra" / f"{unit}.npz"
    if metrics_path.exists() != spectra_path.exists():
        # A process can stop between the two atomic renames.  Recompute the
        # deterministic unit; write_outputs will replace the orphan safely.
        return False
    if not metrics_path.exists():
        return False
    existing = pd.read_parquet(
        metrics_path, columns=["observation_id", "spectrum_id", "implementation_sha256"]
    )
    with np.load(spectra_path) as archive:
        spectra = set(archive.files)
    if (
        len(existing) != len(identifiers)
        or existing["observation_id"].duplicated().any()
        or set(existing["observation_id"]) != identifiers
        or set(existing["spectrum_id"]) != spectra
        or set(existing["implementation_sha256"]) != {IMPLEMENTATION_SHA256}
    ):
        raise RuntimeError(f"Resume validation failed for work unit {unit}.")
    return True


def consolidate_metrics(output: Path, model: str, *, allow_partial: bool = True) -> None:
    shards = sorted((output / "structured_shards" / model / "metrics").glob("*.parquet"))
    if not shards:
        return
    combined = pd.concat((pd.read_parquet(path) for path in shards), ignore_index=True)
    fingerprints = set(combined["implementation_sha256"])
    if fingerprints != {IMPLEMENTATION_SHA256}:
        raise RuntimeError(
            f"Cannot consolidate {model} shards from implementation fingerprints {sorted(fingerprints)}; "
            f"current fingerprint is {IMPLEMENTATION_SHA256}."
        )
    if combined["observation_id"].duplicated().any():
        duplicate = combined.loc[combined["observation_id"].duplicated(), "observation_id"].iloc[0]
        raise RuntimeError(f"Duplicate structured observation across shards: {duplicate}")
    if not allow_partial:
        expected = {
            ("based", "native-prefix"): len(LAYERS["based"]) * len(HEADS["based"]) * 3,
            ("retnet", "native-prefix"): len(LAYERS["retnet"]) * len(HEADS["retnet"]) * 7,
            ("mamba2", "native-prefix"): len(LAYERS["mamba2"]) * len(HEADS["mamba2"]) * 5,
        }[(model, "native-prefix")]
        counts = combined.groupby(["document_id", "length"], dropna=False).size()
        bad = counts[counts != expected]
        if len(bad):
            raise RuntimeError(
                f"Cannot consolidate incomplete {model} output; expected {expected} rows per document/length, "
                f"first mismatch is {bad.index[0]} with {int(bad.iloc[0])}."
            )
    destination = output / f"{model}_metrics.parquet"
    temporary = destination.with_suffix(f".tmp-{os.getpid()}.parquet")
    combined.sort_values("observation_id").to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def records_for(manifest: list[dict], model: str, max_documents: int | None) -> list[dict]:
    rows = [row for row in manifest if row["checkpoint"] == model and row["input_condition"] == "native-prefix"]
    documents = sorted({row["document_id"] for row in rows})
    if max_documents:
        documents = documents[:max_documents]
    return [row for row in rows if row["document_id"] in documents]


def run_based(args, manifest, output: Path) -> None:
    all_rows, spectra = [], {}
    variants_expected = ("native_raw", "native_normalized", "identity")
    for layer in LAYERS["based"]:
        extractor = BasedExtractor(args.checkpoint, torch.device(args.device), target_layer=layer)
        for sample in records_for(manifest, "based", args.max_documents):
            if work_unit_complete(output, "based", sample, layer, variants_expected, HEADS["based"]):
                print(f"Based resume-skip layer={layer} doc={sample['document_id']} L={sample['length']}", flush=True)
                continue
            ids = torch.tensor([sample["token_ids"]], device=args.device)
            with torch.inference_mode():
                hidden = extractor.target_layer_input(ids)
                query = F.linear(hidden, extractor.weight(extractor.layer_key(layer, "mixer.proj_q.weight")))
                key = F.linear(hidden, extractor.weight(extractor.layer_key(layer, "mixer.proj_k.weight")))
            length = sample["length"]
            query = query.view(1, length, extractor.num_heads, extractor.feature_dim).transpose(1, 2)
            key = key.view(1, length, extractor.num_heads, extractor.feature_dim).transpose(1, 2)
            qf, kf = extractor.taylor_exp(query.float())[0], extractor.taylor_exp(key.float())[0]
            causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=args.device))
            for head in HEADS["based"]:
                raw = (qf[head] @ kf[head].T).masked_fill(~causal, 0.0)
                native = raw / (raw.sum(-1, keepdim=True) + 1e-12)
                identity = torch.diag(torch.diag(qf[head] @ kf[head].T))
                base = {
                    "checkpoint": "based",
                    "checkpoint_path": args.checkpoint,
                    "checkpoint_revision": CHECKPOINT_REVISIONS["based"],
                    "corpus": CORPUS,
                    "document_id": sample["document_id"],
                    "sample_offset": sample["sample_offset"],
                    "input_condition": sample["input_condition"],
                    "length": length,
                    "sample_token_sha256": sample_token_sha256(sample),
                    "layer": layer,
                    "head_or_channel": head,
                    **geometry_metrics(qf[head], kf[head]),
                }
                base["low_rank_reconstruction"] = json.dumps(
                    low_rank_reconstruction_rows(
                        qf[head],
                        kf[head],
                        native,
                        lambda qr, kr: (
                            (qr @ kr.T).masked_fill(~causal, 0.0)
                            / ((qr @ kr.T).masked_fill(~causal, 0.0).sum(-1, keepdim=True) + 1e-12)
                        ),
                    )
                )
                validate = should_validate_memory(base, enabled=args.validate_exact_memory)
                all_one_memory = geometric_variant_memory(1.0, length, validate_exact=validate)
                identity_memory = geometric_variant_memory(0.0, length, validate_exact=validate)
                native_normalized_memory = operator_as_mask_metrics(native, "causal_row_normalization")
                all_rows.extend(
                    variant_rows(
                        base,
                        {"native_raw": raw, "native_normalized": native, "identity": identity},
                        {
                            "native_raw": all_one_memory,
                            "native_normalized": native_normalized_memory,
                            "identity": identity_memory,
                        },
                        spectra,
                    )
                )
            print(f"Based layer={layer} doc={sample['document_id']} L={length}", flush=True)
            flush(all_rows, spectra, output, "based")
        del extractor
        gc.collect(); torch.cuda.empty_cache()
    consolidate_metrics(output, "based", allow_partial=True)


def load_retnet(device: str, checkpoint: str = PATHS["retnet"]):
    import fla.models
    from fla.models import RetNetForCausalLM
    return RetNetForCausalLM.from_pretrained(
        checkpoint,
        dtype="auto",
        device_map=device,
        local_files_only=True,
    ).eval()


def run_retnet(args, manifest, output: Path) -> None:
    model = load_retnet(args.device, args.checkpoint)
    all_rows, spectra = [], {}
    variants_expected = ("native", "all_one", "identity", "decay_0.5", "decay_0.9", "decay_0.99", "causal_softmax")
    for sample in records_for(manifest, "retnet", args.max_documents):
        ids = torch.tensor([sample["token_ids"]], device=model.get_input_embeddings().weight.device)
        encoded = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        length = sample["length"]
        for layer in LAYERS["retnet"]:
            if work_unit_complete(output, "retnet", sample, layer, variants_expected, HEADS["retnet"]):
                print(f"RetNet resume-skip layer={layer} doc={sample['document_id']} L={length}", flush=True)
                continue
            attention = model.model.layers[layer].attn
            hidden, _ = capture_module_input(model, attention, encoded)
            with torch.inference_mode():
                q = attention.q_proj(hidden).view(1, length, attention.num_heads, attention.head_k_dim)
                k = attention.k_proj(hidden).view(1, length, attention.num_heads, attention.head_k_dim)
                q, k = attention.rotary(q, k, seqlen_offset=0, max_seqlen=max(length, 2048))
            positions = torch.arange(length, device=q.device)
            lag = positions[:, None] - positions[None, :]
            causal = lag >= 0
            for head in HEADS["retnet"]:
                score = q[0, :, head].float() @ k[0, :, head].float().T / math.sqrt(attention.head_k_dim)
                gamma = 1.0 - 2.0 ** (-5.0 - head)
                native_mask = torch.exp(math.log(gamma) * lag.clamp_min(0)) * causal
                variants = {"native": score * native_mask, "all_one": score * causal, "identity": torch.diag(torch.diag(score))}
                for fixed in (0.5, 0.9, 0.99):
                    variants[f"decay_{fixed}"] = score * torch.exp(math.log(fixed) * lag.clamp_min(0)) * causal
                variants["causal_softmax"] = torch.softmax(score.masked_fill(~causal, torch.finfo(score.dtype).min), dim=-1)
                base = {
                    "checkpoint": "retnet", "checkpoint_path": args.checkpoint,
                    "checkpoint_revision": CHECKPOINT_REVISIONS["retnet"],
                    "corpus": CORPUS, "document_id": sample["document_id"],
                    "sample_offset": 0, "input_condition": "native-prefix", "length": length,
                    "sample_token_sha256": sample_token_sha256(sample),
                    "layer": layer, "head_or_channel": head,
                    **geometry_metrics(q[0, :, head], k[0, :, head]),
                }
                base["low_rank_reconstruction"] = json.dumps(
                    low_rank_reconstruction_rows(
                        q[0, :, head],
                        k[0, :, head],
                        variants["native"],
                        lambda qr, kr: (qr @ kr.T / math.sqrt(attention.head_k_dim)) * native_mask,
                    )
                )
                validate = should_validate_memory(base, enabled=args.validate_exact_memory)
                variant_memory = {
                    "native": geometric_variant_memory(gamma, length, validate_exact=validate),
                    "all_one": geometric_variant_memory(1.0, length, validate_exact=validate),
                    "identity": geometric_variant_memory(0.0, length, validate_exact=validate),
                    "decay_0.5": geometric_variant_memory(0.5, length, validate_exact=validate),
                    "decay_0.9": geometric_variant_memory(0.9, length, validate_exact=validate),
                    "decay_0.99": geometric_variant_memory(0.99, length, validate_exact=validate),
                    "causal_softmax": operator_as_mask_metrics(variants["causal_softmax"], "causal_softmax_normalization"),
                }
                all_rows.extend(variant_rows(base, variants, variant_memory, spectra))
            print(f"RetNet layer={layer} doc={sample['document_id']} L={length}", flush=True)
            flush(all_rows, spectra, output, "retnet")
    consolidate_metrics(output, "retnet", allow_partial=True)


def load_mamba(device: str, checkpoint: str = PATHS["mamba2"]):
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


def run_mamba(args, manifest, output: Path) -> None:
    model = load_mamba(args.device, args.checkpoint)
    all_rows, spectra = [], {}
    selected = list(HEADS["mamba2"])
    variants_expected = ("native", "all_one", "identity", "constant_decay", "transition_shuffled")
    for sample in records_for(manifest, "mamba2", args.max_documents):
        ids = torch.tensor([sample["token_ids"]], device=model.get_input_embeddings().weight.device)
        encoded = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        length = sample["length"]
        for layer in LAYERS["mamba2"]:
            if work_unit_complete(output, "mamba2", sample, layer, variants_expected, HEADS["mamba2"]):
                print(f"Mamba resume-skip layer={layer} doc={sample['document_id']} L={length}", flush=True)
                continue
            mixer = model.backbone.layers[layer].mixer
            hidden, mask = capture_module_input(model, mixer, encoded)
            with torch.inference_mode():
                matrices = extract_mamba2_matrices(mixer, hidden, attention_mask=mask, selected_heads=selected)
                lag = torch.arange(length, device=hidden.device)[:, None] - torch.arange(length, device=hidden.device)[None, :]
                causal = lag >= 0
                for local_head, head in enumerate(selected):
                    q = matrices.q[0, local_head]
                    k = matrices.k[0, local_head]
                    score = q @ k.T
                    a = matrices.a[0, local_head]
                    native = matrices.m[0, local_head]
                    gamma = float(torch.exp(torch.log(a.clamp_min(1e-30)).mean()))
                    generator = torch.Generator(device=a.device).manual_seed(20270802 + layer * 1000 + head * 10 + sample["article_index"])
                    shuffled_a = a.clone()
                    if length > 1:
                        permutation = torch.randperm(length - 1, generator=generator, device=a.device) + 1
                        shuffled_a[1:] = a[permutation]
                    variants = {
                        "native": native,
                        "all_one": score * causal,
                        "identity": torch.diag(torch.diag(score)),
                        "constant_decay": score * torch.exp((math.log(gamma) * lag.clamp_min(0).double())).to(score.dtype) * causal,
                        "transition_shuffled": score * explicit_semiseparable_mask(shuffled_a),
                    }
                    base = {
                        "checkpoint": "mamba2", "checkpoint_path": args.checkpoint,
                        "checkpoint_revision": CHECKPOINT_REVISIONS["mamba2"],
                        "corpus": CORPUS, "document_id": sample["document_id"],
                        "sample_offset": 0, "input_condition": "native-prefix", "length": length,
                        "sample_token_sha256": sample_token_sha256(sample),
                        "layer": layer, "head_or_channel": head,
                        **geometry_metrics(q, k),
                    }
                    base["low_rank_reconstruction"] = json.dumps(
                        low_rank_reconstruction_rows(
                            q,
                            k,
                            native,
                            lambda qr, kr: matrices.l[0, local_head] * (qr @ kr.T),
                        )
                    )
                    validate = should_validate_memory(base, enabled=args.validate_exact_memory)
                    variant_memory = {
                        "native": transition_memory_metrics(a, "input_dependent", validate_exact=validate),
                        "all_one": geometric_variant_memory(1.0, length, validate_exact=validate),
                        "identity": geometric_variant_memory(0.0, length, validate_exact=validate),
                        "constant_decay": geometric_variant_memory(gamma, length, validate_exact=validate),
                        "transition_shuffled": transition_memory_metrics(shuffled_a, "transition_shuffled", validate_exact=validate),
                    }
                    all_rows.extend(variant_rows(base, variants, variant_memory, spectra))
                    del variants, base, score, q, k, a, native, shuffled_a
            print(f"Mamba layer={layer} doc={sample['document_id']} L={length}", flush=True)
            flush(all_rows, spectra, output, "mamba2")
            del matrices, hidden, lag, causal
            gc.collect()
            torch.cuda.empty_cache()
    consolidate_metrics(output, "mamba2", allow_partial=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("based", "retnet", "mamba2"), required=True)
    parser.add_argument(
        "--checkpoint",
        help="Local checkpoint path; defaults to the selected model's environment-configured path.",
    )
    parser.add_argument("--manifest", default="experiments_2027/sample_manifest.jsonl")
    parser.add_argument("--out-dir", default="experiments_2027")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--lengths", default="64,128,256,384,512,768,1024")
    parser.add_argument("--document-start", type=int, default=0)
    parser.add_argument("--document-stop", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--validate-exact-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-consolidation", action="store_true")
    parser.add_argument(
        "--layers",
        help="Optional comma-separated layer override for a focused replication.",
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = PATHS[args.model]
    if args.layers:
        LAYERS[args.model] = tuple(int(value) for value in args.layers.split(",") if value.strip())
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested_lengths = {int(value) for value in args.lengths.split(",") if value.strip()}
    manifest = [row for row in read_jsonl(args.manifest) if row["length"] in requested_lengths]
    selected_documents = sorted({row["document_id"] for row in manifest})[args.document_start : args.document_stop]
    manifest = [row for row in manifest if row["document_id"] in selected_documents]
    if args.consolidate_only:
        consolidate_metrics(output, args.model, allow_partial=not args.strict_consolidation)
    else:
        {"based": run_based, "retnet": run_retnet, "mamba2": run_mamba}[args.model](args, manifest, output)


if __name__ == "__main__":
    main()
