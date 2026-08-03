#!/usr/bin/env python3
"""Exact GPT-2 attention experiments for WP0, WP2, WP3, and WP5.

The public entry points are deliberately independent of the experiment driver:

``capture_all_layers``
    Runs the transformer once and captures fused Q/K/V projections for every
    requested layer.  WP0 can additionally capture the exact tensor entering
    ``c_proj`` (the native attention core output before output projection).

``wp0_reconstruction_rows``
    Reconstructs the native causal attention in the checkpoint dtype and
    compares ``A @ V`` with the captured pre-projection output.

``analyze_capture``
    Computes the exact spectra, sink diagnostics, score diagnostics, trained
    Q/K geometry, score counterfactuals, and matched-mask interventions.

Dense token-by-token matrices are never retained.  ``AnalysisOutput.arrays``
contains compressed-output candidates such as singular values, leading
singular vectors, and rowwise diagnostic vectors.  The metric rows refer to
those arrays by stable string keys.

The implementation targets the eager GPT-2 attention implementation in
Transformers 4.57.1 and the standard ``openai-community/gpt2`` checkpoint.
It nevertheless reads the scale flags from each attention module so that a
nonstandard GPT-2 configuration is diagnosed rather than silently treated as
the standard checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
DEFAULT_CHECKPOINT = os.environ.get("GPT2_MODEL_PATH", str(MODEL_ROOT / "gpt2"))
INPUT_CONDITIONS = ("native-prefix", "explicit-boundary")
DEFAULT_ALIGNMENT_RANKS = (1, 2, 4, 8, 16)
DEFAULT_TRUNCATION_RANKS = (1, 2, 4, 8, 16, 32)
DEFAULT_DECAYS = (0.5, 0.9, 0.99)


@dataclasses.dataclass
class LayerCapture:
    """Projected head tensors for one GPT-2 layer.

    Tensors have shape ``[batch, heads, tokens, head_dim]``.  They retain the
    checkpoint inference dtype.  ``native_core`` is populated only for WP0.
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    native_core: torch.Tensor | None = None


@dataclasses.dataclass
class CaptureBundle:
    layers: dict[int, LayerCapture]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    model_output_dtype: str
    offloaded_to_cpu: bool


@dataclasses.dataclass
class AnalysisConfig:
    """Controls costly diagnostics without changing their definitions."""

    analysis_device: str = "cpu"
    analysis_dtype: str = "float32"
    svd_device: str = "cpu"
    svd_dtype: str = "float32"
    store_full_spectra: bool = True
    store_variant_spectra: bool = False
    compute_nuclear_sink_residual: bool = True
    compute_score_localization: bool = True
    compute_qk_geometry: bool = True
    compute_truncations: bool = True
    compute_counterfactuals: bool = True
    compute_matched_masks: bool = True
    compute_mask_gain: bool = True
    alignment_ranks: tuple[int, ...] = DEFAULT_ALIGNMENT_RANKS
    truncation_ranks: tuple[int, ...] = DEFAULT_TRUNCATION_RANKS
    decay_gammas: tuple[float, ...] = DEFAULT_DECAYS
    margin_c_values: tuple[float, ...] = (0.5, 0.75, 1.0)
    fixed_margin_C: float = 1.0
    scale_directions: int = 8
    array_dtype: str = "float32"


@dataclasses.dataclass
class AnalysisOutput:
    rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    arrays: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def extend(self, other: "AnalysisOutput") -> None:
        overlap = set(self.arrays).intersection(other.arrays)
        if overlap:
            raise KeyError(f"Duplicate array keys while merging output: {sorted(overlap)[:3]}")
        self.rows.extend(other.rows)
        self.arrays.update(other.arrays)


class ArrayRegistry:
    """Adds deterministic, collision-resistant keys to an AnalysisOutput."""

    def __init__(self, output: AnalysisOutput, prefix: str, dtype: str = "float32"):
        self.output = output
        self.prefix = _safe_key(prefix)
        self.dtype = np.dtype(dtype)

    def add(self, name: str, value: torch.Tensor | np.ndarray | Sequence[float]) -> str:
        raw_key = f"{self.prefix}:{_safe_key(name)}"
        key = raw_key
        if key in self.output.arrays:
            digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:8]
            key = f"{raw_key}:{digest}"
            suffix = 2
            while key in self.output.arrays:
                key = f"{raw_key}:{digest}:{suffix}"
                suffix += 1
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(self.dtype, copy=False)
        self.output.arrays[key] = np.ascontiguousarray(array)
        return key


def _safe_key(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "array"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).replace("torch.", "").lower()
    choices = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if normalized not in choices:
        raise ValueError(f"Unsupported floating dtype: {name}")
    return choices[normalized]


def _input_device(model: torch.nn.Module) -> torch.device:
    transformer = getattr(model, "transformer", model)
    embeddings = getattr(transformer, "wte", None)
    if embeddings is not None:
        return embeddings.weight.device
    return next(model.parameters()).device


def load_gpt2(
    checkpoint: str = DEFAULT_CHECKPOINT,
    *,
    device_map: str | Mapping[str, Any] | None = "auto",
    dtype: str | torch.dtype = "auto",
    local_files_only: bool = True,
):
    """Load GPT-2 with eager attention so native behavior is inspectable."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
        attn_implementation="eager",
    )
    model.eval()
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    return model, tokenizer


def prepare_condition_input_ids(
    base_token_ids: Sequence[int] | torch.Tensor,
    length: int,
    condition: str,
    boundary_token_id: int,
) -> torch.Tensor:
    """Create a nested GPT-2 prefix under either preregistered condition."""

    if condition not in INPUT_CONDITIONS:
        raise ValueError(f"condition must be one of {INPUT_CONDITIONS}, got {condition!r}")
    if length < 1:
        raise ValueError("length must be positive")
    ids = torch.as_tensor(base_token_ids, dtype=torch.long).flatten()
    required = length if condition == "native-prefix" else length - 1
    if ids.numel() < required:
        raise ValueError(f"Need at least {required} base tokens for length {length}; got {ids.numel()}")
    if condition == "native-prefix":
        selected = ids[:length]
    else:
        boundary = torch.tensor([int(boundary_token_id)], dtype=torch.long)
        selected = torch.cat((boundary, ids[: length - 1]))
    return selected.unsqueeze(0)


def _offload(value: torch.Tensor, to_cpu: bool) -> torch.Tensor:
    value = value.detach().contiguous()
    return value.cpu() if to_cpu else value


def capture_all_layers(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    layer_indices: Sequence[int] | None = None,
    capture_native_core: bool = False,
    offload_to_cpu: bool = True,
) -> CaptureBundle:
    """Capture all requested layers in one transformer forward pass.

    Q/K/V are captured at ``c_attn``.  When requested, a pre-hook on ``c_proj``
    captures the core output after native attention/dropout and immediately
    before the output projection.  ``model.eval()`` is required so attention
    dropout is disabled.
    """

    if model.training:
        raise ValueError("capture_all_layers requires model.eval(); dropout must be disabled")
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "h"):
        raise TypeError("Expected a GPT-2 model with model.transformer.h")
    total_layers = len(transformer.h)
    layers = list(range(total_layers)) if layer_indices is None else [int(x) for x in layer_indices]
    if not layers:
        raise ValueError("At least one layer must be captured")
    if min(layers) < 0 or max(layers) >= total_layers:
        raise IndexError(f"Layer indices {layers} are outside [0, {total_layers})")

    partial: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layers}
    handles: list[Any] = []

    def make_qkv_hook(layer: int, attention: torch.nn.Module):
        def hook(module, _args, output):
            projected = output[0] if isinstance(output, (tuple, list)) else output
            if projected.ndim != 3:
                raise RuntimeError(f"Layer {layer} c_attn output has shape {tuple(projected.shape)}")
            query, key, value = projected.split(attention.split_size, dim=-1)
            batch, tokens, _ = query.shape
            heads = int(attention.num_heads)
            head_dim = int(attention.head_dim)

            def shape(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.view(batch, tokens, heads, head_dim).transpose(1, 2)

            partial[layer]["query"] = _offload(shape(query), offload_to_cpu)
            partial[layer]["key"] = _offload(shape(key), offload_to_cpu)
            partial[layer]["value"] = _offload(shape(value), offload_to_cpu)

        return hook

    def make_core_hook(layer: int):
        def hook(module, args, kwargs):
            core = args[0] if args else kwargs.get("hidden_states")
            if core is None or core.ndim != 3:
                raise RuntimeError(f"Could not capture layer {layer} c_proj input")
            batch, tokens, width = core.shape
            heads = int(transformer.h[layer].attn.num_heads)
            head_dim = int(transformer.h[layer].attn.head_dim)
            if width != heads * head_dim:
                raise RuntimeError(
                    f"Layer {layer} c_proj input width {width} != {heads} * {head_dim}",
                )
            shaped = core.view(batch, tokens, heads, head_dim).transpose(1, 2)
            partial[layer]["native_core"] = _offload(shaped, offload_to_cpu)

        return hook

    try:
        for layer in layers:
            attention = transformer.h[layer].attn
            handles.append(attention.c_attn.register_forward_hook(make_qkv_hook(layer, attention)))
            if capture_native_core:
                handles.append(
                    attention.c_proj.register_forward_pre_hook(make_core_hook(layer), with_kwargs=True),
                )
        device = _input_device(model)
        input_on_device = input_ids.to(device)
        mask_on_device = attention_mask.to(device) if attention_mask is not None else None
        with torch.inference_mode():
            transformer(
                input_ids=input_on_device,
                attention_mask=mask_on_device,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    captured: dict[int, LayerCapture] = {}
    for layer in layers:
        required = {"query", "key", "value"}
        missing = required.difference(partial[layer])
        if capture_native_core and "native_core" not in partial[layer]:
            missing.add("native_core")
        if missing:
            raise RuntimeError(f"Layer {layer} capture missing {sorted(missing)}")
        captured[layer] = LayerCapture(
            query=partial[layer]["query"],
            key=partial[layer]["key"],
            value=partial[layer]["value"],
            native_core=partial[layer].get("native_core"),
        )
    first_dtype = captured[layers[0]].query.dtype
    stored_ids = input_ids.detach().cpu()
    stored_mask = attention_mask.detach().cpu() if attention_mask is not None else None
    return CaptureBundle(
        layers=captured,
        input_ids=stored_ids,
        attention_mask=stored_mask,
        model_output_dtype=str(first_dtype),
        offloaded_to_cpu=offload_to_cpu,
    )


def score_denominator(attention: torch.nn.Module, layer: int) -> float:
    denominator = 1.0
    if bool(getattr(attention, "scale_attn_weights", True)):
        denominator *= math.sqrt(int(attention.head_dim))
    if bool(getattr(attention, "scale_attn_by_inverse_layer_idx", False)):
        layer_idx = getattr(attention, "layer_idx", layer)
        denominator *= float(int(layer_idx) + 1)
    return denominator


def causal_softmax_from_scores(
    scores: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply GPT-2 causal masking and row softmax to scaled raw scores."""

    if scores.ndim not in (2, 4):
        raise ValueError(f"Expected [T,T] or [B,H,T,T] scores, got {tuple(scores.shape)}")
    tokens = scores.shape[-1]
    if scores.shape[-2] != tokens:
        raise ValueError("Causal self-attention scores must be square")
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=scores.device).tril()
    mask_value = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~causal, mask_value)
    if attention_mask is not None:
        mask = attention_mask.to(scores.device)
        if mask.ndim == 2:
            if scores.ndim == 2:
                if mask.shape[0] != 1:
                    raise ValueError("A 2-D score matrix only supports batch size one")
                masked = masked.masked_fill(mask[0].view(1, -1) == 0, mask_value)
            else:
                masked = masked.masked_fill(mask[:, None, None, :] == 0, mask_value)
        elif mask.ndim == 4:
            masked = masked + mask.to(dtype=masked.dtype)
        else:
            raise ValueError(f"Unsupported attention mask shape {tuple(mask.shape)}")
    return torch.softmax(masked, dim=-1)


def reconstruct_native_attention(
    attention: torch.nn.Module,
    capture: LayerCapture,
    layer: int,
    *,
    attention_mask: torch.Tensor | None = None,
    analysis_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(A @ V, A, unmasked_scaled_scores)``.

    With ``analysis_dtype=None`` the calculation stays in the captured native
    dtype for WP0.  Passing FP32/FP64 is appropriate for diagnostics.
    """

    query, key, value = capture.query, capture.key, capture.value
    if analysis_dtype is not None:
        query = query.to(dtype=analysis_dtype)
        key = key.to(dtype=analysis_dtype)
        value = value.to(dtype=analysis_dtype)
    elif bool(getattr(attention, "reorder_and_upcast_attn", False)):
        query = query.float()
        key = key.float()
    scores = torch.matmul(query, key.transpose(-1, -2)) / score_denominator(attention, layer)
    matrix = causal_softmax_from_scores(scores, attention_mask=attention_mask)
    matrix_for_value = matrix.to(dtype=value.dtype)
    core = torch.matmul(matrix_for_value, value)
    return core, matrix, scores


def wp0_reconstruction_rows(
    model: torch.nn.Module,
    bundle: CaptureBundle,
    *,
    layers: Sequence[int] | None = None,
    heads_by_layer: Mapping[int, Sequence[int]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compare reconstructed ``A @ V`` to the native pre-``c_proj`` core."""

    metadata = dict(metadata or {})
    selected_layers = sorted(bundle.layers) if layers is None else [int(x) for x in layers]
    rows: list[dict[str, Any]] = []
    for layer in selected_layers:
        capture = bundle.layers[layer]
        if capture.native_core is None:
            raise ValueError(f"Layer {layer} has no native_core; recapture with capture_native_core=True")
        attention = model.transformer.h[layer].attn
        # The efficient main capture defaults to CPU offload, but native FP16
        # attention should be reconstructed on the actual layer device.
        layer_device = attention.c_attn.weight.device
        working_capture = LayerCapture(
            query=capture.query.to(layer_device),
            key=capture.key.to(layer_device),
            value=capture.value.to(layer_device),
        )
        working_mask = (
            bundle.attention_mask.to(layer_device) if bundle.attention_mask is not None else None
        )
        core, matrix, _ = reconstruct_native_attention(
            attention,
            working_capture,
            layer,
            attention_mask=working_mask,
            analysis_dtype=None,
        )
        native = capture.native_core.to(core.device, dtype=core.dtype)
        available = list(range(core.shape[1]))
        heads = available if heads_by_layer is None else [int(x) for x in heads_by_layer[layer]]
        for head in heads:
            difference = (core[:, head] - native[:, head]).float()
            native_float = native[:, head].float()
            denominator = max(float(torch.linalg.vector_norm(native_float)), 1e-12)
            rows.append(
                {
                    **metadata,
                    "work_package": "WP0",
                    "layer": layer,
                    "head_or_channel": head,
                    "length": int(core.shape[-2]),
                    "relative_error": float(torch.linalg.vector_norm(difference)) / denominator,
                    "max_absolute_error": float(difference.abs().max()),
                    "qkv_dtype": str(capture.query.dtype),
                    "attention_dtype": str(matrix.dtype),
                    "native_core_dtype": str(capture.native_core.dtype),
                    "tensor_shape": _json(list(core[:, head].shape)),
                    "causal_mask": True,
                    "scale_denominator": score_denominator(attention, layer),
                    "dropout_training": bool(attention.training),
                    "included_operations": "causal_mask,scaled_qk,row_softmax,A@V",
                    "excluded_operations": "c_proj,resid_dropout,residual,layer_norm",
                },
            )
        selected = torch.as_tensor(heads, device=core.device, dtype=torch.long)
        all_difference = (core.index_select(1, selected) - native.index_select(1, selected)).float()
        all_native = native.index_select(1, selected).float()
        rows.append(
            {
                **metadata,
                "work_package": "WP0",
                "layer": layer,
                "head_or_channel": "all_selected",
                "length": int(core.shape[-2]),
                "relative_error": float(torch.linalg.vector_norm(all_difference))
                / max(float(torch.linalg.vector_norm(all_native)), 1e-12),
                "max_absolute_error": float(all_difference.abs().max()),
                "qkv_dtype": str(capture.query.dtype),
                "attention_dtype": str(matrix.dtype),
                "native_core_dtype": str(capture.native_core.dtype),
                "tensor_shape": _json(list(all_difference.shape)),
                "causal_mask": True,
                "scale_denominator": score_denominator(attention, layer),
                "dropout_training": bool(attention.training),
                "included_operations": "causal_mask,scaled_qk,row_softmax,A@V",
                "excluded_operations": "c_proj,resid_dropout,residual,layer_norm",
            },
        )
        del working_capture, working_mask, core, matrix, native
    return rows


def _exact_svdvals(matrix: torch.Tensor, config: AnalysisConfig) -> torch.Tensor:
    device = torch.device(config.svd_device)
    dtype = _torch_dtype(config.svd_dtype)
    work = matrix.detach().to(device=device, dtype=dtype)
    try:
        values = torch.linalg.svdvals(work)
    except RuntimeError:
        if device.type == "cpu":
            raise
        values = torch.linalg.svdvals(work.float().cpu())
    return values.detach().double().cpu()


def _rank_metrics(singular_values: torch.Tensor, *, normalizer: int | None = None) -> dict[str, Any]:
    singular_values = singular_values.detach().double().cpu().flatten()
    squared = singular_values.square()
    nuclear = singular_values.sum()
    fro2 = squared.sum()
    top = singular_values[0] if singular_values.numel() else torch.tensor(0.0)
    if float(nuclear) <= 0.0 or float(fro2) <= 0.0 or float(top) <= 0.0:
        result: dict[str, Any] = {
            "srank": 0.0,
            "erank_1": 0.0,
            "erank_2": 0.0,
            "operator_norm": 0.0,
            "frobenius_norm": 0.0,
            "nuclear_norm": 0.0,
            "best_rank1_error": 0.0,
        }
    else:
        p1 = singular_values / nuclear
        p2 = squared / fro2
        entropy1 = -(p1[p1 > 0] * p1[p1 > 0].log()).sum()
        entropy2 = -(p2[p2 > 0] * p2[p2 > 0].log()).sum()
        result = {
            "srank": float(fro2 / top.square()),
            "erank_1": float(entropy1.exp()),
            "erank_2": float(entropy2.exp()),
            "operator_norm": float(top),
            "frobenius_norm": float(fro2.sqrt()),
            "nuclear_norm": float(nuclear),
            "best_rank1_error": math.sqrt(max(0.0, 1.0 - float(top.square() / fro2))),
        }
    if singular_values.numel():
        if float(fro2) > 0.0:
            cumulative = squared.cumsum(0) / fro2
            for threshold in (0.90, 0.95, 0.99):
                rank = int(torch.searchsorted(cumulative, threshold).item()) + 1
                result[f"r{int(threshold * 100)}"] = rank
        else:
            for threshold in (90, 95, 99):
                result[f"r{threshold}"] = 0
        if float(top) > 0.0:
            ratios = singular_values / top
            result["count_rel_gt_1e-2"] = int((ratios > 1e-2).sum())
            result["count_rel_gt_1e-3"] = int((ratios > 1e-3).sum())
        else:
            result["count_rel_gt_1e-2"] = 0
            result["count_rel_gt_1e-3"] = 0
    if normalizer is not None:
        denominator = max(int(normalizer), 1)
        for name in (
            "srank",
            "erank_1",
            "erank_2",
            "r90",
            "r95",
            "r99",
            "count_rel_gt_1e-2",
            "count_rel_gt_1e-3",
        ):
            if name in result:
                result[f"{name}_fraction"] = float(result[name]) / denominator
    return result


def _binary_entropy_bound(rho: float, length: int) -> float:
    if not math.isfinite(rho) or rho < 0.0 or rho > 0.5 or length < 2:
        return math.nan
    if rho == 0.0:
        entropy = 0.0
    else:
        entropy = -rho * math.log(rho) - (1.0 - rho) * math.log1p(-rho)
    return math.exp(entropy + rho * math.log(length - 1))


def sink_diagnostics(
    attention_matrix: torch.Tensor,
    *,
    native_singular_values: torch.Tensor | None,
    config: AnalysisConfig,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Compute the theorem's exact first-column S/D/rho diagnostics."""

    matrix = attention_matrix.float()
    length = matrix.shape[0]
    sink_column = matrix[:, 0]
    sink_S = float(sink_column.square().sum())
    residual = matrix.clone()
    residual[:, 0] = 0.0
    sink_D = float(residual.square().sum())  # Direct residual calculation by design.
    total = sink_S + sink_D
    rho2 = sink_D / total if total > 0.0 else math.nan
    d_over_s = sink_D / sink_S if sink_S > 0.0 else math.inf

    denominator = 1.0 - sink_column[1:]
    conditional = matrix[1:, 1:].clone()
    causal = torch.ones(length - 1, length - 1, device=matrix.device, dtype=torch.bool).tril()
    conditional = conditional.masked_fill(~causal, 0.0)
    valid = denominator.abs() > 10.0 * torch.finfo(matrix.dtype).eps
    chi = torch.full_like(denominator, torch.nan)
    if valid.any():
        chi[valid] = (conditional[valid] / denominator[valid, None]).square().sum(dim=-1)

    residual_singular_values = None
    residual_nuclear = math.nan
    rho1 = math.nan
    if config.compute_nuclear_sink_residual:
        residual_singular_values = _exact_svdvals(residual, config)
        residual_nuclear = float(residual_singular_values.sum())
        root_S = math.sqrt(sink_S)
        rho1 = residual_nuclear / (root_S + residual_nuclear) if root_S + residual_nuclear > 0 else math.nan

    best_error = math.nan
    stable_minus_one = math.nan
    if native_singular_values is not None:
        rank = _rank_metrics(native_singular_values)
        best_error = float(rank["best_rank1_error"])
        stable_minus_one = float(rank["srank"]) - 1.0
    result = {
        "sink_S": sink_S,
        "sink_D": sink_D,
        "sink_D_over_S": d_over_s,
        "sink_D_over_S_log_length": d_over_s * math.log(length) if length > 1 else math.nan,
        "sink_rho2": rho2,
        "sink_rho1": rho1,
        "sink_column_relative_error": math.sqrt(max(rho2, 0.0)) if math.isfinite(rho2) else math.nan,
        "best_rank1_error": best_error,
        "sink_to_best_rank1_error_ratio": (
            math.sqrt(rho2) / best_error
            if math.isfinite(rho2) and math.isfinite(best_error) and best_error > 0
            else math.nan
        ),
        "srank_minus_1": stable_minus_one,
        "srank_minus_1_upper_bound": d_over_s,
        "sink_residual_nuclear_norm": residual_nuclear,
        "sink_rho1_log_length": rho1 * math.log(length) if math.isfinite(rho1) else math.nan,
        "sink_erank2_upper_bound": _binary_entropy_bound(rho2, length),
        "sink_erank1_upper_bound": _binary_entropy_bound(rho1, length),
        "conditional_collision_valid_fraction": float(valid.float().mean()) if valid.numel() else 1.0,
        "conditional_collision_median": _quantile(chi[valid], 0.5) if valid.any() else math.nan,
        "conditional_collision_p95": _quantile(chi[valid], 0.95) if valid.any() else math.nan,
        "conditional_collision_scaled_median": (
            _quantile(chi[valid] * torch.arange(1, length, device=matrix.device)[valid], 0.5)
            if valid.any()
            else math.nan
        ),
        "_conditional_collision": chi,
    }
    return result, residual_singular_values


def column_sink_summary(attention_matrix: torch.Tensor, column: int) -> dict[str, Any]:
    """Exploratory column-energy diagnostic on causally accessible rows."""

    matrix = attention_matrix.float()
    length = matrix.shape[0]
    column = int(column)
    if not 0 <= column < length:
        raise IndexError(column)
    accessible = matrix[column:, :]
    selected = accessible[:, column]
    residual = accessible.clone()
    residual[:, column] = 0.0
    sink_S = float(selected.square().sum())
    sink_D = float(residual.square().sum())
    total = sink_S + sink_D
    return {
        "column_0based": column,
        "column_1based": column + 1,
        "accessible_rows": length - column,
        "column_energy": sink_S,
        "residual_energy_accessible_rows": sink_D,
        "rho2_accessible_rows": sink_D / total if total > 0 else math.nan,
    }


def _quantile(value: torch.Tensor, probability: float) -> float:
    if value.numel() == 0:
        return math.nan
    return float(torch.quantile(value.float(), float(probability)))


def pre_softmax_diagnostics(
    scores: torch.Tensor,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Compute accessible-row margins, collisions, ranges, and violations."""

    scores = scores.float()
    length = scores.shape[0]
    if length < 2:
        return {
            "margin_median": math.nan,
            "margin_p05": math.nan,
            "collision_scaled_median": math.nan,
            "collision_scaled_p95": math.nan,
            "accessible_oscillation_B": 0.0,
            "_margin": torch.empty(0, device=scores.device),
            "_collision": torch.empty(0, device=scores.device),
            "_collision_scaled": torch.empty(0, device=scores.device),
            "_nonsink_range": torch.empty(0, device=scores.device),
            "_accessible_oscillation": torch.zeros(1, device=scores.device),
        }
    rows = torch.arange(length, device=scores.device)[:, None]
    columns = torch.arange(length, device=scores.device)[None, :]
    causal = columns <= rows
    nonsink = (columns >= 1) & causal & (rows >= 1)
    neg_inf = torch.tensor(-torch.inf, device=scores.device)
    pos_inf = torch.tensor(torch.inf, device=scores.device)
    nonsink_max = scores.masked_fill(~nonsink, neg_inf).amax(dim=-1)[1:]
    nonsink_min = scores.masked_fill(~nonsink, pos_inf).amin(dim=-1)[1:]
    margin = scores[1:, 0] - nonsink_max
    nonsink_range = nonsink_max - nonsink_min

    nonsink_logits = scores.masked_fill(~nonsink, neg_inf)[1:]
    log_collision = torch.logsumexp(2.0 * nonsink_logits, dim=-1) - 2.0 * torch.logsumexp(
        nonsink_logits,
        dim=-1,
    )
    collision = log_collision.exp()
    nonsink_count = torch.arange(1, length, device=scores.device, dtype=torch.float32)
    collision_scaled = nonsink_count * collision

    accessible_max = scores.masked_fill(~causal, neg_inf).amax(dim=-1)
    accessible_min = scores.masked_fill(~causal, pos_inf).amin(dim=-1)
    accessible_oscillation = accessible_max - accessible_min
    log_i = torch.arange(2, length + 1, device=scores.device, dtype=torch.float32).log()
    design = torch.stack((log_i, torch.ones_like(log_i)), dim=-1)
    fit = torch.linalg.lstsq(design, margin[:, None]).solution[:, 0]
    slope, intercept = float(fit[0]), float(fit[1])

    fixed_fractions: dict[str, float] = {}
    descriptive_C95: dict[str, float] = {}
    for c in config.margin_c_values:
        key = f"c_{float(c):g}"
        threshold = float(c) * log_i - float(config.fixed_margin_C)
        fixed_fractions[key] = float((margin >= threshold).float().mean())
        # This fitted C makes the condition hold on 95% of observed rows.
        descriptive_C95[key] = _quantile(float(c) * log_i - margin, 0.95)

    return {
        "margin_median": _quantile(margin, 0.5),
        "margin_p05": _quantile(margin, 0.05),
        "margin_log_i_slope": slope,
        "margin_log_i_intercept": intercept,
        "fixed_margin_C": float(config.fixed_margin_C),
        "fixed_margin_row_fractions": _json(fixed_fractions),
        "descriptive_C_p95": _json(descriptive_C95),
        "collision_scaled_median": _quantile(collision_scaled, 0.5),
        "collision_scaled_p95": _quantile(collision_scaled, 0.95),
        "nonsink_range_median": _quantile(nonsink_range, 0.5),
        "nonsink_range_p95": _quantile(nonsink_range, 0.95),
        "accessible_oscillation_B": float(accessible_oscillation.max()),
        "accessible_oscillation_median": _quantile(accessible_oscillation, 0.5),
        "accessible_oscillation_p95": _quantile(accessible_oscillation, 0.95),
        "_margin": margin,
        "_collision": collision,
        "_collision_scaled": collision_scaled,
        "_nonsink_range": nonsink_range,
        "_accessible_oscillation": accessible_oscillation,
    }


def _thin_product_svd(
    left: torch.Tensor,
    right: torch.Tensor,
    denominator: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact thin SVD of ``left @ right.T / denominator`` via QR."""

    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right, mode="reduced")
    core = (r_left @ r_right.transpose(-1, -2)) / float(denominator)
    core_u, singular_values, core_vh = torch.linalg.svd(core, full_matrices=False)
    token_u = q_left @ core_u
    token_v = q_right @ core_vh.transpose(-1, -2)
    return token_u, singular_values, token_v


def score_localization_diagnostics(
    query: torch.Tensor,
    key: torch.Tensor,
    scores: torch.Tensor,
    denominator: float,
) -> dict[str, Any]:
    """Leading-mode localization of the offline row-centered score matrix."""

    length = scores.shape[0]
    centered_key = key - key.mean(dim=0, keepdim=True)
    token_u, singular_values, token_v = _thin_product_svd(query, centered_key, denominator)
    u = token_u[:, 0]
    v = token_v[:, 0]
    positive_count = int((u > 0).sum())
    negative_count = int((u < 0).sum())
    if positive_count < negative_count or (positive_count == negative_count and float(v[0]) < 0.0):
        u = -u
        v = -v
    sigma1 = singular_values[0]
    sigma2 = singular_values[1] if singular_values.numel() > 1 else torch.zeros_like(sigma1)
    centered_scores = scores - scores.mean(dim=-1, keepdim=True)
    residual = centered_scores - sigma1 * torch.outer(u, v)
    residual_inf = residual.abs().amax(dim=-1)

    if length >= 2:
        rows = torch.arange(length, device=scores.device)[:, None]
        columns = torch.arange(length, device=scores.device)[None, :]
        nonsink = (columns >= 1) & (columns <= rows) & (rows >= 1)
        vmax = v[None, :].expand(length, -1).masked_fill(~nonsink, -torch.inf).amax(dim=-1)[1:]
        vmin = v[None, :].expand(length, -1).masked_fill(~nonsink, torch.inf).amin(dim=-1)[1:]
        aligned_gap_lhs = sigma1 * u[1:] * (v[0] - vmax) - 2.0 * residual_inf[1:]
        aligned_background_lhs = sigma1 * u[1:] * (vmax - vmin) + 2.0 * residual_inf[1:]
    else:
        aligned_gap_lhs = torch.empty(0, device=scores.device)
        aligned_background_lhs = torch.empty(0, device=scores.device)

    abs_v_squared = v.square()
    ratio = float(sigma1 / sigma2) if float(sigma2) > 0.0 else math.inf
    return {
        "score_centered_sigma1": float(sigma1),
        "score_centered_sigma2": float(sigma2),
        "score_centered_sigma1_over_sigma2": ratio,
        "score_v1_first_mass": float(abs_v_squared[0]),
        "score_v1_first4_mass": float(abs_v_squared[: min(4, length)].sum()),
        "score_v1_first16_mass": float(abs_v_squared[: min(16, length)].sum()),
        "score_v1_argmax_abs_0based": int(v.abs().argmax()),
        "score_v1_argmax_abs_1based": int(v.abs().argmax()) + 1,
        "score_u1_positive_fraction": float((u > 0).float().mean()),
        "score_u1_min": float(u.min()),
        "score_u1_p05": _quantile(u, 0.05),
        "score_u1_median": _quantile(u, 0.5),
        "score_leading_residual_inf_max": float(residual_inf.max()),
        "score_leading_residual_inf_p95": _quantile(residual_inf, 0.95),
        "aligned_gap_lhs_min": float(aligned_gap_lhs.min()) if aligned_gap_lhs.numel() else math.nan,
        "aligned_gap_lhs_p05": _quantile(aligned_gap_lhs, 0.05),
        "aligned_gap_lhs_median": _quantile(aligned_gap_lhs, 0.5),
        "aligned_background_lhs_max": (
            float(aligned_background_lhs.max()) if aligned_background_lhs.numel() else math.nan
        ),
        "aligned_background_lhs_p95": _quantile(aligned_background_lhs, 0.95),
        "aligned_background_lhs_median": _quantile(aligned_background_lhs, 0.5),
        "_u1": u,
        "_v1": v,
        "_centered_singular_values": singular_values,
        "_residual_inf": residual_inf,
        "_aligned_gap_lhs": aligned_gap_lhs,
        "_aligned_background_lhs": aligned_background_lhs,
    }


def _factor_svd(factor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.linalg.svd(factor, full_matrices=False)


def qk_geometry_diagnostics(
    query: torch.Tensor,
    key: torch.Tensor,
    denominator: float,
    config: AnalysisConfig,
    registry: ArrayRegistry,
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    """Numerical ranks and token/projection-side Q/K geometry."""

    uq, sq, vhq = _factor_svd(query)
    uk, sk, vhk = _factor_svd(key)
    _, score_s, _ = _thin_product_svd(query, key, denominator)
    normalizer = min(query.shape)
    q_metrics = _rank_metrics(sq.detach().double().cpu(), normalizer=normalizer)
    k_metrics = _rank_metrics(sk.detach().double().cpu(), normalizer=normalizer)
    score_metrics = _rank_metrics(score_s.detach().double().cpu(), normalizer=normalizer)

    q_ref = registry.add("q_singular_values", sq)
    k_ref = registry.add("k_singular_values", sk)
    score_ref = registry.add("score_thin_singular_values", score_s)
    alignments: list[dict[str, Any]] = []
    projection_overlaps: list[dict[str, Any]] = []
    valid_dimension = min(uq.shape[1], uk.shape[1], vhq.shape[0], vhk.shape[0])
    head_dim = query.shape[1]
    for rank in config.alignment_ranks:
        rank = int(rank)
        if rank < 1 or rank > valid_dimension:
            continue
        token_cross = uq[:, :rank].transpose(0, 1) @ uk[:, :rank]
        canonical = torch.linalg.svdvals(token_cross)
        alignment_score = float(token_cross.square().sum() / rank)
        procrustes = math.sqrt(max(0.0, 2.0 * rank - 2.0 * float(canonical.sum())))
        canonical_ref = registry.add(f"token_canonical_correlations_r{rank}", canonical)
        projection_cross = vhq[:rank] @ vhk[:rank].transpose(0, 1)
        scaled_entries = math.sqrt(head_dim) * projection_cross.flatten()
        projection_sv = torch.linalg.svdvals(projection_cross)
        entries_ref = registry.add(f"projection_overlap_scaled_entries_r{rank}", scaled_entries)
        projection_sv_ref = registry.add(f"projection_overlap_singular_values_r{rank}", projection_sv)
        alignments.append(
            {
                "rank": rank,
                "alignment_score": alignment_score,
                "independent_subspace_reference": rank / query.shape[0],
                "procrustes_distance": procrustes,
                "procrustes_distance_normalized": procrustes / math.sqrt(2.0 * rank),
                "canonical_correlations_ref": canonical_ref,
            },
        )
        projection_overlaps.append(
            {
                "rank": rank,
                "scaled_entry_mean": float(scaled_entries.mean()),
                "scaled_entry_variance": float(scaled_entries.var(unbiased=False)),
                "scaled_entry_max_abs": float(scaled_entries.abs().max()),
                "scaled_entries_ref": entries_ref,
                "singular_values_ref": projection_sv_ref,
            },
        )

    directions = min(config.scale_directions, sq.numel(), sk.numel())
    scale_products = (
        sq[:directions, None]
        * sk[None, :directions]
        / (query.shape[0] * math.sqrt(query.shape[1]))
    )
    scale_ref = registry.add("qk_singular_scale_products", scale_products)
    result = {
        "q_rank_metrics": _json(q_metrics),
        "k_rank_metrics": _json(k_metrics),
        "score_rank_metrics": _json(score_metrics),
        "q_spectrum_ref": q_ref,
        "k_spectrum_ref": k_ref,
        "score_thin_spectrum_ref": score_ref,
        "token_alignment": _json(alignments),
        "projection_overlap": _json(projection_overlaps),
        "qk_scale_products_ref": scale_ref,
        "qk_scale_denominator": query.shape[0] * math.sqrt(query.shape[1]),
    }
    return result, (uq, sq, vhq, uk, sk, vhk, score_s)


def _rowwise_nonsink_median(scores: torch.Tensor) -> torch.Tensor:
    """Exact midpoint median of positions 2..i in every mathematical row i."""

    length = scores.shape[0]
    medians = scores[:, 0].clone()
    if length < 2:
        return medians
    values = scores[:, 1:]
    row = torch.arange(length, device=scores.device)[:, None]
    column = torch.arange(1, length, device=scores.device)[None, :]
    accessible = (column <= row) & (row >= 1)
    ordered = values.masked_fill(~accessible, torch.inf).sort(dim=-1).values
    count = torch.arange(length, device=scores.device)
    lower = ((count - 1).clamp_min(0) // 2).view(-1, 1)
    upper = (count.clamp_min(0) // 2).view(-1, 1)
    low_value = ordered.gather(1, lower).squeeze(1)
    high_value = ordered.gather(1, upper).squeeze(1)
    medians[1:] = 0.5 * (low_value[1:] + high_value[1:])
    return medians


def score_counterfactuals(scores: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Construct native, neutralize-first, and late-spike analytical logits."""

    length = scores.shape[0]
    medians = _rowwise_nonsink_median(scores)
    neutral = scores.clone()
    if length >= 2:
        neutral[1:, 0] = medians[1:]
    # The plan uses mathematical position floor(ell/2); convert to zero-based.
    late_position = max(math.floor(length / 2) - 1, 0)
    late = neutral.clone()
    if length >= 2 and late_position >= 1:
        advantage = scores[:, 0] - medians
        accessible_rows = torch.arange(length, device=scores.device) >= late_position
        accessible_rows[:1] = False
        late[accessible_rows, late_position] = medians[accessible_rows] + advantage[accessible_rows]
    variants = {
        "native_softmax": causal_softmax_from_scores(scores),
        "neutralize_first": causal_softmax_from_scores(neutral),
        "late_spike": causal_softmax_from_scores(late),
    }
    metadata = {
        "late_spike_position_0based": late_position,
        "late_spike_position_1based": late_position + 1,
        "intervention_label": "analytical_not_language_model_forward",
    }
    return variants, metadata


_MASK_GAIN_CACHE: dict[tuple[int, float], float] = {}


def causal_mask_gain(length: int, gamma: float) -> float:
    """Exact operator norm of ``gamma**(i-j) 1{i>=j}``, cached by shape."""

    key = (int(length), float(gamma))
    if key not in _MASK_GAIN_CACHE:
        position = torch.arange(length, dtype=torch.float64)
        lag = position[:, None] - position[None, :]
        mask = float(gamma) ** lag.clamp_min(0)
        mask = mask * (lag >= 0)
        _MASK_GAIN_CACHE[key] = float(torch.linalg.svdvals(mask)[0])
    return _MASK_GAIN_CACHE[key]


def matched_mask_variants(
    scores: torch.Tensor,
    gammas: Iterable[float] = DEFAULT_DECAYS,
) -> dict[str, torch.Tensor]:
    """WP5 operators with trained Q/K fixed and only mask/normalization varied."""

    length = scores.shape[0]
    position = torch.arange(length, device=scores.device)
    lag = position[:, None] - position[None, :]
    causal = lag >= 0
    output = {
        "native_softmax": causal_softmax_from_scores(scores),
        "causal_raw_all_one": scores * causal,
    }
    for gamma in gammas:
        decay = torch.exp(math.log(float(gamma)) * lag.clamp_min(0).float())
        output[f"causal_raw_decay_{float(gamma):g}"] = scores * decay * causal
    return output


def _rank_row(
    matrix: torch.Tensor,
    config: AnalysisConfig,
    registry: ArrayRegistry,
    *,
    variant: str,
    base: Mapping[str, Any],
    store_spectrum: bool,
) -> tuple[dict[str, Any], torch.Tensor]:
    singular_values = _exact_svdvals(matrix, config)
    row = {**base, "operator_variant": variant, **_rank_metrics(singular_values)}
    if store_spectrum:
        row["spectrum_ref"] = registry.add(f"{variant}_singular_values", singular_values)
        squared = singular_values.square()
        if float(squared.sum()) > 0:
            row["cumulative_squared_energy_first32_ref"] = registry.add(
                f"{variant}_cumulative_squared_energy_first32",
                (squared.cumsum(0) / squared.sum())[:32],
            )
    return row, singular_values


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)


def truncation_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    scores: torch.Tensor,
    native_attention: torch.Tensor,
    native_singular_values: torch.Tensor,
    factor_svd: tuple[torch.Tensor, ...],
    denominator: float,
    config: AnalysisConfig,
    registry: ArrayRegistry,
    base: Mapping[str, Any],
) -> list[dict[str, Any]]:
    uq, sq, vhq, uk, sk, vhk, _score_s = factor_svd
    native_metrics = _rank_metrics(native_singular_values)
    native_score_norm = max(float(torch.linalg.vector_norm(scores)), 1e-12)
    native_mixer_norm = max(float(torch.linalg.vector_norm(native_attention)), 1e-12)
    valid = min(sq.numel(), sk.numel())
    requested = sorted({int(r) for r in config.truncation_ranks if 0 < int(r) <= valid})
    ranks: list[tuple[str, int]] = [(f"r{rank}", rank) for rank in requested]
    ranks.append(("full", valid))
    rows: list[dict[str, Any]] = []
    q_total = max(float(sq.square().sum()), 1e-12)
    k_total = max(float(sk.square().sum()), 1e-12)
    for label, rank in ranks:
        if rank >= valid:
            approx_scores = scores
            approx_attention = native_attention
            approx_singular_values = native_singular_values
        else:
            query_r = (uq[:, :rank] * sq[:rank]) @ vhq[:rank]
            key_r = (uk[:, :rank] * sk[:rank]) @ vhk[:rank]
            approx_scores = (query_r @ key_r.transpose(0, 1)) / denominator
            approx_attention = causal_softmax_from_scores(approx_scores)
            approx_singular_values = _exact_svdvals(approx_attention, config)
        metrics = _rank_metrics(approx_singular_values)
        row = {
            **base,
            "work_package": "WP3",
            "operator_variant": f"softmax_trunc_{label}",
            "truncation_rank": rank,
            "truncation_is_full": label == "full",
            "truncation_is_algebraically_full": rank == valid,
            **metrics,
            "q_squared_energy_retained": float(sq[:rank].square().sum()) / q_total,
            "k_squared_energy_retained": float(sk[:rank].square().sum()) / k_total,
            "score_relative_frobenius_error": float(torch.linalg.vector_norm(approx_scores - scores))
            / native_score_norm,
            "mixer_relative_frobenius_error": float(
                torch.linalg.vector_norm(approx_attention - native_attention),
            )
            / native_mixer_norm,
            "leading_singular_value_relative_error": _relative_error(
                float(metrics["operator_norm"]),
                float(native_metrics["operator_norm"]),
            ),
            "srank_relative_error": _relative_error(
                float(metrics["srank"]),
                float(native_metrics["srank"]),
            ),
            "erank_1_relative_error": _relative_error(
                float(metrics["erank_1"]),
                float(native_metrics["erank_1"]),
            ),
            "erank_2_relative_error": _relative_error(
                float(metrics["erank_2"]),
                float(native_metrics["erank_2"]),
            ),
        }
        if config.store_variant_spectra:
            row["spectrum_ref"] = registry.add(
                f"softmax_trunc_{label}_singular_values",
                approx_singular_values,
            )
        rows.append(row)
    return rows


def _extract_arrays(
    diagnostics: dict[str, Any],
    registry: ArrayRegistry,
    prefix: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in list(diagnostics):
        value = diagnostics[key]
        if key.startswith("_"):
            diagnostics.pop(key)
            output[f"{prefix}_{key[1:]}_ref"] = registry.add(f"{prefix}_{key[1:]}", value)
    return output


def analyze_layer_capture(
    attention: torch.nn.Module,
    capture: LayerCapture,
    layer: int,
    *,
    config: AnalysisConfig,
    metadata: Mapping[str, Any],
    heads: Sequence[int] | None = None,
    full_spectrum_heads: Sequence[int] | None = None,
    intensive_heads: Sequence[int] | None = None,
    attention_mask: torch.Tensor | None = None,
) -> AnalysisOutput:
    """Run WP2/WP3/WP5 diagnostics for one captured GPT-2 layer.

    ``heads`` controls which heads receive exact sink/score/QK diagnostics.
    ``full_spectrum_heads`` controls the exact native operator SVD used by WP1.
    ``intensive_heads`` independently controls additional dense SVDs for the
    sink nuclear residual, counterfactuals, truncations, and matched masks.
    It defaults to ``full_spectrum_heads`` for backward compatibility.  Heads
    outside both subsets still retain all cheap exact sink/score diagnostics.
    """

    if capture.query.shape[0] != 1:
        raise ValueError("Analysis rows use one document as the unit; batch size must be one")
    output = AnalysisOutput()
    analysis_device = torch.device(config.analysis_device)
    analysis_dtype = _torch_dtype(config.analysis_dtype)
    available_heads = capture.query.shape[1]
    selected_heads = list(range(available_heads)) if heads is None else [int(x) for x in heads]
    dense_spectrum_heads = (
        set(selected_heads)
        if full_spectrum_heads is None
        else {int(x) for x in full_spectrum_heads}
    )
    if not dense_spectrum_heads.issubset(set(selected_heads)):
        raise ValueError("full_spectrum_heads must be a subset of heads")
    intensive_diagnostic_heads = (
        set(dense_spectrum_heads)
        if intensive_heads is None
        else {int(x) for x in intensive_heads}
    )
    if not intensive_diagnostic_heads.issubset(dense_spectrum_heads):
        raise ValueError("intensive_heads must be a subset of full_spectrum_heads")
    length = int(capture.query.shape[-2])
    denominator = score_denominator(attention, layer)
    if attention_mask is not None and not bool(torch.as_tensor(attention_mask).bool().all()):
        raise ValueError(
            "WP2/WP3/WP5 diagnostics require one unpadded document prefix; "
            "padding masks are supported only by the WP0 reconstruction path",
        )

    for head in selected_heads:
        if not 0 <= head < available_heads:
            raise IndexError(f"Head {head} outside [0, {available_heads})")
        base = {
            **dict(metadata),
            "model": "gpt2",
            "length": length,
            "layer": int(layer),
            "head_or_channel": int(head),
            "head_dim": int(capture.query.shape[-1]),
            "score_denominator": denominator,
            "scale_attn_weights": bool(getattr(attention, "scale_attn_weights", True)),
            "scale_attn_by_inverse_layer_idx": bool(
                getattr(attention, "scale_attn_by_inverse_layer_idx", False),
            ),
            "analysis_dtype": str(analysis_dtype),
            "svd_dtype": str(_torch_dtype(config.svd_dtype)),
            "svd_device": str(config.svd_device),
            "supports_work_packages": "WP2,WP3,WP5",
        }
        prefix = "_".join(
            str(base.get(name, "na"))
            for name in (
                "corpus",
                "document_id",
                "sample_offset",
                "input_condition",
                "length",
                "layer",
                "head_or_channel",
            )
        )
        registry = ArrayRegistry(output, prefix=prefix, dtype=config.array_dtype)
        has_full_spectrum = head in dense_spectrum_heads
        has_intensive_diagnostics = head in intensive_diagnostic_heads
        query = capture.query[0, head].to(device=analysis_device, dtype=analysis_dtype)
        key = capture.key[0, head].to(device=analysis_device, dtype=analysis_dtype)
        scores = (query @ key.transpose(0, 1)) / denominator
        head_attention_mask = attention_mask
        if head_attention_mask is not None:
            head_attention_mask = head_attention_mask.to(analysis_device)
        native_attention = causal_softmax_from_scores(scores, attention_mask=head_attention_mask)

        if has_full_spectrum:
            native_row, native_singular_values = _rank_row(
                native_attention,
                config,
                registry,
                variant="native_softmax",
                base={**base, "work_package": "WP2"},
                store_spectrum=config.store_full_spectra,
            )
        else:
            native_singular_values = None
            native_row = {
                **base,
                "work_package": "WP2",
                "operator_variant": "native_softmax",
            }
        native_row["full_operator_spectrum"] = has_full_spectrum
        native_row["intensive_dense_diagnostics"] = has_intensive_diagnostics
        sink, residual_singular_values = sink_diagnostics(
            native_attention,
            native_singular_values=native_singular_values,
            config=(
                config
                if has_intensive_diagnostics
                else dataclasses.replace(config, compute_nuclear_sink_residual=False)
            ),
        )
        native_row.update(_extract_arrays(sink, registry, "sink"))
        native_row.update(sink)
        if residual_singular_values is not None and config.store_full_spectra:
            native_row["sink_residual_spectrum_ref"] = registry.add(
                "sink_residual_singular_values",
                residual_singular_values,
            )

        pre = pre_softmax_diagnostics(scores, config)
        native_row.update(_extract_arrays(pre, registry, "score"))
        native_row.update(pre)

        column_energy = native_attention.square().sum(dim=0)
        early_limit = min(16, length)
        early_column = int(column_energy[:early_limit].argmax())
        global_column = int(column_energy.argmax())
        native_row["confirmatory_first_column"] = _json(column_sink_summary(native_attention, 0))
        native_row["exploratory_dominant_early_column"] = _json(
            column_sink_summary(native_attention, early_column),
        )
        native_row["exploratory_dominant_global_column"] = _json(
            column_sink_summary(native_attention, global_column),
        )

        if config.compute_score_localization:
            localization = score_localization_diagnostics(query, key, scores, denominator)
            native_row.update(_extract_arrays(localization, registry, "score_localization"))
            native_row.update(localization)

        factor_svd = None
        if config.compute_qk_geometry or config.compute_truncations:
            qk, factor_svd = qk_geometry_diagnostics(
                query,
                key,
                denominator,
                config,
                registry,
            )
            native_row.update(qk)
        output.rows.append(native_row)

        if config.compute_counterfactuals and has_intensive_diagnostics:
            counterfactuals, intervention_metadata = score_counterfactuals(scores)
            for variant in ("neutralize_first", "late_spike"):
                row, singular_values = _rank_row(
                    counterfactuals[variant],
                    config,
                    registry,
                    variant=variant,
                    base={
                        **base,
                        "work_package": "WP2",
                        **intervention_metadata,
                    },
                    store_spectrum=config.store_variant_spectra,
                )
                intervention_sink, _ = sink_diagnostics(
                    counterfactuals[variant],
                    native_singular_values=singular_values,
                    config=dataclasses.replace(config, compute_nuclear_sink_residual=False),
                )
                intervention_sink.pop("_conditional_collision", None)
                row.update(intervention_sink)
                row["paired_native_srank_difference"] = row["srank"] - native_row["srank"]
                row["paired_native_erank_1_difference"] = row["erank_1"] - native_row["erank_1"]
                row["paired_native_erank_2_difference"] = row["erank_2"] - native_row["erank_2"]
                row["paired_native_best_rank1_error_difference"] = (
                    row["best_rank1_error"] - native_row["best_rank1_error"]
                )
                output.rows.append(row)

        if config.compute_truncations and has_intensive_diagnostics:
            if factor_svd is None:
                raise AssertionError("Truncation requires the cached Q/K SVD")
            output.rows.extend(
                truncation_rows(
                    query,
                    key,
                    scores,
                    native_attention,
                    native_singular_values,
                    factor_svd,
                    denominator,
                    config,
                    registry,
                    base=base,
                ),
            )

        if config.compute_matched_masks and has_intensive_diagnostics:
            matched = matched_mask_variants(scores, config.decay_gammas)
            for variant, matrix in matched.items():
                if variant == "native_softmax":
                    continue
                row, _ = _rank_row(
                    matrix,
                    config,
                    registry,
                    variant=variant,
                    base={
                        **base,
                        "work_package": "WP5",
                        "intervention_label": "conditional_operator_not_language_model_forward",
                    },
                    store_spectrum=config.store_variant_spectra,
                )
                if variant == "causal_raw_all_one":
                    gamma = 1.0
                else:
                    gamma = float(variant.rsplit("_", 1)[-1])
                row["mask_gamma"] = gamma
                row["mask_kappa"] = causal_mask_gain(length, gamma) if config.compute_mask_gain else math.nan
                row["paired_native_srank_difference"] = row["srank"] - native_row["srank"]
                row["paired_native_erank_1_difference"] = row["erank_1"] - native_row["erank_1"]
                row["paired_native_erank_2_difference"] = row["erank_2"] - native_row["erank_2"]
                row["paired_native_best_rank1_error_difference"] = (
                    row["best_rank1_error"] - native_row["best_rank1_error"]
                )
                output.rows.append(row)

        del query, key, scores, native_attention
        if analysis_device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def analyze_capture(
    model: torch.nn.Module,
    bundle: CaptureBundle,
    *,
    config: AnalysisConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
    heads_by_layer: Mapping[int, Sequence[int]] | None = None,
    full_spectrum_heads_by_layer: Mapping[int, Sequence[int]] | None = None,
    intensive_heads_by_layer: Mapping[int, Sequence[int]] | None = None,
) -> AnalysisOutput:
    """Analyze every captured layer while preserving document-level metadata."""

    config = config or AnalysisConfig()
    metadata = dict(metadata or {})
    output = AnalysisOutput()
    for layer in sorted(bundle.layers):
        heads = None if heads_by_layer is None else heads_by_layer.get(layer)
        full_spectrum_heads = (
            None
            if full_spectrum_heads_by_layer is None
            else full_spectrum_heads_by_layer.get(layer, ())
        )
        intensive_heads = (
            None
            if intensive_heads_by_layer is None
            else intensive_heads_by_layer.get(layer, ())
        )
        layer_output = analyze_layer_capture(
            model.transformer.h[layer].attn,
            bundle.layers[layer],
            layer,
            config=config,
            metadata=metadata,
            heads=heads,
            full_spectrum_heads=full_spectrum_heads,
            intensive_heads=intensive_heads,
            attention_mask=bundle.attention_mask,
        )
        output.extend(layer_output)
    return output


def save_analysis_output(output: AnalysisOutput, output_prefix: str | Path) -> tuple[Path, Path]:
    """Write scalar rows as JSONL and compact arrays as compressed NPZ."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows_path = prefix.with_suffix(".jsonl")
    arrays_path = prefix.with_suffix(".npz")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in output.rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
    np.savez_compressed(arrays_path, **output.arrays)
    return rows_path, arrays_path


def _parse_indices(text: str, total: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(total))
    output: list[int] = []
    for piece in text.split(","):
        value = int(piece.strip())
        if value < 0:
            value += total
        if not 0 <= value < total:
            raise ValueError(f"Index {piece!r} outside a collection of size {total}")
        output.append(value)
    return sorted(set(output))


def _smoke_base_ids(tokenizer, text: str, needed: int) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Smoke-test text tokenized to an empty sequence")
    repeated = list(ids)
    while len(repeated) < needed:
        repeated.extend(ids)
    return repeated[:needed]


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--lengths", default="16,32,64")
    parser.add_argument("--layers", default="0,-1")
    parser.add_argument("--heads", default="0")
    parser.add_argument("--conditions", default=",".join(INPUT_CONDITIONS))
    parser.add_argument(
        "--text",
        default=(
            "Spectral diagnostics should reconstruct the native operator before "
            "drawing conclusions about rank. "
        ),
    )
    parser.add_argument("--analysis-device", default="cpu")
    parser.add_argument("--svd-device", default="cpu")
    parser.add_argument("--svd-dtype", default="float32")
    parser.add_argument("--output", default="gpt2_smoke")
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Enable every truncation rank and exact mask gains; smoke mode otherwise uses a cheap subset.",
    )
    args = parser.parse_args()

    model, tokenizer = load_gpt2(
        args.checkpoint,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=True,
    )
    total_layers = len(model.transformer.h)
    total_heads = int(model.transformer.h[0].attn.num_heads)
    layers = _parse_indices(args.layers, total_layers)
    heads = _parse_indices(args.heads, total_heads)
    lengths = [int(x) for x in args.lengths.split(",")]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    for condition in conditions:
        if condition not in INPUT_CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r}")
    base_ids = _smoke_base_ids(tokenizer, args.text, max(lengths))
    revision = getattr(model.config, "_commit_hash", None) or "local_checkpoint_revision_unknown"

    combined = AnalysisOutput()
    reconstruction_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for length in lengths:
            input_ids = prepare_condition_input_ids(
                base_ids,
                length,
                condition,
                tokenizer.eos_token_id,
            )
            bundle = capture_all_layers(
                model,
                input_ids,
                layer_indices=layers,
                capture_native_core=True,
                offload_to_cpu=False,
            )
            metadata = {
                "checkpoint": args.checkpoint,
                "checkpoint_revision": revision,
                "corpus": "standalone_smoke_text",
                "document_id": "smoke",
                "input_condition": condition,
                "sample_offset": 0,
            }
            reconstruction_rows.extend(
                wp0_reconstruction_rows(
                    model,
                    bundle,
                    heads_by_layer={layer: heads for layer in layers},
                    metadata=metadata,
                ),
            )
            config = AnalysisConfig(
                analysis_device=args.analysis_device,
                svd_device=args.svd_device,
                svd_dtype=args.svd_dtype,
                compute_mask_gain=args.full_analysis,
                truncation_ranks=(1, 2, 4, 8, 16, 32) if args.full_analysis else (1, 2),
                alignment_ranks=(1, 2, 4, 8, 16) if args.full_analysis else (1, 2),
            )
            analysis = analyze_capture(
                model,
                bundle,
                config=config,
                metadata=metadata,
                heads_by_layer={layer: heads for layer in layers},
            )
            combined.extend(analysis)

    rows_path, arrays_path = save_analysis_output(combined, args.output)
    reconstruction_path = Path(args.output).with_name(Path(args.output).name + "_reconstruction.json")
    with reconstruction_path.open("w", encoding="utf-8") as handle:
        json.dump(reconstruction_rows, handle, indent=2, sort_keys=True, allow_nan=True)
    errors = [float(row["relative_error"]) for row in reconstruction_rows]
    summary = {
        "rows": len(combined.rows),
        "arrays": len(combined.arrays),
        "rows_path": str(rows_path),
        "arrays_path": str(arrays_path),
        "reconstruction_path": str(reconstruction_path),
        "wp0_max_relative_error": max(errors) if errors else math.nan,
        "wp0_median_relative_error": float(np.median(errors)) if errors else math.nan,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli()
