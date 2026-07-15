"""Compare token-mixing rank proxies across trained sequence models.

The matrices follow the definitions in ``article/draft.tex``:

* Mamba-2: ``L * (Q K^T)`` from the SSD representation.
* RetNet: ``gamma_h ** (i-j) * (Q K^T)`` on the causal triangle.
* Based: the causal Taylor-feature score matrix, both before and after the
  row normalization used by the trained mixer.
* GPT-2: the causal row-softmax attention matrix.

For every matrix, the script reports stable rank, entropy effective rank of
the singular values (erank_1), and entropy effective rank of the squared
singular values (erank_2). Results are appended incrementally and can be
resumed safely.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import random
import re
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_mamba_matrices import _input_device, extract_mamba2_matrices


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MAMBA_PATH = "/225040471/model/Mamba-Codestral-7B-v0.1"
DEFAULT_RETNET_PATH = "/225040471/model/retnet-1.3B-100B"
DEFAULT_BASED_PATH = "/225040471/model/based-360m"
DEFAULT_GPT2_PATH = "/225040471/model/gpt2"
DEFAULT_DATASET_ROOT = "/225040471/dataset/wikitext"

SERIES_LABELS = {
    "mamba2_m": "Mamba-2 (SSD)",
    "retnet_decay": "RetNet (decay)",
    "based_raw": "Based (raw causal linear)",
    "based_normalized": "Based (row-normalized)",
    "gpt2_softmax": "GPT-2 (causal softmax)",
}

SERIES_COLORS = {
    "mamba2_m": "#33658a",
    "retnet_decay": "#2a9d8f",
    "based_raw": "#e09f3e",
    "based_normalized": "#c44536",
    "gpt2_softmax": "#6a4c93",
}

SERIES_COMPACT_LABELS = {
    "mamba2_m": "empirical median + IQR",
    "retnet_decay": "empirical median + IQR",
    "based_raw": "raw median + IQR",
    "based_normalized": "row-normalized median + IQR",
    "gpt2_softmax": "empirical median + IQR",
}

MODEL_LABELS = {
    "mamba2": "Mamba-2 (SSD)",
    "retnet": "RetNet (decay)",
    "based": "Based",
    "gpt2": "GPT-2 (causal softmax)",
}

MODEL_ORDER = ["mamba2", "retnet", "based", "gpt2"]

# These are the sequence-length rates proved in article/draft.tex.  The
# theorem constants depend on the mask/model parameters and are not known for
# the trained checkpoints, so the plotted reference curves use unit
# prefactors.  The Based result is for the raw causal-linear operator, not its
# row-normalized variant.
THEORY_RATE_LABELS = {
    "mamba2": {
        "stable_rank": r"theory rate: $\ell/(\log \ell)^2$",
        "erank_1": r"theory rate: $\ell/\log \ell$",
        "erank_2": r"theory rate: $\ell/(\log \ell)^2$",
    },
    "retnet": {
        "stable_rank": r"theory rate: $\ell/(\log \ell)^2$",
        "erank_1": r"theory rate: $\ell/\log \ell$",
        "erank_2": r"theory rate: $\ell/(\log \ell)^2$",
    },
    "based": {
        "stable_rank": r"raw theory rate: $1$",
        "erank_1": r"raw theory rate: $\log \ell$",
        "erank_2": r"raw theory rate: $1$",
    },
}

METRICS = [
    ("stable_rank", "stable rank"),
    ("erank_1", r"$\mathrm{erank}_1$"),
    ("erank_2", r"$\mathrm{erank}_2$"),
]

RESULT_KEY_COLUMNS = [
    "series",
    "layer",
    "window_id",
    "requested_length",
    "head",
]


class _StopAfterCapture(Exception):
    pass


@dataclass(frozen=True)
class TextWindow:
    window_id: str
    requested_length: int
    window_index: int
    reference_start_token: int
    text: str


def parse_csv_ints(value: str) -> list[int]:
    parsed = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not parsed or parsed[0] <= 0:
        raise ValueError("Expected a comma-separated list of positive integers.")
    return parsed


def parse_models(value: str) -> list[str]:
    allowed = {"mamba2", "retnet", "based", "gpt2"}
    models = []
    for part in value.split(","):
        model = part.strip().lower()
        if model and model not in models:
            models.append(model)
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose from {sorted(allowed)}.")
    if not models:
        raise ValueError("--models must select at least one model.")
    return models


def clean_wikitext(text: str) -> str:
    text = (text or "").strip()
    text = text.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text.strip()


def is_heading(text: str) -> bool:
    return bool(re.fullmatch(r"=+\s*.*?\s*=+", text.strip()))


def load_wikitext_stream(dataset_root: str, subset: str, split: str) -> str:
    pattern = os.path.join(dataset_root, subset, f"{split}-*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No WikiText parquet files matched {pattern}.")

    texts: list[str] = []
    for path in paths:
        table = pq.read_table(path, columns=["text"])
        for value in table.column("text").to_pylist():
            text = clean_wikitext(value)
            if text and not is_heading(text):
                texts.append(text)
    if not texts:
        raise RuntimeError("The selected WikiText split contains no usable text.")
    return "\n\n".join(texts)


def build_shared_windows(
    full_text: str,
    reference_tokenizer,
    lengths: list[int],
    windows_per_length: int,
    seed: int,
    text_oversample: float,
) -> list[TextWindow]:
    original_model_max_length = reference_tokenizer.model_max_length
    reference_tokenizer.model_max_length = max(original_model_max_length, 10**9)
    try:
        reference_ids = reference_tokenizer(full_text, add_special_tokens=False)["input_ids"]
    finally:
        reference_tokenizer.model_max_length = original_model_max_length
    longest_span = max(int(math.ceil(max(lengths) * text_oversample)), max(lengths) + 32)
    if len(reference_ids) < longest_span:
        raise RuntimeError(
            f"WikiText stream has {len(reference_ids)} reference tokens, fewer than the required {longest_span}.",
        )

    rng = random.Random(seed)
    windows: list[TextWindow] = []
    for length in lengths:
        span = max(int(math.ceil(length * text_oversample)), length + 32)
        max_start = len(reference_ids) - span
        starts = [0]
        starts.extend(rng.randrange(0, max_start + 1) for _ in range(windows_per_length - 1))
        for window_index, start in enumerate(starts):
            text = reference_tokenizer.decode(
                reference_ids[start : start + span],
                skip_special_tokens=True,
            )
            windows.append(
                TextWindow(
                    window_id=f"L{length:05d}_W{window_index:02d}",
                    requested_length=length,
                    window_index=window_index,
                    reference_start_token=start,
                    text=text,
                ),
            )
    return windows


def save_windows(windows: list[TextWindow], path: Path) -> None:
    pd.DataFrame(
        [
            {
                "window_id": window.window_id,
                "requested_length": window.requested_length,
                "window_index": window.window_index,
                "reference_start_token": window.reference_start_token,
                "text": window.text,
            }
            for window in windows
        ],
    ).to_csv(path, index=False)


def evenly_spaced_heads(num_heads: int, count: int) -> list[int]:
    if count <= 0 or count >= num_heads:
        return list(range(num_heads))
    return sorted({int(index * num_heads / count) for index in range(count)})


def encode_window(tokenizer, window: TextWindow) -> tuple[dict[str, torch.Tensor], int]:
    encoded = tokenizer(
        window.text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=window.requested_length,
    )
    used_length = int(encoded["input_ids"].shape[1])
    if used_length < window.requested_length:
        warnings.warn(
            f"{window.window_id} produced only {used_length} tokens instead of "
            f"{window.requested_length}; increase --text-oversample.",
        )
    return encoded, used_length


def capture_module_input(
    model: torch.nn.Module,
    module: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    inputs = {key: value.to(_input_device(model)) for key, value in encoded.items()}
    capture: dict[str, torch.Tensor | None] = {}

    def hook(_module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError(f"{_module.__class__.__name__} received no hidden_states input.")
        capture["hidden_states"] = hidden_states.detach()
        capture["attention_mask"] = kwargs.get("attention_mask")
        raise _StopAfterCapture

    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            try:
                model(**inputs, use_cache=False)
            except _StopAfterCapture:
                pass
    finally:
        handle.remove()

    if "hidden_states" not in capture:
        raise RuntimeError(f"Failed to capture input to {module.__class__.__name__}.")
    return capture["hidden_states"], capture["attention_mask"]


def rank_rows_from_matrix(
    matrix: torch.Tensor,
    heads: list[int],
    rank_device: torch.device,
) -> list[dict[str, float | int]]:
    if matrix.ndim != 3 or matrix.shape[0] != len(heads) or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"Expected matrix [selected_heads, seq, seq], got {tuple(matrix.shape)} for heads {heads}.",
        )

    work = matrix.detach().float().to(rank_device)
    try:
        singular_values = torch.linalg.svdvals(work)
    except RuntimeError as error:
        if rank_device.type != "cuda":
            raise
        warnings.warn(f"CUDA SVD failed ({error}); retrying this batch on CPU.")
        singular_values = torch.linalg.svdvals(work.cpu())

    singular_values = singular_values.double().cpu()
    rows: list[dict[str, float | int]] = []
    for local_index, head in enumerate(heads):
        sv = singular_values[local_index]
        squared = sv.square()
        sum_sv = sv.sum()
        sum_squared = squared.sum()
        max_sv = sv.max()

        if float(sum_sv) == 0.0:
            stable_rank = erank_1 = erank_2 = 0.0
        else:
            p1 = sv / sum_sv
            p2 = squared / sum_squared
            entropy_1 = -(p1[p1 > 0] * p1[p1 > 0].log()).sum()
            entropy_2 = -(p2[p2 > 0] * p2[p2 > 0].log()).sum()
            stable_rank = float(sum_squared / max_sv.square())
            erank_1 = float(entropy_1.exp())
            erank_2 = float(entropy_2.exp())

        rows.append(
            {
                "head": head,
                "stable_rank": stable_rank,
                "erank_1": erank_1,
                "erank_2": erank_2,
                "sv_max": float(max_sv),
                "sv_fro": float(sum_squared.sqrt()),
                "sv_nuclear": float(sum_sv),
            },
        )
    del work, singular_values
    return rows


def load_done_keys(csv_path: Path) -> set[tuple]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(csv_path, usecols=RESULT_KEY_COLUMNS)
    return set(frame.itertuples(index=False, name=None))


def append_result_rows(csv_path: Path, rows: list[dict], done_keys: set[tuple]) -> int:
    fresh_rows = []
    for row in rows:
        key = tuple(row[column] for column in RESULT_KEY_COLUMNS)
        if key not in done_keys:
            fresh_rows.append(row)
            done_keys.add(key)
    if not fresh_rows:
        return 0
    pd.DataFrame(fresh_rows).to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists() or csv_path.stat().st_size == 0,
        index=False,
    )
    return len(fresh_rows)


def completed(
    done_keys: set[tuple],
    series: Iterable[str],
    layer: int,
    window: TextWindow,
    heads: Iterable[int],
) -> bool:
    return all(
        (name, layer, window.window_id, window.requested_length, head) in done_keys
        for name in series
        for head in heads
    )


def add_metadata(
    rank_rows: list[dict],
    *,
    model: str,
    series: str,
    model_path: str,
    matrix_definition: str,
    layer: int,
    window: TextWindow,
    used_token_len: int,
) -> list[dict]:
    output = []
    for rank_row in rank_rows:
        output.append(
            {
                "model": model,
                "series": series,
                "model_path": model_path,
                "matrix_definition": matrix_definition,
                "layer": layer,
                "window_id": window.window_id,
                "window_index": window.window_index,
                "reference_start_token": window.reference_start_token,
                "requested_length": window.requested_length,
                "used_token_len": used_token_len,
                **rank_row,
            },
        )
    return output


def release_memory(*objects) -> None:
    for item in objects:
        del item
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BasedExtractor:
    """Pure-PyTorch forward to a trained Based global linear-attention layer.

    The released package is tied to legacy FlashAttention extensions. This
    reproduces its inference-time BaseConv, Taylor linear-attention,
    sliding-attention, RMSNorm, and SwiGLU operations using current PyTorch.
    """

    def __init__(self, model_path: str, device: torch.device, target_layer: int):
        config_path = Path(model_path) / "config.json"
        weight_path = Path(model_path) / "pytorch_model.bin"
        if not config_path.exists() or not weight_path.exists():
            raise FileNotFoundError(
                f"Based requires {config_path} and {weight_path}. Download hazyresearch/based-360m.",
            )
        with config_path.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.target_layer = int(target_layer)
        self.linear_layers = set(int(layer) for layer in self.config.get("alt_mixer_layers", []))
        self.sliding_layers = set(int(layer) for layer in self.config.get("alt_mixer_2_layers", []))
        if self.target_layer not in self.linear_layers:
            raise ValueError(
                f"Based layer {self.target_layer} is not a global linear-attention layer; "
                f"choose from {sorted(self.linear_layers)}.",
            )
        if not 0 <= self.target_layer < int(self.config["n_layer"]):
            raise ValueError(f"Based layer {self.target_layer} is outside the checkpoint.")

        state = torch.load(weight_path, map_location="cpu", weights_only=True, mmap=True)
        required_keys = self.required_keys()
        missing = [key for key in required_keys if key not in state]
        if missing:
            raise KeyError(f"Based checkpoint is missing required weights: {missing}")
        self.weights = {key: state[key].to(device) for key in required_keys}
        self.device = device
        self.eps = float(self.config.get("layer_norm_epsilon", 1e-5))
        self.num_heads = int(self.config["alt_mixer"]["num_heads"])
        self.feature_dim = int(self.config["alt_mixer"]["feature_dim"])
        self.hidden_size = int(self.config["n_embd"])
        self.head_dim = self.hidden_size // self.num_heads
        self.sliding_window = int(self.config["alt_mixer_2"]["window_size"]) // 2
        del state

    def layer_key(self, layer: int, suffix: str) -> str:
        return f"transformer.layers.{layer}.{suffix}"

    def mixer_type(self, layer: int) -> str:
        if layer in self.sliding_layers:
            return "sliding"
        if layer in self.linear_layers:
            return "linear"
        return "conv"

    def required_keys(self) -> list[str]:
        keys = ["transformer.embeddings.word_embeddings.weight"]
        for layer in range(self.target_layer):
            keys.extend(
                [
                    self.layer_key(layer, "norm1.weight"),
                    self.layer_key(layer, "norm2.weight"),
                    self.layer_key(layer, "mlp.fc1.weight"),
                    self.layer_key(layer, "mlp.fc2.weight"),
                ],
            )
            mixer_type = self.mixer_type(layer)
            if mixer_type == "conv":
                keys.extend(
                    [
                        self.layer_key(layer, "mixer.in_proj.weight"),
                        self.layer_key(layer, "mixer.in_proj.bias"),
                        self.layer_key(layer, "mixer.conv.conv.weight"),
                        self.layer_key(layer, "mixer.out_proj.weight"),
                        self.layer_key(layer, "mixer.out_proj.bias"),
                    ],
                )
            elif mixer_type == "linear":
                keys.extend(
                    [
                        self.layer_key(layer, "mixer.proj_q.weight"),
                        self.layer_key(layer, "mixer.proj_k.weight"),
                        self.layer_key(layer, "mixer.proj_v.weight"),
                        self.layer_key(layer, "mixer.out_proj.weight"),
                    ],
                )
            else:
                keys.extend(
                    [
                        self.layer_key(layer, "mixer.Wqkv.weight"),
                        self.layer_key(layer, "mixer.out_proj.weight"),
                    ],
                )
        keys.extend(
            [
                self.layer_key(self.target_layer, "norm1.weight"),
                self.layer_key(self.target_layer, "mixer.proj_q.weight"),
                self.layer_key(self.target_layer, "mixer.proj_k.weight"),
            ],
        )
        return keys

    def weight(self, suffix: str) -> torch.Tensor:
        return self.weights[suffix]

    def rms_norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized * weight.float()

    def taylor_exp(self, value: torch.Tensor) -> torch.Tensor:
        root_d = math.sqrt(self.feature_dim)
        quadratic = (value.unsqueeze(-1) * value.unsqueeze(-2)).flatten(start_dim=-2) / math.sqrt(2.0)
        return torch.cat(
            [torch.ones_like(value[..., :1]), value / math.sqrt(root_d), quadratic / root_d],
            dim=-1,
        )

    def conv_forward(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        projected = F.linear(
            hidden,
            self.weight(self.layer_key(layer, "mixer.in_proj.weight")),
            self.weight(self.layer_key(layer, "mixer.in_proj.bias")),
        )
        first, second = projected.chunk(2, dim=-1)
        conv_weight = self.weight(self.layer_key(layer, "mixer.conv.conv.weight"))
        convolved = F.conv1d(
            first.transpose(1, 2),
            conv_weight,
            bias=None,
            padding=conv_weight.shape[-1] - 1,
            groups=first.shape[-1],
        )[..., : first.shape[1]].transpose(1, 2)
        return F.linear(
            F.silu(convolved) * second,
            self.weight(self.layer_key(layer, "mixer.out_proj.weight")),
            self.weight(self.layer_key(layer, "mixer.out_proj.bias")),
        )

    def linear_attention_forward(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        query = F.linear(hidden, self.weight(self.layer_key(layer, "mixer.proj_q.weight")))
        key = F.linear(hidden, self.weight(self.layer_key(layer, "mixer.proj_k.weight")))
        value = F.linear(hidden, self.weight(self.layer_key(layer, "mixer.proj_v.weight")))
        query = query.view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_features = self.taylor_exp(query)
        key_features = self.taylor_exp(key)
        scores = torch.einsum("bhnd,bhmd->bhnm", query_features, key_features)
        scores = torch.tril(scores)
        mixed = torch.einsum("bhnm,bhme->bhne", scores, value)
        denominator = torch.einsum(
            "bhld,bhld->bhl",
            query_features,
            key_features.cumsum(dim=2),
        )
        mixed = mixed / (denominator[..., None] + 1e-12)
        mixed = mixed.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return F.linear(mixed, self.weight(self.layer_key(layer, "mixer.out_proj.weight")))

    def apply_rotary(self, value: torch.Tensor) -> torch.Tensor:
        seq_len = value.shape[1]
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.head_dim, 2, device=value.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        frequencies = torch.outer(
            torch.arange(seq_len, device=value.device, dtype=torch.float32),
            inv_freq,
        )
        cosine = torch.cat([frequencies.cos(), frequencies.cos()], dim=-1)[None, :, None]
        sine = torch.cat([frequencies.sin(), frequencies.sin()], dim=-1)[None, :, None]
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat([-second, first], dim=-1)
        return value * cosine.to(value.dtype) + rotated * sine.to(value.dtype)

    def sliding_attention_forward(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        qkv = F.linear(hidden, self.weight(self.layer_key(layer, "mixer.Wqkv.weight")))
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        attention_dtype = torch.bfloat16 if hidden.device.type == "cuda" else hidden.dtype
        query = self.apply_rotary(qkv[:, :, 0]).transpose(1, 2).to(attention_dtype)
        key = self.apply_rotary(qkv[:, :, 1]).transpose(1, 2).to(attention_dtype)
        value = qkv[:, :, 2].transpose(1, 2).to(attention_dtype)
        positions = torch.arange(seq_len, device=hidden.device)
        lag = positions[:, None] - positions[None, :]
        attention_mask = (lag >= 0) & (lag <= self.sliding_window)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask[None, None],
            dropout_p=0.0,
        )
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return F.linear(
            context.to(hidden.dtype),
            self.weight(self.layer_key(layer, "mixer.out_proj.weight")),
        )

    def mixer_forward(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        mixer_type = self.mixer_type(layer)
        if mixer_type == "conv":
            return self.conv_forward(hidden, layer)
        if mixer_type == "linear":
            return self.linear_attention_forward(hidden, layer)
        return self.sliding_attention_forward(hidden, layer)

    def mlp_forward(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        projected = F.linear(hidden, self.weight(self.layer_key(layer, "mlp.fc1.weight")))
        value, gate = projected.chunk(2, dim=-1)
        return F.linear(
            value * F.silu(gate),
            self.weight(self.layer_key(layer, "mlp.fc2.weight")),
        )

    def target_layer_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        hidden = F.embedding(input_ids, self.weight("transformer.embeddings.word_embeddings.weight"))
        residual = None
        for layer in range(self.target_layer):
            residual = hidden if residual is None else residual + hidden
            mixer_input = self.rms_norm(residual, self.weight(self.layer_key(layer, "norm1.weight")))
            hidden = self.mixer_forward(mixer_input, layer)
            residual = residual + hidden
            mlp_input = self.rms_norm(residual, self.weight(self.layer_key(layer, "norm2.weight")))
            hidden = self.mlp_forward(mlp_input, layer)
        residual = hidden if residual is None else residual + hidden
        return self.rms_norm(
            residual,
            self.weight(self.layer_key(self.target_layer, "norm1.weight")),
        )

    def matrices(self, input_ids: torch.Tensor, heads: list[int]) -> dict[str, torch.Tensor]:
        hidden = self.target_layer_input(input_ids)
        query = F.linear(
            hidden,
            self.weight(self.layer_key(self.target_layer, "mixer.proj_q.weight")),
        )
        key = F.linear(
            hidden,
            self.weight(self.layer_key(self.target_layer, "mixer.proj_k.weight")),
        )
        batch_size, seq_len, _ = query.shape
        query = query.view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(1, 2)
        head_index = torch.as_tensor(heads, device=self.device, dtype=torch.long)
        query = query.index_select(1, head_index).float()
        key = key.index_select(1, head_index).float()
        query_features = self.taylor_exp(query)
        key_features = self.taylor_exp(key)
        raw = torch.einsum("bhtd,bhsd->bhts", query_features, key_features)[0]
        causal = torch.tril(torch.ones(seq_len, seq_len, device=raw.device, dtype=torch.bool))
        raw = raw.masked_fill(~causal[None], 0.0)
        normalized = raw / (raw.sum(dim=-1, keepdim=True) + 1e-12)
        return {"based_raw": raw, "based_normalized": normalized}


def run_based(
    args,
    windows: list[TextWindow],
    csv_path: Path,
    done_keys: set[tuple],
    rank_device: torch.device,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_path, local_files_only=True)
    device = torch.device(args.based_device)
    layer = int(args.based_layer)
    extractor = BasedExtractor(args.based_path, device, target_layer=layer)
    heads = evenly_spaced_heads(extractor.num_heads, args.heads_per_model)
    print(f"Based layer {layer} heads={heads} device={device}")

    for index, window in enumerate(windows, 1):
        if completed(done_keys, ["based_raw", "based_normalized"], layer, window, heads):
            print(f"[Based {index:03d}/{len(windows):03d}] {window.window_id} already complete")
            continue
        encoded, used_length = encode_window(tokenizer, window)
        print(f"[Based {index:03d}/{len(windows):03d}] {window.window_id} T={used_length}")
        with torch.inference_mode():
            matrices = extractor.matrices(encoded["input_ids"], heads)
        for series, matrix in matrices.items():
            definition = (
                "tril(Phi(Q) Phi(K)^T)"
                if series == "based_raw"
                else "row_normalize(tril(Phi(Q) Phi(K)^T))"
            )
            rows = rank_rows_from_matrix(matrix, heads, rank_device)
            rows = add_metadata(
                rows,
                model="based",
                series=series,
                model_path=args.based_path,
                matrix_definition=definition,
                layer=layer,
                window=window,
                used_token_len=used_length,
            )
            append_result_rows(csv_path, rows, done_keys)
        del matrices, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    release_memory(extractor, tokenizer)


def run_retnet(
    args,
    windows: list[TextWindow],
    csv_path: Path,
    done_keys: set[tuple],
    rank_device: torch.device,
) -> None:
    import fla.models  # noqa: F401 -- registers RetNet with Transformers
    from fla.models import RetNetForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.retnet_path, local_files_only=True)
    model = RetNetForCausalLM.from_pretrained(
        args.retnet_path,
        dtype="auto",
        device_map=args.retnet_device,
        local_files_only=True,
    ).eval()
    layer = int(args.retnet_layer)
    retention = model.model.layers[layer].attn
    heads = evenly_spaced_heads(retention.num_heads, args.heads_per_model)
    print(f"RetNet layer {layer} heads={heads} device_map={args.retnet_device}")

    for index, window in enumerate(windows, 1):
        if completed(done_keys, ["retnet_decay"], layer, window, heads):
            print(f"[RetNet {index:03d}/{len(windows):03d}] {window.window_id} already complete")
            continue
        encoded, used_length = encode_window(tokenizer, window)
        print(f"[RetNet {index:03d}/{len(windows):03d}] {window.window_id} T={used_length}")
        hidden, _ = capture_module_input(model, retention, encoded)
        with torch.inference_mode():
            query = retention.q_proj(hidden)
            key = retention.k_proj(hidden)
            query = query.view(1, used_length, retention.num_heads, retention.head_k_dim)
            key = key.view(1, used_length, retention.num_kv_heads, retention.head_k_dim)
            query, key = retention.rotary(query, key, seqlen_offset=0, max_seqlen=used_length)
            if retention.num_kv_groups > 1:
                key = key.repeat_interleave(retention.num_kv_groups, dim=2)
            head_index = torch.as_tensor(heads, device=query.device, dtype=torch.long)
            query = query.index_select(2, head_index).transpose(1, 2).float()
            key = key.index_select(2, head_index).transpose(1, 2).float()
            scores = torch.einsum("bhtd,bhsd->bhts", query, key)[0] / math.sqrt(retention.head_k_dim)

            positions = torch.arange(used_length, device=scores.device)
            lags = positions[:, None] - positions[None, :]
            causal = lags >= 0
            head_tensor = torch.as_tensor(heads, device=scores.device, dtype=torch.float32)
            gamma = 1.0 - torch.pow(2.0, -5.0 - head_tensor)
            decay = torch.exp(gamma.log()[:, None, None] * lags.clamp_min(0)[None])
            matrix = scores * decay * causal[None]

        rows = rank_rows_from_matrix(matrix, heads, rank_device)
        rows = add_metadata(
            rows,
            model="retnet",
            series="retnet_decay",
            model_path=args.retnet_path,
            matrix_definition="tril(gamma_h^(i-j) * Q K^T / sqrt(d))",
            layer=layer,
            window=window,
            used_token_len=used_length,
        )
        append_result_rows(csv_path, rows, done_keys)
        del encoded, hidden, query, key, scores, decay, matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    release_memory(model, tokenizer)


def run_gpt2(
    args,
    windows: list[TextWindow],
    csv_path: Path,
    done_keys: set[tuple],
    rank_device: torch.device,
) -> None:
    weight_path = Path(args.gpt2_path) / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Missing {weight_path}. Download openai-community/gpt2 model.safetensors into that directory.",
        )
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.gpt2_path,
        dtype="auto",
        device_map=args.gpt2_device,
        local_files_only=True,
    ).eval()
    layer = int(args.gpt2_layer)
    attention = model.transformer.h[layer].attn
    heads = evenly_spaced_heads(attention.num_heads, args.heads_per_model)
    print(f"GPT-2 layer {layer} heads={heads} device_map={args.gpt2_device}")

    for index, window in enumerate(windows, 1):
        if completed(done_keys, ["gpt2_softmax"], layer, window, heads):
            print(f"[GPT-2 {index:03d}/{len(windows):03d}] {window.window_id} already complete")
            continue
        encoded, used_length = encode_window(tokenizer, window)
        print(f"[GPT-2 {index:03d}/{len(windows):03d}] {window.window_id} T={used_length}")
        hidden, _ = capture_module_input(model, attention, encoded)
        with torch.inference_mode():
            query, key, _ = attention.c_attn(hidden).split(attention.split_size, dim=2)
            query = query.view(1, used_length, attention.num_heads, attention.head_dim).transpose(1, 2)
            key = key.view(1, used_length, attention.num_heads, attention.head_dim).transpose(1, 2)
            head_index = torch.as_tensor(heads, device=query.device, dtype=torch.long)
            query = query.index_select(1, head_index).float()
            key = key.index_select(1, head_index).float()
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(attention.head_dim)
            if attention.scale_attn_by_inverse_layer_idx:
                scores = scores / float(layer + 1)
            causal = torch.tril(torch.ones(used_length, used_length, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
            matrix = torch.softmax(scores, dim=-1)[0]

        rows = rank_rows_from_matrix(matrix, heads, rank_device)
        rows = add_metadata(
            rows,
            model="gpt2",
            series="gpt2_softmax",
            model_path=args.gpt2_path,
            matrix_definition="softmax(tril(Q K^T / sqrt(d)))",
            layer=layer,
            window=window,
            used_token_len=used_length,
        )
        append_result_rows(csv_path, rows, done_keys)
        del encoded, hidden, query, key, scores, matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    release_memory(model, tokenizer)


def run_mamba2(
    args,
    windows: list[TextWindow],
    csv_path: Path,
    done_keys: set[tuple],
    rank_device: torch.device,
) -> None:
    # Import the Transformers implementation explicitly. Importing fla.models
    # for RetNet also registers FLA's Mamba-2 class globally, whose fused
    # normalization path is CUDA-only and cannot honor --mamba-device=cpu.
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.mamba_path, local_files_only=True)
    config = Mamba2Config.from_pretrained(args.mamba_path, local_files_only=True)
    model = Mamba2ForCausalLM.from_pretrained(
        args.mamba_path,
        config=config,
        dtype="auto",
        device_map=args.mamba_device,
        local_files_only=True,
    ).eval()
    if args.mamba_chunk_size > 0:
        for block in model.backbone.layers:
            block.mixer.chunk_size = int(args.mamba_chunk_size)
    layer = int(args.mamba_layer)
    mixer = model.backbone.layers[layer].mixer
    heads = evenly_spaced_heads(mixer.num_heads, args.heads_per_model)
    print(f"Mamba-2 layer {layer} heads={heads} device_map={args.mamba_device}")

    for index, window in enumerate(windows, 1):
        if completed(done_keys, ["mamba2_m"], layer, window, heads):
            print(f"[Mamba-2 {index:03d}/{len(windows):03d}] {window.window_id} already complete")
            continue
        encoded, used_length = encode_window(tokenizer, window)
        print(f"[Mamba-2 {index:03d}/{len(windows):03d}] {window.window_id} T={used_length}")
        hidden, attention_mask = capture_module_input(model, mixer, encoded)
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden.device)
        with torch.inference_mode():
            matrices = extract_mamba2_matrices(
                mixer,
                hidden,
                attention_mask=attention_mask,
                selected_heads=heads,
            )
            matrix = matrices.m[0]

        rows = rank_rows_from_matrix(matrix, heads, rank_device)
        rows = add_metadata(
            rows,
            model="mamba2",
            series="mamba2_m",
            model_path=args.mamba_path,
            matrix_definition="L * (C B^T)",
            layer=layer,
            window=window,
            used_token_len=used_length,
        )
        append_result_rows(csv_path, rows, done_keys)
        del encoded, hidden, matrices, matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    release_memory(model, tokenizer)


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    aggregate_columns = ["model", "series", "layer", "requested_length"]
    named_aggregations: dict[str, tuple[str, str | Callable]] = {
        "used_token_len_median": ("used_token_len", "median"),
        "n_head_windows": ("head", "size"),
        "n_heads": ("head", "nunique"),
        "n_windows": ("window_id", "nunique"),
    }
    for metric, _ in METRICS:
        named_aggregations[f"{metric}_median"] = (metric, "median")
        named_aggregations[f"{metric}_q10"] = (metric, lambda values: values.quantile(0.10))
        named_aggregations[f"{metric}_q25"] = (metric, lambda values: values.quantile(0.25))
        named_aggregations[f"{metric}_q75"] = (metric, lambda values: values.quantile(0.75))
        named_aggregations[f"{metric}_q90"] = (metric, lambda values: values.quantile(0.90))
    return (
        raw.groupby(aggregate_columns, as_index=False)
        .agg(**named_aggregations)
        .sort_values(["series", "requested_length"])
        .reset_index(drop=True)
    )


def estimate_asymptotic_slopes(summary: pd.DataFrame, min_length: int) -> pd.DataFrame:
    rows = []
    for series, series_frame in summary.groupby("series"):
        tail = series_frame[series_frame["requested_length"] >= min_length]
        if len(tail) < 2:
            continue
        x = np.log(tail["used_token_len_median"].to_numpy(dtype=float))
        for metric, _ in METRICS:
            y_values = tail[f"{metric}_median"].to_numpy(dtype=float)
            valid = np.isfinite(y_values) & (y_values > 0)
            if valid.sum() < 2:
                continue
            slope, intercept = np.polyfit(x[valid], np.log(y_values[valid]), deg=1)
            prediction = intercept + slope * x[valid]
            residual = np.log(y_values[valid]) - prediction
            total = np.log(y_values[valid]) - np.log(y_values[valid]).mean()
            r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2)) if np.sum(total**2) > 0 else 1.0
            rows.append(
                {
                    "series": series,
                    "label": SERIES_LABELS.get(series, series),
                    "metric": metric,
                    "min_requested_length": int(tail["requested_length"].min()),
                    "max_requested_length": int(tail["requested_length"].max()),
                    "n_lengths": int(valid.sum()),
                    "log_log_slope": float(slope),
                    "log_log_intercept": float(intercept),
                    "r_squared": r_squared,
                },
            )
    return pd.DataFrame(
        rows,
        columns=[
            "series",
            "label",
            "metric",
            "min_requested_length",
            "max_requested_length",
            "n_lengths",
            "log_log_slope",
            "log_log_intercept",
            "r_squared",
        ],
    )


def theoretical_rate(model: str, metric: str, length: np.ndarray) -> np.ndarray | None:
    """Return the theorem's length dependence with a unit prefactor."""
    if model in {"mamba2", "retnet"}:
        if metric == "erank_1":
            return length / np.log(length)
        if metric in {"stable_rank", "erank_2"}:
            return length / np.log(length) ** 2
    if model == "based":
        if metric == "erank_1":
            return np.log(length)
        if metric in {"stable_rank", "erank_2"}:
            return np.ones_like(length)
    return None


def plot_rank_axis(
    axis,
    model_frame: pd.DataFrame,
    model: str,
    metric: str,
    *,
    compact_labels: bool = False,
) -> None:
    """Draw empirical bands and the applicable theory rate on one axis."""
    for series in SERIES_LABELS:
        frame = model_frame[model_frame["series"] == series].sort_values("used_token_len_median")
        if frame.empty:
            continue
        x = frame["used_token_len_median"].to_numpy(dtype=float)
        median = frame[f"{metric}_median"].to_numpy(dtype=float)
        q10 = frame[f"{metric}_q10"].to_numpy(dtype=float)
        q25 = frame[f"{metric}_q25"].to_numpy(dtype=float)
        q75 = frame[f"{metric}_q75"].to_numpy(dtype=float)
        q90 = frame[f"{metric}_q90"].to_numpy(dtype=float)
        color = SERIES_COLORS[series]
        axis.fill_between(x, q10, q90, color=color, alpha=0.10, linewidth=0)
        axis.fill_between(x, q25, q75, color=color, alpha=0.22, linewidth=0)
        label = SERIES_COMPACT_LABELS[series] if compact_labels else f"{SERIES_LABELS[series]} median + IQR"
        axis.plot(
            x,
            median,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=label,
        )

    if model in THEORY_RATE_LABELS:
        x_min = float(model_frame["used_token_len_median"].min())
        x_max = float(model_frame["used_token_len_median"].max())
        theory_x = np.linspace(max(3.0, x_min), x_max, 256)
        theory_y = theoretical_rate(model, metric, theory_x)
        if theory_y is not None:
            axis.plot(
                theory_x,
                theory_y,
                linestyle="--",
                linewidth=1.4,
                color="#444444",
                label=THEORY_RATE_LABELS[model][metric],
            )

    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)
    legend_location = (
        "lower right"
        if model == "based" and metric in {"stable_rank", "erank_2"}
        else "best"
    )
    axis.legend(loc=legend_location, fontsize=8, frameon=False)


def plot_model_rank_panels(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write one three-panel median/IQR rank figure for every available model."""
    written: list[Path] = []
    available_models = set(summary["model"].unique())
    ordered_models = [model for model in MODEL_ORDER if model in available_models]
    ordered_models.extend(sorted(available_models - set(ordered_models)))

    for model in ordered_models:
        model_frame = summary[summary["model"] == model]
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharex=True)
        for axis, (metric, ylabel) in zip(axes, METRICS):
            plot_rank_axis(axis, model_frame, model, metric)
            axis.set_xlabel("sequence length (tokens)")
            axis.set_ylabel(ylabel)

        layers = ", ".join(str(int(layer)) for layer in sorted(model_frame["layer"].unique()))
        layer_label = f"layer {layers}" if ", " not in layers else f"layers {layers}"
        fig.suptitle(
            f"WikiText-103 validation — {MODEL_LABELS.get(model, model)}, {layer_label}\n"
            "median over head/window samples; dark band = IQR, light band = 10–90%",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        out_path = out_dir / f"{model}_rank_vs_length_iqr.svg"
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def plot_rank_grid(
    summary: pd.DataFrame,
    out_dir: Path,
    models: list[str],
    filename: str,
    title: str,
) -> Path | None:
    """Write a model-by-rank-proxy comparison figure."""
    if not set(models).issubset(set(summary["model"].unique())):
        return None

    row_count = len(models)
    fig, axes = plt.subplots(
        row_count,
        3,
        figsize=(16.0, 4.0 * row_count),
        sharex=True,
        squeeze=False,
    )
    for row_index, model in enumerate(models):
        model_frame = summary[summary["model"] == model]
        for column_index, (metric, ylabel) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            plot_rank_axis(axis, model_frame, model, metric, compact_labels=True)
            if row_index == 0:
                axis.set_title(ylabel, fontsize=12)

        layers = ", ".join(str(int(layer)) for layer in sorted(model_frame["layer"].unique()))
        layer_label = f"layer {layers}" if ", " not in layers else f"layers {layers}"
        model_label = MODEL_LABELS.get(model, model)
        if model == "based":
            based_layers = sorted(int(layer) for layer in model_frame["layer"].unique())
            configured_layers = [1, 6, 11, 16, 21]
            if based_layers == [configured_layers[0]]:
                layer_label += "\n(first linear-attention block)"
            elif based_layers == [configured_layers[-1]]:
                layer_label += "\n(last linear-attention block)"
            else:
                layer_label += "\n(global linear-attention block)"
        axes[row_index, 0].text(
            -0.24,
            0.5,
            f"{model_label}\n{layer_label}",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.supxlabel("sequence length (tokens)", fontsize=11)
    fig.suptitle(
        f"WikiText-103 validation — {title}\n"
        "median over head/window samples; dark band = IQR, light band = 10–90%",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.055, 0.035, 1, 0.94), h_pad=2.0, w_pad=1.8)
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_combined_rank_grids(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write the 3x3 structured-model grid and the 4x3 grid with GPT-2."""
    specifications = [
        (
            ["based", "mamba2", "retnet"],
            "based_mamba2_retnet_rank_vs_length_iqr_3x3.svg",
            "structured-model rank comparison",
        ),
        (
            ["based", "mamba2", "retnet", "gpt2"],
            "based_mamba2_retnet_gpt2_rank_vs_length_iqr_4x3.svg",
            "structured models and GPT-2 rank comparison",
        ),
    ]
    written = []
    for models, filename, title in specifications:
        path = plot_rank_grid(summary, out_dir, models, filename, title)
        if path is not None:
            written.append(path)
    return written


def write_aggregates(csv_path: Path, out_dir: Path, slope_min_length: int) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("No rank rows are available to aggregate.")
        return
    raw = pd.read_csv(csv_path)
    raw = raw.drop_duplicates(RESULT_KEY_COLUMNS, keep="last")
    raw = raw.sort_values(["series", "requested_length", "window_index", "head"]).reset_index(drop=True)
    raw.to_csv(csv_path, index=False)

    summary = aggregate_results(raw)
    summary_path = out_dir / "rank_summary.csv"
    summary.to_csv(summary_path, index=False)
    slopes = estimate_asymptotic_slopes(summary, slope_min_length)
    slopes.to_csv(out_dir / "asymptotic_loglog_slopes.csv", index=False)
    figure_paths = plot_model_rank_panels(summary, out_dir)
    grid_paths = plot_combined_rank_grids(summary, out_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {out_dir / 'asymptotic_loglog_slopes.csv'}")
    for figure_path in figure_paths:
        print(f"wrote {figure_path}")
    for grid_path in grid_paths:
        print(f"wrote {grid_path}")


def resolve_rank_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare srank, erank_1, and erank_2 of trained structured/softmax attention matrices.",
    )
    parser.add_argument("--models", default="based,retnet,gpt2,mamba2")
    parser.add_argument("--lengths", default="64,128,256,384,512,768,1024")
    parser.add_argument("--windows-per-length", type=int, default=2)
    parser.add_argument("--heads-per-model", type=int, default=4, help="Use 0 to evaluate every head.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-oversample", type=float, default=2.0)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subset", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--mamba-path", default=DEFAULT_MAMBA_PATH)
    parser.add_argument("--retnet-path", default=DEFAULT_RETNET_PATH)
    parser.add_argument("--based-path", default=DEFAULT_BASED_PATH)
    parser.add_argument("--gpt2-path", default=DEFAULT_GPT2_PATH)
    parser.add_argument("--mamba-layer", type=int, default=0)
    parser.add_argument("--retnet-layer", type=int, default=0)
    parser.add_argument(
        "--based-layer",
        type=int,
        default=1,
        help="Based global linear-attention layer (one of 1,6,11,16,21).",
    )
    parser.add_argument("--gpt2-layer", type=int, default=0)
    parser.add_argument("--mamba-device", default="cpu")
    parser.add_argument(
        "--mamba-chunk-size",
        type=int,
        default=0,
        help="Override the naive Mamba-2 SSD chunk size to reduce peak memory; 0 keeps the checkpoint default.",
    )
    parser.add_argument("--retnet-device", default="auto")
    parser.add_argument("--gpt2-device", default="auto")
    parser.add_argument("--based-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rank-device", default="auto")
    parser.add_argument("--slope-min-length", type=int, default=256)
    parser.add_argument("--out-dir", default="attention_rank_comparison_outputs")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue with other models when a checkpoint is absent or a model fails.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rank_results.csv"
    if args.plot_only:
        write_aggregates(csv_path, out_dir, args.slope_min_length)
        return

    models = parse_models(args.models)
    lengths = parse_csv_ints(args.lengths)
    if args.windows_per_length <= 0:
        raise ValueError("--windows-per-length must be positive.")
    if args.text_oversample <= 1.0:
        raise ValueError("--text-oversample must be greater than 1.")

    reference_tokenizer = AutoTokenizer.from_pretrained(args.gpt2_path, local_files_only=True)
    full_text = load_wikitext_stream(args.dataset_root, args.subset, args.split)
    windows = build_shared_windows(
        full_text,
        reference_tokenizer,
        lengths,
        args.windows_per_length,
        args.seed,
        args.text_oversample,
    )
    save_windows(windows, out_dir / "windows.csv")
    del full_text, reference_tokenizer

    rank_device = resolve_rank_device(args.rank_device)
    done_keys = load_done_keys(csv_path)
    print(
        f"prepared {len(windows)} shared WikiText windows; rank_device={rank_device}; "
        f"resuming from {len(done_keys)} rows",
    )

    runners = {
        "based": run_based,
        "retnet": run_retnet,
        "gpt2": run_gpt2,
        "mamba2": run_mamba2,
    }
    failures = []
    for model_name in models:
        print(f"\n=== {model_name} ===")
        try:
            runners[model_name](args, windows, csv_path, done_keys, rank_device)
        except Exception as error:
            failures.append((model_name, error))
            traceback.print_exc()
            release_memory()
            if not args.skip_missing:
                raise

    write_aggregates(csv_path, out_dir, args.slope_min_length)
    if failures:
        print("\nModels not completed:")
        for model_name, error in failures:
            print(f"- {model_name}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
