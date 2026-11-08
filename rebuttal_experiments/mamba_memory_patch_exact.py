"""Conservative head-chunked Mamba2 fallback for Transformers 4.57.1.

Each expensive broadcast-and-sum is evaluated verbatim on slices of a
non-reduced head axis. There is no einsum or matrix-multiplication rewrite.
The input dtype, multiplication expression, and sum dimension are preserved.
All other native torch_forward statements, including the M product/reduction,
remain unchanged. With 8-head chunks, the largest 2048-token, chunk64,
head_dim64, state128 temporary is about 512 MiB rather than 8 GiB.

API: provenance = install_patch(head_chunk_size=8)
CLI: python mamba_memory_patch_exact.py --self-test --device cuda:0
Restore the process-local method with uninstall_patch(). Never modifies the
installed Transformers source or model parameters. Persist returned provenance.
"""

from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from typing import Any


EXPECTED_VERSION = "4.57.1"
EXPECTED_METHOD_SHA256 = "e08210f7d7cf12b70d9a7014fbf54c1af19dbd8e39095c229f9f9e3467b38790"
_ORIGINAL = None
_REPLACEMENT = None
_PROVENANCE = None

REPLACEMENTS = (
    (
        "G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)",
        'G = _mamba_chunked_contract("cb", C, B)',
    ),
    ("G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)", ""),
    (
        "Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)",
        'Y_diag = _mamba_chunked_contract("diag", M, hidden_states)',
    ),
    (
        "states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)",
        'states = _mamba_chunked_contract("states", B_decay, hidden_states)',
    ),
    (
        "new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)",
        'new_states = _mamba_chunked_contract("recurrence", decay_chunk, states)',
    ),
    (
        "C_times_states = (C[..., None, :] * states[:, :, None, ...])",
        'C_times_states = _mamba_chunked_contract("output", C, states)',
    ),
    (
        "Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])",
        "Y_off = (C_times_states * state_decay_out_permuted[..., None])",
    ),
)


def canonical_source(source: str) -> str:
    """inspect.getsource and print(getsource) differ only in trailing newlines."""
    return textwrap.dedent(source).rstrip() + "\n"


def sha256(source: str) -> str:
    return hashlib.sha256(canonical_source(source).encode()).hexdigest()


def chunked_contract(kind: str, x, y, *, head_chunk_size: int = 8):
    """Original elementwise products and torch.sum, chunked only over heads."""
    import torch
    axes = {"cb": (3, 3, 4), "diag": (4, 3, 3), "states": (3, 3, 2),
            "recurrence": (3, 2, 2), "output": (3, 2, 3)}
    if kind not in axes or head_chunk_size < 1:
        raise ValueError(f"Invalid contraction/chunk size: {kind}/{head_chunk_size}")
    x_axis, y_axis, out_axis = axes[kind]
    heads = x.shape[x_axis]
    if y.shape[y_axis] != heads or heads < 1:
        raise ValueError(f"Mismatched head axes: {tuple(x.shape)}, {tuple(y.shape)}")
    pieces = []
    for start in range(0, heads, head_chunk_size):
        stop = min(start + head_chunk_size, heads)
        x_index = [slice(None)] * x.ndim
        y_index = [slice(None)] * y.ndim
        x_index[x_axis] = y_index[y_axis] = slice(start, stop)
        xs, ys = x[tuple(x_index)], y[tuple(y_index)]
        if kind == "cb":
            result = (xs[:, :, :, None, :, :] * ys[:, :, None, :, :, :]).sum(dim=-1)
        elif kind == "diag":
            result = (xs[..., None] * ys[:, :, None]).sum(dim=3)
        elif kind == "states":
            result = (xs[..., None, :] * ys[..., None]).sum(dim=2)
        elif kind == "recurrence":
            result = (xs[..., None, None] * ys[:, :, None, ...]).sum(dim=1)
        else:
            result = (xs[..., None, :] * ys[:, :, None, ...]).sum(-1)
        pieces.append(result)
    return torch.cat(pieces, dim=out_axis)


def rewrite_source(source: str) -> str:
    source = canonical_source(source)
    if sha256(source) != EXPECTED_METHOD_SHA256:
        raise RuntimeError(f"Unexpected upstream method hash: {sha256(source)}")
    lines = source.splitlines(keepends=True)
    stripped = [line.strip() for line in lines]
    mapping = dict(REPLACEMENTS)
    for old in mapping:
        if stripped.count(old) != 1:
            raise RuntimeError(f"Expected one exact upstream statement: {old}")
    changed = []
    for line in lines:
        statement = line.strip()
        if statement not in mapping:
            changed.append(line)
        elif mapping[statement]:
            indent = line[:len(line) - len(line.lstrip())]
            changed.append(indent + mapping[statement] + "\n")
    return "".join(changed)


def install_patch(*, head_chunk_size: int = 8) -> dict[str, Any]:
    global _ORIGINAL, _REPLACEMENT, _PROVENANCE
    import transformers
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer
    if head_chunk_size < 1:
        raise ValueError("head_chunk_size must be positive")
    if _PROVENANCE is not None:
        if Mamba2Mixer.torch_forward is not _REPLACEMENT or head_chunk_size != _PROVENANCE["head_chunk_size"]:
            raise RuntimeError("Conflicting process-local Mamba patch")
        return dict(_PROVENANCE)
    if transformers.__version__ != EXPECTED_VERSION:
        raise RuntimeError(f"Audited Transformers version {EXPECTED_VERSION}; installed {transformers.__version__}")
    original = Mamba2Mixer.torch_forward
    if original.__closure__:
        raise RuntimeError("Unexpected upstream method closure")
    source = canonical_source(inspect.getsource(original))
    rewritten = rewrite_source(source)
    namespace = dict(original.__globals__)
    namespace["_mamba_chunked_contract"] = functools.partial(chunked_contract, head_chunk_size=head_chunk_size)
    exec(compile(rewritten, str(Path(__file__).resolve()) + "::<head_chunked_torch_forward>", "exec"), namespace)
    replacement = namespace[original.__name__]
    replacement.__module__, replacement.__qualname__, replacement.__doc__ = original.__module__, original.__qualname__, original.__doc__
    provenance = {
        "patch": "head_chunked_native_broadcast_sum_v2", "head_chunk_size": head_chunk_size,
        "transformers_version": transformers.__version__, "target": f"{original.__module__}.{original.__qualname__}",
        "upstream_source_file": inspect.getsourcefile(original),
        "upstream_method_sha256": sha256(source), "patched_method_sha256": sha256(rewritten),
        "patch_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "statement_replacements": [{"old": old, "new": new} for old, new in REPLACEMENTS],
        "scope": "non-reduced head slices; original broadcast products, input dtypes, sum dimensions",
        "numeric_note": "No einsum/matmul substitutions; numerical identity is tested against native fallback",
    }
    _ORIGINAL, _REPLACEMENT, _PROVENANCE = original, replacement, provenance
    Mamba2Mixer.torch_forward = replacement
    return dict(provenance)


def uninstall_patch() -> None:
    global _ORIGINAL, _REPLACEMENT, _PROVENANCE
    if _ORIGINAL is None:
        return
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer
    if Mamba2Mixer.torch_forward is not _REPLACEMENT:
        raise RuntimeError("Refusing to overwrite another process-local patch")
    Mamba2Mixer.torch_forward = _ORIGINAL
    _ORIGINAL = _REPLACEMENT = _PROVENANCE = None


def self_test(device: str = "cpu", head_chunk_size: int = 8) -> dict[str, Any]:
    import torch
    from transformers import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer
    torch.manual_seed(20261108)
    torch.backends.cuda.matmul.allow_tf32 = False
    comparisons = []

    def check(name, original, changed, tolerance=1e-6):
        error = float(torch.linalg.vector_norm((changed - original).double()) /
                      torch.linalg.vector_norm(original.double()).clamp_min(1e-30))
        comparison = dict(name=name, shape=list(original.shape), dtype=str(original.dtype),
            relative_error=error, max_abs_error=float((changed - original).abs().max()),
            bitwise_equal=bool(torch.equal(original, changed)))
        comparisons.append(comparison)
        if not bool(torch.isfinite(changed).all()) or error > tolerance:
            raise AssertionError(f"Conservative contraction check failed: {comparison}")

    with torch.inference_mode():
        # Several and partial head chunks, across both FP32 and FP64 reductions.
        b, c, l, h, n, p = 2, 3, 7, 19, 11, 13
        for dtype in (torch.float32, torch.float64):
            B = torch.randn(b, c, l, h, n, device=device, dtype=dtype)
            C = torch.randn_like(B)
            X = torch.randn(b, c, l, h, p, device=device, dtype=dtype)
            M = torch.randn(b, c, l, l, h, device=device, dtype=dtype)
            states = (B[..., None, :] * X[..., None]).sum(2)
            decay = torch.randn(b, c, c, h, device=device, dtype=dtype)
            references = {
                "cb": ((C[:, :, :, None] * B[:, :, None]).sum(-1), C, B),
                "diag": ((M[..., None] * X[:, :, None]).sum(3), M, X),
                "states": (states, B, X),
                "recurrence": ((decay[..., None, None] * states[:, :, None]).sum(1), decay, states),
                "output": ((C[..., None, :] * states[:, :, None]).sum(-1), C, states),
            }
            for kind, (reference, x, y) in references.items():
                for chunk in (1, 3, head_chunk_size, h):
                    check(f"{kind}_{dtype}_head_chunk_{chunk}", reference,
                          chunked_contract(kind, x, y, head_chunk_size=chunk),
                          tolerance=1e-12 if dtype == torch.float64 else 1e-6)

        config = Mamba2Config(hidden_size=48, state_size=4, num_hidden_layers=1, expand=2,
            num_heads=12, head_dim=8, n_groups=1, chunk_size=8, conv_kernel=4, vocab_size=31, use_cache=False)
        if _ORIGINAL is not None:
            raise RuntimeError("Run self-test before patch installation")
        original_method = Mamba2Mixer.torch_forward
        mixers = {str(dtype): Mamba2Mixer(config, layer_idx=0).to(device=device, dtype=dtype).eval()
                  for dtype in (torch.float32, torch.bfloat16)}
        inputs = {name: {length: torch.randn(2, length, 48, device=device, dtype=next(mixer.parameters()).dtype)
                         for length in (1, 8, 13, 23)} for name, mixer in mixers.items()}
        baselines = {(name, length): original_method(mixer, hidden, cache_params=None)
                     for name, mixer in mixers.items() for length, hidden in inputs[name].items()}
        provenance = install_patch(head_chunk_size=head_chunk_size)
        try:
            for name, mixer in mixers.items():
                for length, hidden in inputs[name].items():
                    check(f"native_mixer_{name}_length_{length}", baselines[name, length],
                          mixer.torch_forward(hidden, cache_params=None), tolerance=1e-6)
                hidden = inputs[name][13]
                mask = torch.ones(hidden.shape[:2], device=device, dtype=torch.long)
                mask[0, -3:] = 0
                check(f"native_mixer_padding_mask_{name}",
                      original_method(mixer, hidden, cache_params=None, attention_mask=mask),
                      mixer.torch_forward(hidden, cache_params=None, attention_mask=mask), tolerance=1e-6)
        finally:
            uninstall_patch()
    return dict(status="passed", device=device, head_chunk_size=head_chunk_size,
                comparisons=comparisons, provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--head-chunk-size", type=int, default=8)
    parser.add_argument("--rewrite-check", type=Path)
    args = parser.parse_args()
    if args.rewrite_check:
        source = args.rewrite_check.read_text()
        changed = rewrite_source(source)
        compile(changed, "<mamba_exact_rewrite_check>", "exec")
        print(json.dumps(dict(status="source_matches", upstream_method_sha256=sha256(textwrap.dedent(source)),
                              patched_method_sha256=sha256(changed), replacements=len(REPLACEMENTS)), sort_keys=True))
    if args.self_test:
        print(json.dumps(self_test(args.device, args.head_chunk_size), sort_keys=True))
    if not args.rewrite_check and not args.self_test:
        parser.error("Use --self-test or --rewrite-check, or import install_patch()")


if __name__ == "__main__":
    main()
