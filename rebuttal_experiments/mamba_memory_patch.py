"""Memory-bounded contractions for Transformers 4.57.1 Mamba2 torch_forward.

The upstream CPU/PyTorch fallback implements contractions as broadcast products
then sums, creating [sequence, heads, head_dim, state] and chunk-squared
intermediates. This narrowly replaces those contractions with equivalent
einsums, preserving convolution, masks, transitions, padding, gates, cache,
normalization, and projections. Floating-point reduction order can differ.

Usage before any forward pass (model loading can precede this call):
    from mamba_memory_patch import install_patch
    provenance = install_patch()

CLI (no pretrained checkpoint or download):
    python mamba_memory_patch.py --self-test --device cuda:0

install_patch() returns the original/new source hashes, installed source path,
helper hash, runtime version, and the exact replacements. Persist that object
in the experiment provenance. Use uninstall_patch() to restore the original.
Unexpected upstream source fails closed rather than partly patching a method.
"""

from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from typing import Any


EXPECTED_TRANSFORMERS_VERSION = "4.57.1"
_ORIGINAL = None
_PROVENANCE = None

# Statements are matched exactly after stripping leading/trailing whitespace.
# Empty replacements delete a now-unneeded broadcast temporary/reduction.
REPLACEMENTS = (
    (
        "G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)",
        'G = torch.einsum("bclhn,bcshn->bclsh", C, B)',
    ),
    ("G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)", ""),
    (
        "M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]",
        "M = G * L.permute(0, 2, 3, 4, 1)",
    ),
    ("M = M_intermediate.sum(dim=-1)", ""),
    (
        "Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)",
        'Y_diag = torch.einsum("bclsh,bcshp->bclhp", M, hidden_states)',
    ),
    (
        "states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)",
        'states = torch.einsum("bclhn,bclhp->bchpn", B_decay, hidden_states)',
    ),
    (
        "new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)",
        'new_states = torch.einsum("bsdh,bshpn->bdhpn", decay_chunk, states)',
    ),
    (
        "C_times_states = (C[..., None, :] * states[:, :, None, ...])",
        'C_times_states = torch.einsum("bclhn,bchpn->bclhp", C, states)',
    ),
    (
        "Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])",
        "Y_off = (C_times_states * state_decay_out_permuted[..., None])",
    ),
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def rewrite_source(source: str) -> str:
    """Rewrite the installed method only if every expected statement is unique."""
    source = textwrap.dedent(source)
    lines = source.splitlines(keepends=True)
    statements = [line.strip() for line in lines]
    for old, _new in REPLACEMENTS:
        count = statements.count(old)
        if count != 1:
            raise RuntimeError(f"Expected exactly one upstream statement; found {count}: {old}")
    mapping = dict(REPLACEMENTS)
    patched = []
    for line in lines:
        stripped = line.strip()
        if stripped not in mapping:
            patched.append(line)
            continue
        new = mapping[stripped]
        if new:
            indent = line[:len(line) - len(line.lstrip())]
            patched.append(indent + new + "\n")
    return "".join(patched)


def install_patch(*, require_version: str | None = EXPECTED_TRANSFORMERS_VERSION) -> dict[str, Any]:
    """Install process-local equivalent contractions and return full provenance."""
    global _ORIGINAL, _PROVENANCE
    import transformers
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

    if _PROVENANCE is not None:
        if Mamba2Mixer.torch_forward is not _PROVENANCE["_replacement"]:
            raise RuntimeError("Mamba2Mixer.torch_forward changed after patch installation")
        return {k: v for k, v in _PROVENANCE.items() if not k.startswith("_")}
    if require_version is not None and transformers.__version__ != require_version:
        raise RuntimeError(f"Patch audited for Transformers {require_version}, got {transformers.__version__}")
    original = Mamba2Mixer.torch_forward
    if original.__closure__:
        raise RuntimeError("Unexpected closure in upstream torch_forward")
    source = textwrap.dedent(inspect.getsource(original))
    rewritten = rewrite_source(source)
    namespace = dict(original.__globals__)
    filename = str(Path(__file__).resolve()) + "::<patched_Mamba2Mixer.torch_forward>"
    exec(compile(rewritten, filename, "exec"), namespace)
    replacement = namespace[original.__name__]
    replacement.__module__ = original.__module__
    replacement.__qualname__ = original.__qualname__
    replacement.__doc__ = original.__doc__
    provenance = {
        "patch": "bounded_contraction_einsums_v1", "transformers_version": transformers.__version__,
        "target": f"{original.__module__}.{original.__qualname__}",
        "upstream_source_file": inspect.getsourcefile(original),
        "upstream_method_sha256": sha256(source), "patched_method_sha256": sha256(rewritten),
        "patch_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "statement_replacements": [{"old": old, "new": new} for old, new in REPLACEMENTS],
        "scope": "all five broadcast-and-reduce contractions; no parameter or boundary changes",
        "numeric_note": "same mathematical contractions; floating-point reduction order differs",
    }
    Mamba2Mixer.torch_forward = replacement
    _ORIGINAL = original
    _PROVENANCE = {**provenance, "_replacement": replacement}
    return provenance


def uninstall_patch() -> None:
    global _ORIGINAL, _PROVENANCE
    if _ORIGINAL is None:
        return
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer
    if Mamba2Mixer.torch_forward is not _PROVENANCE["_replacement"]:
        raise RuntimeError("Refusing to overwrite a different torch_forward patch")
    Mamba2Mixer.torch_forward = _ORIGINAL
    _ORIGINAL = _PROVENANCE = None


def self_test(device: str = "cpu") -> dict[str, Any]:
    """Compare broadcast/einsum math and complete random native mixers, offline."""
    import torch
    from transformers import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

    torch.manual_seed(20261108)
    torch.backends.cuda.matmul.allow_tf32 = False
    reports = []

    def check(name, reference, changed, *, tolerance=2e-5):
        error = float(torch.linalg.vector_norm((changed - reference).double()) /
                      torch.linalg.vector_norm(reference.double()).clamp_min(1e-30))
        maximum = float((changed - reference).abs().max())
        reports.append({"name": name, "shape": list(reference.shape),
                        "relative_error": error, "max_abs_error": maximum})
        if not torch.isfinite(changed).all() or error > tolerance:
            raise AssertionError(f"Contraction mismatch: {reports[-1]}")

    with torch.inference_mode():
        # B=2, chunks=3, tokens/chunk=5, heads=4, state=7, head_dim=6.
        b, c, l, h, n, p = 2, 3, 5, 4, 7, 6
        B = torch.randn(b, c, l, h, n, device=device, dtype=torch.float64)
        C = torch.randn_like(B)
        X = torch.randn(b, c, l, h, p, device=device, dtype=torch.float64)
        G = (C[:, :, :, None] * B[:, :, None]).sum(-1)
        check("content_C_B", G, torch.einsum("bclhn,bcshn->bclsh", C, B), tolerance=1e-12)
        M = torch.randn(b, c, l, l, h, device=device, dtype=torch.float64)
        check("intra_chunk_output", (M[..., None] * X[:, :, None]).sum(3),
              torch.einsum("bclsh,bcshp->bclhp", M, X), tolerance=1e-12)
        states = (B[..., None, :] * X[..., None]).sum(2)
        check("intra_chunk_states", states, torch.einsum("bclhn,bclhp->bchpn", B, X), tolerance=1e-12)
        decay = torch.randn(b, c, c, h, device=device, dtype=torch.float64)
        check("inter_chunk_states", (decay[..., None, None] * states[:, :, None]).sum(1),
              torch.einsum("bsdh,bshpn->bdhpn", decay, states), tolerance=1e-12)
        check("inter_chunk_output", (C[..., None, :] * states[:, :, None]).sum(-1),
              torch.einsum("bclhn,bchpn->bclhp", C, states), tolerance=1e-12)

        # Includes exact chunk boundary, padded tail, and several chunks.
        config = Mamba2Config(hidden_size=16, state_size=4, num_hidden_layers=1,
            expand=2, num_heads=4, head_dim=8, n_groups=1, chunk_size=8,
            conv_kernel=4, vocab_size=31, use_cache=False)
        mixer = Mamba2Mixer(config, layer_idx=0).to(device=device, dtype=torch.float32).eval()
        if _ORIGINAL is not None:
            raise RuntimeError("Run self_test before installing this patch")
        original = Mamba2Mixer.torch_forward
        inputs = {length: torch.randn(2, length, 16, device=device) for length in (1, 8, 13, 23)}
        baseline = {length: original(mixer, hidden, cache_params=None) for length, hidden in inputs.items()}
        provenance = install_patch()
        try:
            for length, hidden in inputs.items():
                changed = mixer.torch_forward(hidden, cache_params=None)
                check(f"native_mixer_length_{length}", baseline[length], changed)
            # Padding attention-mask path uses the same unchanged causal convolution.
            hidden = inputs[13]
            mask = torch.ones(hidden.shape[:2], device=device, dtype=torch.long)
            mask[0, -3:] = 0
            reference = original(mixer, hidden, cache_params=None, attention_mask=mask)
            changed = mixer.torch_forward(hidden, cache_params=None, attention_mask=mask)
            check("native_mixer_padding_mask", reference, changed)
        finally:
            uninstall_patch()
        if Mamba2Mixer.torch_forward is not original:
            raise AssertionError("Patch restore failed")
    return {"status": "passed", "device": device, "comparisons": reports, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rewrite-check", type=Path, help="Validate exact source matches without torch")
    args = parser.parse_args()
    if args.rewrite_check:
        original = args.rewrite_check.read_text()
        changed = rewrite_source(original)
        compile(changed, "<rewrite-check>", "exec")
        print(json.dumps({"status": "source_matches", "original_sha256": sha256(textwrap.dedent(original)),
            "patched_sha256": sha256(changed), "replacements": len(REPLACEMENTS)}, sort_keys=True))
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
    if not args.self_test and not args.rewrite_check:
        parser.error("Use --self-test or --rewrite-check; runtime callers use install_patch()")


if __name__ == "__main__":
    main()
