"""Versioned small-scale SVD correction; frozen v2 helper remains unchanged.

For FP32 matrices below sqrt(minimum-normal), normalize entries in FP64,
solve in FP32 at unit scale, and restore singular values in FP64. Entries
need not be subnormal: internal FP32 products can underflow much earlier.
This corrects the solver's scale, not rounding in native matrix formation.
"""
from __future__ import annotations

import math

import spectral_qc as frozen_v2

POLICY = "small_scale_fp32_maxnorm_v1"


def scaling_info(matrix):
    import torch
    # FP64 reduction is deliberate: preserve tiny, including subnormal, FP32 entries.
    maximum = float(matrix.detach().double().abs().max()) if matrix.numel() else 0.0
    cutoff = math.sqrt(torch.finfo(torch.float32).tiny)
    scaled = matrix.dtype == torch.float32 and 0.0 < maximum < cutoff
    return dict(svd_scale_policy=POLICY, svd_input_max_abs=maximum,
                svd_rescaled=scaled, svd_normalization_scale=maximum if scaled else 1.0,
                svd_scale_cutoff=cutoff,
                svd_scale_note="small-scale normalization does not recover matrix-formation rounding")


def exact_svdvals(matrix, *, driver="gesvd"):
    info = scaling_info(matrix)
    if not info["svd_rescaled"]:
        return frozen_v2.exact_svdvals(matrix, driver=driver)
    scale = info["svd_normalization_scale"]
    # Avoid FP32 division/reciprocal on tiny scalars; normalized entries are O(1).
    normalized = (matrix.double() / scale).to(dtype=matrix.dtype).contiguous()
    singular = frozen_v2.exact_svdvals(normalized, driver=driver)
    # Restoring in FP32 would quantize/underflow tiny singular values again.
    return singular.double() * scale


def entry_energy_qc(matrix, singular, *, driver="gesvd", tolerance=1e-5):
    result = frozen_v2.entry_energy_qc(matrix, singular, driver=driver, tolerance=tolerance)
    result.update(scaling_info(matrix), svd_return_dtype=str(singular.dtype))
    return result


def self_test(device="cpu"):
    import torch
    generator = torch.Generator().manual_seed(20261108)
    normal = torch.randn((48, 24), generator=generator, dtype=torch.float64)
    reports = []
    for label, multiplier in (("zero", 0.0), ("ordinary", 1.0),
                              ("small_normal_entries", 1e-23), ("subnormal_entries", 1e-42)):
        # Form/cast on CPU so test input values are fixed before any GPU arithmetic.
        matrix = (normal * multiplier).float().to(device)
        singular = exact_svdvals(matrix, driver="gesvd")
        qc = entry_energy_qc(matrix, singular, driver="gesvd", tolerance=1e-5)
        reference = torch.linalg.svdvals(matrix.double().cpu())
        error = float(torch.linalg.vector_norm(singular.double().cpu()-reference) /
                      torch.linalg.vector_norm(reference)) if bool(reference.any()) else 0.0
        assert error < 2e-6, (label, error)
        assert qc["svd_energy_qc_pass"]
        assert qc["svd_rescaled"] == (label in ("small_normal_entries", "subnormal_entries"))
        reports.append(dict(case=label, device=device, singular_relative_error=error, **qc))
    return reports


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test; production access is through the versioned launcher")
    print(json.dumps(self_test(args.device), indent=2))
