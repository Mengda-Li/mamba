"""Explicit SVD solver and independent entry-energy checks for corrected runs."""
from __future__ import annotations
import math


def exact_svdvals(matrix, *, driver="gesvd"):
    import torch
    if driver not in ("gesvd", "gesvdj", "default"):
        raise ValueError(f"Unsupported SVD driver: {driver}")
    kwargs = {"driver": driver} if matrix.device.type == "cuda" and driver != "default" else {}
    return torch.linalg.svdvals(matrix, **kwargs)


def entry_energy_qc(matrix, singular, *, driver="gesvd", tolerance=1e-5):
    entry_energy = float(matrix.double().square().sum())
    svd_energy = float(singular.double().square().sum())
    entry_fro, svd_fro = math.sqrt(entry_energy), math.sqrt(svd_energy)
    error = abs(svd_fro - entry_fro) / entry_fro if entry_fro else (0.0 if svd_fro == 0 else math.inf)
    passed = math.isfinite(error) and error <= tolerance
    if driver == "gesvd" and not passed:
        raise RuntimeError(f"SVD Frobenius QC failed: relative error {error:.9g} > {tolerance}; entry={entry_fro}, SVD={svd_fro}")
    return dict(entry_squared_energy=entry_energy, svd_squared_energy=svd_energy,
        entry_frobenius_norm=entry_fro, svd_frobenius_norm=svd_fro,
        svd_frobenius_relative_error=error,
        svd_squared_energy_relative_error=(svd_energy - entry_energy) / entry_energy if entry_energy else 0.0,
        svd_energy_qc_pass=passed, svd_energy_qc_tolerance=tolerance,
        svd_driver_requested=driver,
        svd_driver_effective=(driver if driver != "default" else "cuda_default") if matrix.device.type == "cuda" else "cpu_default")


def helper_rank_metrics(matrix, helper, *, svd_device, driver="gesvd", tolerance=1e-5):
    import torch
    work = matrix.to(device=svd_device, dtype=torch.float32).contiguous()
    singular = exact_svdvals(work, driver=driver)
    metrics = helper._rank_metrics(singular)
    qc = entry_energy_qc(work, singular, driver=driver, tolerance=tolerance)
    top = metrics["operator_norm"]
    metrics["srank"] = qc["entry_squared_energy"] / top**2 if top else 0.0
    metrics["frobenius_norm"] = qc["entry_frobenius_norm"]
    metrics["best_rank1_error"] = math.sqrt(max(0.0, 1.0 - top**2 / qc["entry_squared_energy"])) if qc["entry_squared_energy"] else 0.0
    metrics.update(qc)
    return singular, metrics
