"""Shared, deterministic utilities for the ICLR 2027 experiment suite."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch


CHECKPOINT_REVISIONS = {
    "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "mamba2": "4f086c08c1e0f07bdc50ca25125dbbf7475d21da",
    "retnet": "7fddefc4d5e196a8d1f076bb7612d54321b3effe",
    "based": "4a63ec61ddb6043a88c198463a7ae75fa90d4038",
    "wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
}

PRIMARY_LENGTHS = (64, 128, 256, 384, 512, 768, 1024)
LAYERS = {
    "gpt2": (0, 5, 11),
    "based": (1, 11, 21),
    "mamba2": (0, 31, 63),
    "retnet": (0, 11, 23),
}
HEADS = {
    "gpt2": tuple(range(12)),
    "based": tuple(range(16)),
    "mamba2": (0, 16, 32, 48, 64, 80, 96, 112),
    "retnet": tuple(range(8)),
}


def clean_wikitext(text: str) -> str:
    text = (text or "").strip()
    text = text.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text.strip()


def is_heading(text: str) -> bool:
    return bool(re.fullmatch(r"=+\s*.*?\s*=+", (text or "").strip()))


def is_article_heading(text: str) -> bool:
    return bool(re.fullmatch(r"=\s+[^=].*?\s+=", (text or "").strip()))


@dataclass(frozen=True)
class WikiDocument:
    document_id: str
    title: str
    article_index: int
    row_start: int
    row_end: int
    text: str


def load_wikitext_documents(parquet_path: str | Path) -> list[WikiDocument]:
    rows = pq.read_table(parquet_path, columns=["text"]).column("text").to_pylist()
    starts = [index for index, value in enumerate(rows) if is_article_heading(value)]
    starts.append(len(rows))
    documents: list[WikiDocument] = []
    for article_index, (row_start, row_end) in enumerate(zip(starts, starts[1:])):
        title = clean_wikitext(rows[row_start]).strip("= ")
        body = [clean_wikitext(value) for value in rows[row_start + 1 : row_end]]
        text = "\n\n".join(value for value in body if value and not is_heading(value))
        digest = hashlib.sha256(f"{article_index}:{title}".encode()).hexdigest()[:12]
        documents.append(
            WikiDocument(
                document_id=f"wikitext103-val-{article_index:03d}-{digest}",
                title=title,
                article_index=article_index,
                row_start=row_start,
                row_end=row_end,
                text=text,
            )
        )
    return documents


def choose_documents(
    documents: Sequence[WikiDocument],
    tokenizers: dict[str, object],
    *,
    count: int = 32,
    minimum_tokens: int = 1056,
    seed: int = 20270802,
) -> list[WikiDocument]:
    qualified = []
    for document in documents:
        lengths = {
            name: len(tokenizer(document.text, add_special_tokens=False, verbose=False)["input_ids"])
            for name, tokenizer in tokenizers.items()
        }
        if min(lengths.values()) >= minimum_tokens:
            qualified.append(document)
    if len(qualified) < count:
        raise RuntimeError(f"Only {len(qualified)} documents have at least {minimum_tokens} tokens.")
    rng = random.Random(seed)
    return sorted(rng.sample(qualified, count), key=lambda document: document.article_index)


def prefix_record(document: WikiDocument, tokenizer, model: str, length: int, condition: str) -> dict:
    boundary_id = tokenizer.eos_token_id
    ids = tokenizer(document.text, add_special_tokens=False, verbose=False)["input_ids"]
    if condition == "explicit-boundary":
        if boundary_id is None:
            raise ValueError(f"Tokenizer for {model} has no boundary/eos token.")
        ids = [int(boundary_id), *ids]
    elif condition != "native-prefix":
        raise ValueError(f"Unknown input condition: {condition}")
    used = [int(value) for value in ids[:length]]
    if len(used) != length:
        raise RuntimeError(f"{document.document_id}/{model} has only {len(used)} tokens at length {length}.")
    decoded = tokenizer.decode(used, skip_special_tokens=False)
    return {
        "checkpoint": model,
        "document_id": document.document_id,
        "title": document.title,
        "article_index": document.article_index,
        "source_row_start": document.row_start,
        "source_row_end": document.row_end,
        "sample_offset": 0,
        "input_condition": condition,
        "length": length,
        "token_ids": used,
        "decoded_text": decoded,
        "text_sha256": hashlib.sha256(decoded.encode()).hexdigest(),
    }


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def singular_metrics(matrix: torch.Tensor, *, return_spectrum: bool = False) -> dict:
    singular = torch.linalg.svdvals(matrix.detach().float()).double().cpu()
    squared = singular.square()
    sum_s = singular.sum()
    sum_sq = squared.sum()
    if not singular.numel() or float(sum_sq) == 0.0:
        metrics = {
            "srank": 0.0,
            "erank_1": 0.0,
            "erank_2": 0.0,
            "operator_norm": 0.0,
            "frobenius_norm": 0.0,
            "nuclear_norm": 0.0,
            "best_rank1_error": 0.0,
        }
    else:
        p1 = singular / sum_s
        p2 = squared / sum_sq
        h1 = -(p1[p1 > 0] * p1[p1 > 0].log()).sum()
        h2 = -(p2[p2 > 0] * p2[p2 > 0].log()).sum()
        metrics = {
            "srank": float(sum_sq / squared[0]),
            "erank_1": float(h1.exp()),
            "erank_2": float(h2.exp()),
            "operator_norm": float(singular[0]),
            "frobenius_norm": float(sum_sq.sqrt()),
            "nuclear_norm": float(sum_s),
            "best_rank1_error": float(torch.sqrt(torch.clamp(1.0 - squared[0] / sum_sq, min=0.0))),
        }
    cumulative = torch.cumsum(squared, dim=0) / sum_sq.clamp_min(torch.finfo(sum_sq.dtype).tiny)
    metrics["top32_singular_values"] = singular[:32].tolist()
    metrics["top32_energy_fractions"] = cumulative[:32].tolist()
    if return_spectrum:
        metrics["singular_values"] = singular.numpy()
    return metrics


def rank_metrics_from_singular_values(
    singular_values: torch.Tensor | np.ndarray | Sequence[float],
    *,
    dimension: int,
) -> dict:
    """Rank diagnostics from a (possibly thin) nonnegative singular spectrum."""
    singular = torch.as_tensor(singular_values, dtype=torch.float64).flatten().cpu()
    singular = singular[torch.isfinite(singular) & (singular >= 0)]
    singular, _ = torch.sort(singular, descending=True)
    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError(f"Rank-normalization dimension must be positive, got {dimension}.")
    if singular.numel() == 0 or float(singular[0]) == 0.0:
        return {
            "srank": 0.0,
            "erank_1": 0.0,
            "erank_2": 0.0,
            "r90": 0,
            "r95": 0,
            "r99": 0,
            "count_rel_gt_1e-2": 0,
            "count_rel_gt_1e-3": 0,
            "count_rel_gt_1e-2_fraction": 0.0,
            "count_rel_gt_1e-3_fraction": 0.0,
            "dimension": dimension,
            "srank_fraction": 0.0,
            "erank_1_fraction": 0.0,
            "erank_2_fraction": 0.0,
            "r90_fraction": 0.0,
            "r95_fraction": 0.0,
            "r99_fraction": 0.0,
            "top32_singular_values": singular[:32].tolist(),
            "top32_energy_fractions": [0.0] * min(32, singular.numel()),
        }

    squared = singular.square()
    total = squared.sum()
    cumulative = torch.cumsum(squared, 0) / total.clamp_min(torch.finfo(total.dtype).tiny)
    p1 = singular / singular.sum()
    p2 = squared / total
    h1 = -(p1[p1 > 0] * p1[p1 > 0].log()).sum()
    h2 = -(p2[p2 > 0] * p2[p2 > 0].log()).sum()
    metrics = {
        "srank": float(total / squared[0]),
        "erank_1": float(h1.exp()),
        "erank_2": float(h2.exp()),
        "r90": int(torch.searchsorted(cumulative, 0.90).item() + 1),
        "r95": int(torch.searchsorted(cumulative, 0.95).item() + 1),
        "r99": int(torch.searchsorted(cumulative, 0.99).item() + 1),
        "count_rel_gt_1e-2": int((singular / singular[0] > 1e-2).sum()),
        "count_rel_gt_1e-3": int((singular / singular[0] > 1e-3).sum()),
        "dimension": dimension,
        "top32_singular_values": singular[:32].tolist(),
        "top32_energy_fractions": cumulative[:32].tolist(),
    }
    for key in ("srank", "erank_1", "erank_2", "r90", "r95", "r99"):
        metrics[f"{key}_fraction"] = float(metrics[key]) / dimension
    metrics["count_rel_gt_1e-2_fraction"] = metrics["count_rel_gt_1e-2"] / dimension
    metrics["count_rel_gt_1e-3_fraction"] = metrics["count_rel_gt_1e-3"] / dimension
    return metrics


def factor_metrics(factor: torch.Tensor) -> dict:
    singular = torch.linalg.svdvals(factor.float()).double().cpu()
    return rank_metrics_from_singular_values(singular, dimension=min(factor.shape))


def bootstrap_document_slopes(
    document_values: dict[str, dict[int, float]],
    *,
    min_length: int = 256,
    resamples: int = 2000,
    seed: int = 20270802,
) -> dict:
    documents = sorted(document_values)
    lengths = sorted({length for values in document_values.values() for length in values if length >= min_length})
    if len(lengths) < 2 or not documents:
        raise ValueError("At least two lengths and one document are required.")

    def slope(sample: Sequence[str]) -> float:
        medians = [np.median([document_values[doc][length] for doc in sample]) for length in lengths]
        return float(np.polyfit(np.log(lengths), np.log(medians), 1)[0])

    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [slope([documents[index] for index in rng.integers(0, len(documents), len(documents))]) for _ in range(resamples)]
    )
    return {
        "slope": slope(documents),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "n_documents": len(documents),
        "n_resamples": resamples,
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def explicit_semiseparable_mask(a: torch.Tensor) -> torch.Tensor:
    a = a.float()
    if a.ndim < 1 or a.shape[-1] < 1:
        raise ValueError("A transition sequence must have positive length.")
    # Preserve an exact zero transition.  The prefix-ratio shortcut cannot do
    # this because subtracting two prefixes after a zero cancels its clipped
    # log value.  The zero-safe path is used only in explicit validation and
    # intervention construction, never in the implicit memory calculation.
    if bool((a == 0).any()):
        length = a.shape[-1]
        rows = []
        for index in range(length):
            row = torch.zeros_like(a)
            row[..., index] = 1.0
            product = torch.ones_like(a[..., 0])
            for source in range(index - 1, -1, -1):
                product = product * a[..., source + 1]
                row[..., source] = product
            rows.append(row)
        return torch.stack(rows, dim=-2)
    log_a = torch.log(a).double()
    prefix = torch.cumsum(log_a, dim=-1)
    log_l = prefix[..., :, None] - prefix[..., None, :]
    causal = torch.tril(torch.ones(a.shape[-1], a.shape[-1], dtype=torch.bool, device=a.device))
    return torch.exp(log_l).masked_fill(~causal, 0.0).to(a.dtype)


def _effective_half_life_summary(a: torch.Tensor) -> dict:
    """Summarize first strictly-sub-half retained products over valid starts."""
    length = a.numel()
    if length <= 1:
        return {
            "threshold": 0.5,
            "strictly_below": True,
            "n_start_positions": 0,
            "n_observed": 0,
            "censored_fraction": 1.0,
            "minimum": None,
            "median": None,
            "q95": None,
            "maximum": None,
        }

    # Prefix indexing below is zero based over the transition sequence
    # (a[1], ..., a[length-1]). A start token j targets the prefix after at
    # least one transition, so search begins at prefix index j + 1.
    costs = -torch.log(a[1:].clamp_min(torch.finfo(a.dtype).tiny)).double().cpu()
    prefix = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(costs, 0)))
    starts = torch.arange(length - 1, dtype=torch.long)
    targets = prefix[:-1] + math.log(2.0)
    strict_targets = targets + 1e-7 * targets.abs().clamp_min(1.0) + 1e-12
    stops = torch.searchsorted(prefix, strict_targets, right=False)
    stops = torch.maximum(stops, starts + 1)
    # A stop at the last prefix index is observable if the terminal product is
    # strictly below one half.  Equal-to-half remains censored by definition.
    observed = (stops < length) | ((stops == length) & (prefix[-1] > strict_targets))
    lags = (stops[observed] - starts[observed]).double()

    def statistic(kind: str):
        if not lags.numel():
            return None
        if kind == "minimum":
            return int(lags.min())
        if kind == "maximum":
            return int(lags.max())
        if kind == "median":
            return float(torch.quantile(lags, 0.5))
        return float(torch.quantile(lags, 0.95))

    return {
        "threshold": 0.5,
        "strictly_below": True,
        "n_start_positions": length - 1,
        "n_observed": int(observed.sum()),
        "censored_fraction": float(1.0 - observed.double().mean()),
        "minimum": statistic("minimum"),
        "median": statistic("median"),
        "q95": statistic("q95"),
        "maximum": statistic("maximum"),
    }


def memory_metrics(
    a: torch.Tensor,
    power_iterations: int = 200,
    *,
    power_tolerance: float = 1e-7,
    validate_exact: bool = False,
    validation_rtol: float = 5e-4,
    validation_atol: float = 2e-5,
) -> dict:
    """Compute semiseparable-mask memory without materializing the mask.

    ``a[0]`` is retained for transition-distribution summaries but, consistently
    with ``L[i,j] = prod(a[j+1:i+1])``, it does not affect the mask.
    """
    a = a.detach().float().flatten()
    length = a.numel()
    if length < 1:
        raise ValueError("A transition sequence must have positive length.")
    if not torch.isfinite(a).all():
        raise ValueError("Transition values must be finite.")
    tolerance = 32 * torch.finfo(a.dtype).eps
    if bool((a < -tolerance).any()) or bool((a > 1.0 + tolerance).any()):
        raise ValueError("Semiseparable transitions must lie in [0, 1].")
    a = a.clamp(0.0, 1.0)
    a64 = a.double()

    def build_recurrence(transitions: torch.Tensor):
        # All power iterates are positive because L is nonnegative.  This
        # log-cumulative form avoids O(length) kernel launches. Split at exact
        # zeros because each such transition resets the recurrence.
        starts = [0, *([int(index) + 1 for index in torch.nonzero(transitions[1:] == 0, as_tuple=False).flatten().cpu()])]
        stops = [*starts[1:], length]
        segments = []
        for start, stop in zip(starts, stops):
            segment_transitions = transitions[start:stop]
            log_prefix = torch.cat(
                (torch.zeros(1, dtype=torch.float64, device=a.device), torch.cumsum(torch.log(segment_transitions[1:]), 0))
            )
            segments.append((start, stop, log_prefix))

        def apply(vector: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(vector)
            tiny = torch.finfo(torch.float64).tiny
            for start, stop, log_prefix in segments:
                segment_vector = vector[start:stop]
                output[start:stop] = torch.exp(
                    log_prefix + torch.logcumsumexp(torch.log(segment_vector.clamp_min(tiny)) - log_prefix, 0)
                )
            return output

        return apply

    reverse_transitions = torch.cat((torch.ones_like(a64[:1]), a64[1:].flip(0)))
    forward_recurrence = build_recurrence(a64)
    reverse_recurrence = build_recurrence(reverse_transitions)

    def apply_l(vector: torch.Tensor) -> torch.Tensor:
        return forward_recurrence(vector)

    def apply_lt(vector: torch.Tensor) -> torch.Tensor:
        return reverse_recurrence(vector.flip(0)).flip(0)

    row = apply_l(torch.ones_like(a64))
    col = apply_lt(torch.ones_like(a64))
    vector = torch.full_like(a64, 1.0 / math.sqrt(length))
    eigenvalue = torch.tensor(0.0, device=a.device)
    iterations_used = 0
    for iteration in range(power_iterations):
        applied = apply_lt(apply_l(vector))
        next_vector = applied / torch.linalg.vector_norm(applied).clamp_min(torch.finfo(applied.dtype).tiny)
        next_eigenvalue = torch.dot(vector, applied)
        iterations_used = iteration + 1
        if iteration and abs(float(next_eigenvalue - eigenvalue)) <= power_tolerance * max(float(next_eigenvalue), 1.0):
            vector = next_vector
            break
        vector, eigenvalue = next_vector, next_eigenvalue
    final_applied = apply_lt(apply_l(vector))
    eigenvalue = torch.dot(vector, final_applied)
    gain = torch.sqrt(eigenvalue.clamp_min(0.0))
    residual = torch.linalg.vector_norm(final_applied - eigenvalue * vector)
    residual /= eigenvalue.abs().clamp_min(torch.finfo(eigenvalue.dtype).tiny)
    block_products = {}
    for block in (1, 4, 8, 16, 32, 64):
        if block < length:
            values = a64[1:].unfold(0, block, 1).prod(-1)
            block_products[str(block)] = {
                "max": float(values.max()),
                "q95": float(torch.quantile(values, 0.95)),
                "median": float(torch.median(values)),
            }
    half_life = _effective_half_life_summary(a)
    result = {
        "a_min": float(a.min()),
        "a_mean": float(a.mean()),
        "a_max": float(a.max()),
        "a_q90": float(torch.quantile(a, 0.90)),
        "a_q95": float(torch.quantile(a, 0.95)),
        "a_q99": float(torch.quantile(a, 0.99)),
        "a_median": float(torch.median(a)),
        "mask_R": float(row.max()),
        "mask_C": float(col.max()),
        "mask_kappa": float(gain),
        "mask_kappa_power_iterations": iterations_used,
        "mask_kappa_power_residual": float(residual),
        "block_products": block_products,
        "effective_half_life_summary": half_life,
        "effective_half_life_min": half_life["minimum"],
        "effective_half_life_median": half_life["median"],
        "effective_half_life_q95": half_life["q95"],
        "effective_half_life_max": half_life["maximum"],
        "effective_half_life_censored_fraction": half_life["censored_fraction"],
    }
    if validate_exact:
        explicit = explicit_semiseparable_mask(a)
        exact_r = explicit.sum(-1).max().double()
        exact_c = explicit.sum(-2).max().double()
        exact_gain = torch.linalg.svdvals(explicit)[0].double()
        errors = {
            "mask_R_exact": float(exact_r),
            "mask_C_exact": float(exact_c),
            "mask_kappa_exact": float(exact_gain),
            "mask_R_validation_relative_error": float(abs(row.max() - exact_r) / exact_r.clamp_min(1e-30)),
            "mask_C_validation_relative_error": float(abs(col.max() - exact_c) / exact_c.clamp_min(1e-30)),
            "mask_kappa_validation_relative_error": float(abs(gain - exact_gain) / exact_gain.clamp_min(1e-30)),
        }
        passed = (
            torch.allclose(row.max(), exact_r, rtol=validation_rtol, atol=validation_atol)
            and torch.allclose(col.max(), exact_c, rtol=validation_rtol, atol=validation_atol)
            and torch.allclose(gain, exact_gain, rtol=validation_rtol, atol=validation_atol)
        )
        errors["mask_exact_validation_passed"] = bool(passed)
        result.update(errors)
        if not passed:
            raise RuntimeError(f"Implicit semiseparable memory validation failed: {errors}")
    return result


def geometric_memory_metrics(
    gamma: float,
    length: int,
    *,
    validate_exact: bool = False,
    power_iterations: int = 200,
) -> dict:
    """Memory diagnostics and analytic envelopes for a geometric mask."""
    gamma = float(gamma)
    length = int(length)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"Geometric decay must lie in [0, 1], got {gamma}.")
    if length < 1:
        raise ValueError(f"Mask length must be positive, got {length}.")
    transitions = torch.full((length,), gamma, dtype=torch.float32)
    result = memory_metrics(
        transitions,
        power_iterations=power_iterations,
        validate_exact=validate_exact,
    )
    finite_sum = float(length) if gamma == 1.0 else float((1.0 - gamma**length) / (1.0 - gamma))
    envelope = None if gamma == 1.0 else float(1.0 / (1.0 - gamma))
    result.update(
        {
            "mask_family": "geometric",
            "mask_gamma": gamma,
            "mask_R_analytic_finite": finite_sum,
            "mask_C_analytic_finite": finite_sum,
            "mask_R_analytic_relative_error": abs(result["mask_R"] - finite_sum) / max(finite_sum, 1e-30),
            "mask_C_analytic_relative_error": abs(result["mask_C"] - finite_sum) / max(finite_sum, 1e-30),
            "mask_analytic_envelope": envelope,
            "mask_kappa_analytic_envelope": envelope,
            "mask_R_over_analytic_envelope": None if envelope is None else result["mask_R"] / envelope,
            "mask_C_over_analytic_envelope": None if envelope is None else result["mask_C"] / envelope,
            "mask_kappa_over_analytic_envelope": None if envelope is None else result["mask_kappa"] / envelope,
        }
    )
    if result["mask_R_analytic_relative_error"] > 2e-5 or result["mask_C_analytic_relative_error"] > 2e-5:
        raise RuntimeError("Geometric memory recurrence disagrees with its finite analytic sum.")
    if envelope is not None and result["mask_kappa"] > envelope * (1.0 + 5e-4):
        raise RuntimeError("Measured geometric-mask gain exceeds the analytic envelope.")
    return result
