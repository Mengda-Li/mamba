"""Summarize paired relative prediction-loss and perplexity changes.

Each recorded trial gives 100 * (intervention_loss - native_loss) / native_loss
and 100 * expm1(intervention_loss - native_loss), respectively.  For both
metrics, heads are averaged within a document before equal-weight document
averaging.  Confidence intervals bootstrap those paired document summaries.
No model execution or external dependencies are required.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def write_csv(path: Path, rows: list[dict], compressed: bool = False) -> None:
    fields = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode()
    path.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-documents", type=int, default=32)
    parser.add_argument("--expected-heads", type=int, default=12)
    parser.add_argument("--expected-rows", type=int, default=6912)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    if args.bootstrap_resamples < 2:
        parser.error("At least two bootstrap resamples are required")
    opener = gzip.open if args.input.suffix == ".gz" else open
    with opener(args.input, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} trials, found {len(rows)}")

    seen = set()
    baselines = {}
    layer_groups = defaultdict(lambda: defaultdict(list))
    head_groups = defaultdict(dict)
    max_recorded_delta_error = 0.0
    max_sum_to_loss_error = 0.0
    configs, checkpoints = set(), set()
    for row in rows:
        if row["status"] != "ok":
            raise ValueError("Non-successful trial in numeric input")
        for key in ("length", "layer", "head_or_channel", "loss_token_count"):
            row[key] = int(row[key])
        for key in ("native_loss", "intervention_loss", "delta_loss",
                    "native_nll_sum", "intervention_nll_sum"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Non-finite {key}")
        if row["native_loss"] <= 0:
            raise ValueError("Relative loss requires a strictly positive native NLL")
        if row["loss_token_count"] != row["length"] - 1:
            raise ValueError("Mismatched next-token target count")
        configs.add(row["config_id"])
        checkpoints.add(row["checkpoint"])
        key = (row["config_id"], row["checkpoint"], row["document_id"],
               row["length"], row["layer"], row["head_or_channel"], row["intervention"])
        if key in seen:
            raise ValueError("Duplicate trial key")
        seen.add(key)
        baseline_key = (row["config_id"], row["checkpoint"], row["document_id"], row["length"])
        baseline = (row["native_loss"], row["native_nll_sum"], row["loss_token_count"])
        if baseline_key in baselines and baselines[baseline_key] != baseline:
            raise ValueError("Native baseline changes across heads/layers/interventions")
        baselines[baseline_key] = baseline
        delta = row["intervention_loss"] - row["native_loss"]
        max_recorded_delta_error = max(max_recorded_delta_error, abs(delta - row["delta_loss"]))
        for prefix in ("native", "intervention"):
            max_sum_to_loss_error = max(max_sum_to_loss_error,
                abs(row[f"{prefix}_nll_sum"] / row["loss_token_count"] - row[f"{prefix}_loss"]))
        row["relative_loss_percent"] = 100 * delta / row["native_loss"]
        row["relative_ppl_percent"] = 100 * math.expm1(delta)
        if not math.isfinite(row["relative_ppl_percent"]):
            raise ValueError("Non-finite relative perplexity effect")
        group = (row["checkpoint"], row["length"], row["layer"], row["intervention"])
        layer_groups[group][row["document_id"]].append(row)
        head_key = group + (row["head_or_channel"],)
        head_groups[head_key][row["document_id"]] = row
    if len(configs) != 1 or len(checkpoints) != 1:
        raise ValueError("Analyze one frozen run and checkpoint at a time")
    if max_recorded_delta_error > 1e-12 or max_sum_to_loss_error > 1e-12:
        raise ValueError("Recorded loss/delta/NLL-sum inconsistency")

    document_ids = sorted({row["document_id"] for row in rows})
    if len(document_ids) != args.expected_documents:
        raise ValueError("Unexpected document cohort size")
    for group, documents in layer_groups.items():
        if sorted(documents) != document_ids:
            raise ValueError(f"Incomplete document cohort: {group}")
        for trial_rows in documents.values():
            if sorted(row["head_or_channel"] for row in trial_rows) != list(range(args.expected_heads)):
                raise ValueError(f"Incomplete head cohort: {group}")

    rng = random.Random(args.seed)
    samples = [rng.choices(range(len(document_ids)), k=len(document_ids))
               for _ in range(args.bootstrap_resamples)]

    def interval(values: list[float]) -> tuple[float, float]:
        draws = [statistics.fmean(values[index] for index in sample) for sample in samples]
        return percentile(draws, .025), percentile(draws, .975)

    document_rows, layer_summary, head_summary = [], [], []
    config_id = next(iter(configs))
    for (checkpoint, length, layer, intervention), documents in sorted(layer_groups.items()):
        identity = dict(config_id=config_id, checkpoint=checkpoint, length=length,
                        layer=layer, intervention=intervention, metric="relative_loss_percent")
        doc_values, native_values, intervention_values, delta_values = [], [], [], []
        for doc in document_ids:
            trials = documents[doc]
            value = statistics.fmean(row["relative_loss_percent"] for row in trials)
            native = trials[0]["native_loss"]
            modified = statistics.fmean(row["intervention_loss"] for row in trials)
            delta = statistics.fmean(row["delta_loss"] for row in trials)
            doc_values.append(value)
            native_values.append(native)
            intervention_values.append(modified)
            delta_values.append(delta)
            document_rows.append({**identity, "document_id": doc, "relative_loss_percent": value,
                                  "native_loss": native, "intervention_loss_mean_heads": modified,
                                  "delta_loss_mean_heads": delta, "n_heads": len(trials)})
        lo, hi = interval(doc_values)
        layer_summary.append({**identity, "estimate": statistics.fmean(doc_values),
                              "ci_low": lo, "ci_high": hi, "n_documents": len(doc_values),
                              "n_observations": len(doc_values) * args.expected_heads,
                              "n_heads_per_document": args.expected_heads,
                              "mean_native_loss": statistics.fmean(native_values),
                              "mean_intervention_loss": statistics.fmean(intervention_values),
                              "mean_delta_loss": statistics.fmean(delta_values)})

    final_layer = max(row["layer"] for row in rows)
    final_length = max(row["length"] for row in rows)
    for (checkpoint, length, layer, intervention, head), documents in sorted(head_groups.items()):
        if layer != final_layer or length != final_length:
            continue
        if sorted(documents) != document_ids:
            raise ValueError("Incomplete head-level document cohort")
        trials = [documents[doc] for doc in document_ids]
        values = [row["relative_loss_percent"] for row in trials]
        lo, hi = interval(values)
        head_summary.append(dict(config_id=config_id, checkpoint=checkpoint, length=length,
            layer=layer, intervention=intervention, head_or_channel=head,
            metric="relative_loss_percent", estimate=statistics.fmean(values), ci_low=lo,
            ci_high=hi, n_documents=len(values), n_observations=len(values),
            mean_native_loss=statistics.fmean(row["native_loss"] for row in trials),
            mean_intervention_loss=statistics.fmean(row["intervention_loss"] for row in trials),
            mean_delta_loss=statistics.fmean(row["delta_loss"] for row in trials)))

    # Transform each paired trial before any head/document aggregation.
    # Exponentiating a mean delta would give a different, geometric summary.
    ppl_document_rows, ppl_layer_summary, ppl_head_summary = [], [], []
    for (checkpoint, length, layer, intervention), documents in sorted(layer_groups.items()):
        identity = dict(config_id=config_id, checkpoint=checkpoint, length=length,
                        layer=layer, intervention=intervention, metric="relative_ppl_percent")
        values = []
        for doc in document_ids:
            trials = documents[doc]
            value = statistics.fmean(row["relative_ppl_percent"] for row in trials)
            values.append(value)
            ppl_document_rows.append({**identity, "document_id": doc,
                "relative_ppl_percent": value, "n_heads": len(trials)})
        lo, hi = interval(values)
        ppl_layer_summary.append({**identity, "estimate": statistics.fmean(values),
            "ci_low": lo, "ci_high": hi, "n_documents": len(values),
            "n_observations": len(values) * args.expected_heads,
            "n_heads_per_document": args.expected_heads})
    for (checkpoint, length, layer, intervention, head), documents in sorted(head_groups.items()):
        if layer != final_layer or length != final_length:
            continue
        values = [documents[doc]["relative_ppl_percent"] for doc in document_ids]
        lo, hi = interval(values)
        ppl_head_summary.append(dict(config_id=config_id, checkpoint=checkpoint,
            length=length, layer=layer, intervention=intervention, head_or_channel=head,
            metric="relative_ppl_percent", estimate=statistics.fmean(values), ci_low=lo,
            ci_high=hi, n_documents=len(values), n_observations=len(values)))

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "relative_loss_layer_summary.csv", layer_summary)
    write_csv(args.output / "relative_loss_head_summary.csv", head_summary)
    write_csv(args.output / "relative_loss_document_summary.csv.gz", document_rows, compressed=True)
    write_csv(args.output / "relative_ppl_layer_summary.csv", ppl_layer_summary)
    write_csv(args.output / "relative_ppl_head_summary.csv", ppl_head_summary)
    write_csv(args.output / "relative_ppl_document_summary.csv.gz", ppl_document_rows, compressed=True)
    manifest = dict(metric="relative_loss_percent", units="percent of the same document's native mean next-token NLL",
        formula="100 * (intervention_loss - native_loss) / native_loss for each recorded trial",
        relative_ppl_formula="100 * expm1(intervention_loss - native_loss) for each recorded trial; losses are mean next-token NLL in nats",
        relative_ppl_units="percent of the same trial's native perplexity",
        aggregation="arithmetic head mean within document, then equal-weight arithmetic document mean",
        relative_ppl_aggregation="transform each trial first, then arithmetic head mean within document and equal-weight arithmetic document mean; not expm1(mean delta)",
        bootstrap=dict(unit="document", resamples=args.bootstrap_resamples, seed=args.seed,
                       interval="95% percentile; linear-interpolation quantiles",
                       pairing="relative effect computed before resampling; same sorted document IDs and resamples across summaries"),
        input_path=str(args.input), input_sha256=sha256(args.input), source_code_sha256=sha256(Path(__file__)),
        config_id=config_id, source_trials=len(rows), source_samples=len(baselines),
        document_ids=document_ids, documents=len(document_ids),
        layer_cells=len(layer_summary), final_layer_head_cells=len(head_summary),
        relative_ppl_layer_cells=len(ppl_layer_summary), relative_ppl_head_cells=len(ppl_head_summary),
        final_layer=final_layer, final_length=final_length,
        max_recorded_delta_error=max_recorded_delta_error, max_sum_to_loss_error=max_sum_to_loss_error,
        native_baseline_constant_within_document_and_length=True,
        retained_outcomes="all trials and signs; no selection by effect direction")
    manifest["outputs"] = {name: sha256(args.output / name) for name in
        ("relative_loss_layer_summary.csv", "relative_loss_head_summary.csv", "relative_loss_document_summary.csv.gz",
         "relative_ppl_layer_summary.csv", "relative_ppl_head_summary.csv", "relative_ppl_document_summary.csv.gz")}
    (args.output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "relative_loss_summary.json").write_text(json.dumps(
        {"layer_summary": layer_summary, "head_summary": head_summary}, indent=2) + "\n")
    (args.output / "relative_ppl_summary.json").write_text(json.dumps(
        {"layer_summary": ppl_layer_summary, "head_summary": ppl_head_summary}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "documents": len(document_ids), "layer_cells": len(layer_summary),
                      "final_layer_head_cells": len(head_summary), "relative_ppl_layer_cells": len(ppl_layer_summary),
                      "relative_ppl_head_cells": len(ppl_head_summary), "output": str(args.output)}))


if __name__ == "__main__":
    main()
