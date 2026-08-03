"""Model-free tests for GPT-2 production scheduling and persistence."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import run_gpt2_suite as runner


class _Output:
    def __init__(self, rows, arrays):
        self.rows = rows
        self.arrays = arrays


def synthetic_output(length: int) -> _Output:
    full, intensive = runner.dense_head_policy(length)
    rows = []
    arrays = {}
    for layer in runner.ALL_LAYERS:
        for head in runner.ALL_HEADS:
            row = {
                "document_id": "doc",
                "input_condition": "native-prefix",
                "length": length,
                "layer": layer,
                "head_or_channel": head,
                "work_package": "WP2",
                "operator_variant": "native_softmax",
                "full_operator_spectrum": head in full[layer],
                "intensive_dense_diagnostics": head in intensive[layer],
                "sink_S": 1.0,
                "sink_D": 0.5,
                "sink_D_over_S": 0.5,
                "sink_rho2": 1.0 / 3.0,
                "margin_median": 0.0,
                "collision_scaled_median": 1.0,
            }
            if head in full[layer]:
                ref = f"doc_native-prefix_{length}_{layer}_{head}:native_softmax_singular_values"
                row.update(
                    {
                        "srank": 2.0,
                        "erank_1": 3.0,
                        "erank_2": 2.5,
                        "spectrum_ref": ref,
                    },
                )
                arrays[ref] = [1.0, 0.5]
            rows.append(row)
    return _Output(rows, arrays)


def fake_numpy_module() -> types.ModuleType:
    module = types.ModuleType("numpy")

    def savez_compressed(handle, **arrays):
        payload = json.dumps(arrays, sort_keys=True, separators=(",", ":")).encode("ascii")
        handle.write(b"FAKE_NPZ\0" + payload)

    module.savez_compressed = savez_compressed
    return module


class Gpt2RunnerTests(unittest.TestCase):
    def test_result_config_distinguishes_model_execution_device(self):
        common = {
            "checkpoint_revision": "revision",
            "manifest_sha256": "manifest",
            "source_hashes": {"runner": "source"},
            "environment_versions": {"python": "test"},
            "analysis_device": "cpu",
            "svd_device": "cpu",
            "svd_dtype": "float32",
        }
        cuda = runner.protocol_config(model_device_map="cuda", **common)
        cpu = runner.protocol_config(model_device_map="cpu", **common)
        self.assertNotEqual(runner.fingerprint(cuda), runner.fingerprint(cpu))

    def test_long_context_policy_preserves_wp1_all_heads(self):
        full, intensive = runner.dense_head_policy(1024)
        for layer in runner.WP1_LAYERS:
            self.assertEqual(full[layer], runner.ALL_HEADS)
        for layer in runner.ALL_LAYERS:
            expected = runner.PREREGISTERED_HEADS if layer in runner.WP1_LAYERS else ()
            self.assertEqual(intensive[layer], expected)
        runner.validate_analysis_coverage(synthetic_output(1024).rows, 1024)

    def test_short_context_uses_the_same_preregistered_scope(self):
        full, intensive = runner.dense_head_policy(512)
        for layer in runner.ALL_LAYERS:
            expected_full = runner.ALL_HEADS if layer in runner.WP1_LAYERS else runner.PREREGISTERED_HEADS
            expected_intensive = runner.PREREGISTERED_HEADS if layer in runner.WP1_LAYERS else ()
            self.assertEqual(full[layer], expected_full)
            self.assertEqual(intensive[layer], expected_intensive)
        runner.validate_analysis_coverage(synthetic_output(512).rows, 512)

    def test_nested_array_references_are_detected(self):
        rows = [
            {
                "spectrum_ref": "native",
                "token_alignment": json.dumps(
                    [{"canonical_correlations_ref": "canonical"}],
                    sort_keys=True,
                ),
            },
        ]
        self.assertEqual(runner.collect_array_refs(rows), {"native", "canonical"})

    def test_atomic_sample_resume_and_deduplicated_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            out_dir = Path(temporary_directory) / "experiments_2027"
            run_root = out_dir / "gpt2_runs" / "config-test"
            sample = {
                "checkpoint": "gpt2",
                "document_id": "doc",
                "article_index": 1,
                "sample_offset": 0,
                "input_condition": "native-prefix",
                "length": 1024,
                "token_ids": list(range(1024)),
                "text_sha256": "text",
            }
            revision = "revision"
            sample_name = runner.sample_id(sample, revision)
            sample_hash = runner.sample_fingerprint(sample, revision)
            output = synthetic_output(1024)
            fake_numpy = fake_numpy_module()
            with mock.patch.dict(sys.modules, {"numpy": fake_numpy}):
                marker = runner.write_sample_shard(
                    run_root=run_root,
                    output=output,
                    sample=sample,
                    sample_name=sample_name,
                    sample_hash=sample_hash,
                    config_id="config-test",
                    impl_id="implementation-test",
                )

            marker_path = run_root / "completed" / f"{sample_name}.json"
            valid, reason = runner.valid_completion_marker(
                marker_path,
                expected_config_id="config-test",
                expected_implementation_id="implementation-test",
                expected_sample_id=sample_name,
                expected_sample_fingerprint=sample_hash,
            )
            self.assertTrue(valid, reason)
            self.assertEqual(marker["row_count"], 144)

            first = runner.rebuild_metrics_index(
                run_root=run_root,
                out_dir=out_dir,
                config_id="config-test",
                impl_id="implementation-test",
            )
            second = runner.rebuild_metrics_index(
                run_root=run_root,
                out_dir=out_dir,
                config_id="config-test",
                impl_id="implementation-test",
            )
            self.assertEqual(first["deduplicated_metric_rows"], 144)
            self.assertEqual(second["deduplicated_metric_rows"], 144)
            index_rows = runner.read_jsonl(out_dir / "gpt2_metrics.jsonl")
            self.assertEqual(len(index_rows), 144)
            self.assertEqual(len({row["row_id"] for row in index_rows}), 144)
            self.assertTrue(all(row["array_archive"].endswith(".npz") for row in index_rows))

            arrays_path = run_root / marker["arrays_path"]
            with arrays_path.open("ab") as handle:
                handle.write(b"corrupt")
            valid, reason = runner.valid_completion_marker(
                marker_path,
                expected_config_id="config-test",
                expected_implementation_id="implementation-test",
            )
            self.assertFalse(valid)
            self.assertEqual(reason, "arrays_hash_mismatch")


if __name__ == "__main__":
    unittest.main()
