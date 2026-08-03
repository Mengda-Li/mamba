"""Focused model-free tests for the structured memory exact-SVD audit."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import audit_structured_memory_svd as audit


def exact_fields(transitions: np.ndarray) -> dict:
    row, column = audit.memory_sums(transitions)
    gain = audit.exact_mask_gain(transitions, dense=True)
    return {
        "mask_R": row,
        "mask_C": column,
        "mask_kappa": gain,
        "mask_kappa_power_iterations": 12,
        "mask_kappa_power_residual": 1e-12,
        "mask_R_exact": row,
        "mask_C_exact": column,
        "mask_kappa_exact": gain,
        "mask_kappa_validation_relative_error": 0.0,
        "mask_exact_validation_passed": True,
    }


def base_row(observation: str, variant: str, length: int = 8) -> dict:
    return {
        "checkpoint": "mamba2",
        "document_id": "doc-0",
        "length": length,
        "layer": 0,
        "head_or_channel": 0,
        "operator_variant": variant,
        "observation_id": observation,
        "spectrum_id": f"spectrum-{observation}",
        "mask_definition": "scalar_semiseparable_transitions",
    }


def write_unit(root: Path, rows: list[dict], transitions: dict[str, np.ndarray] | None = None) -> None:
    metrics_dir = root / "structured_shards" / "mamba2" / "metrics"
    spectra_dir = root / "structured_shards" / "mamba2" / "spectra"
    metrics_dir.mkdir(parents=True)
    spectra_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(metrics_dir / "unit.parquet", index=False)
    np.savez_compressed(
        spectra_dir / "unit.npz",
        **{row["spectrum_id"]: np.asarray([1.0]) for row in rows},
    )
    if transitions:
        transition_dir = root / "structured_shards" / "mamba2" / "transitions"
        transition_dir.mkdir(parents=True)
        np.savez_compressed(transition_dir / "unit.npz", **transitions)


class StructuredMemorySvdAuditTests(unittest.TestCase):
    def test_explicit_mask_uses_a_j_plus_one_and_preserves_zeros(self):
        transitions = np.asarray([0.123, 0.5, 0.0, 0.25], dtype=np.float64)
        expected = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.25, 1.0],
            ]
        )
        np.testing.assert_allclose(audit.explicit_semiseparable_mask(transitions), expected)

    def test_bidiagonal_solver_matches_dense_svd(self):
        rng = np.random.default_rng(20270802)
        cases = [
            np.zeros(12),
            np.ones(12),
            np.full(17, 0.9),
            rng.uniform(0.0, 1.0, size=31),
            np.asarray([0.4, 1.0, 0.0, 0.95, 0.2, 1.0]),
        ]
        for transitions in cases:
            with self.subTest(length=len(transitions)):
                structured = audit.exact_mask_gain(transitions)
                dense = audit.exact_mask_gain(transitions, dense=True)
                self.assertAlmostEqual(structured, dense, delta=1e-10 * max(dense, 1.0))

    def test_1024_selection_matches_production_hash_rule_for_all_variants(self):
        selected_document = None
        for index in range(1000):
            row = pd.Series(base_row(f"obs-{index}", "native", length=1024) | {"document_id": f"doc-{index}"})
            payload = {
                "checkpoint": "mamba2",
                "document_id": f"doc-{index}",
                "layer": 0,
                "head_or_channel": 0,
            }
            expected_digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            selected, _, digest = audit.selected_for_exact_audit(row)
            self.assertEqual(digest, expected_digest)
            self.assertEqual(selected, int(expected_digest[:8], 16) % 10 == 0)
            if selected:
                selected_document = f"doc-{index}"
                break
        self.assertIsNotNone(selected_document)
        for variant in ("native", "transition_shuffled", "constant_decay"):
            row = pd.Series(base_row(f"obs-{variant}", variant, length=1024) | {"document_id": selected_document})
            self.assertTrue(audit.selected_for_exact_audit(row)[0])

    def test_raw_geometric_and_archival_evidence_are_distinguished(self):
        raw = np.asarray([0.2, 0.8, 0.0, 0.9, 0.7, 1.0, 0.5, 0.4], dtype=np.float64)
        geometric = np.full(8, np.float32(0.75), dtype=np.float32).astype(np.float64)
        archival = np.asarray([0.1, 0.6, 0.7, 0.8, 0.9, 0.3, 0.2, 0.4])

        raw_row = base_row("raw", "native") | exact_fields(raw)
        geometric_row = base_row("geometric", "constant_decay") | exact_fields(geometric) | {
            "mask_definition": "geometric_scalar_transitions",
            "mask_family": "geometric",
            "mask_gamma": 0.75,
        }
        archival_row = base_row("archival", "transition_shuffled") | exact_fields(archival)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_unit(root, [raw_row, geometric_row, archival_row], {"transition-raw": raw})
            report, details, exit_code = audit.run_audit(
                root,
                ("mamba2",),
                bidiagonal_crosscheck_max_length=16,
                require_full_length_scope=False,
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(report["status"], "PASS_WITH_ARCHIVAL_LIMITATIONS")
        evidence = {row["observation_id"]: row["evidence"] for row in details}
        self.assertEqual(evidence["raw"], "independent_exact_recomputation")
        self.assertEqual(evidence["geometric"], "independent_exact_recomputation")
        self.assertEqual(evidence["archival"], "archived_runner_exact_fields_only")

    def test_missing_payload_and_exact_fields_is_incomplete(self):
        transitions = np.full(8, 0.8)
        row = base_row("missing", "native")
        row.update({key: value for key, value in exact_fields(transitions).items() if not key.endswith("_exact")})
        row.pop("mask_exact_validation_passed", None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_unit(root, [row])
            report, details, exit_code = audit.run_audit(root, ("mamba2",), require_full_length_scope=False)

        self.assertEqual(exit_code, 2)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(details[0]["evidence"], "missing_exact_evidence")

    def test_wrong_stored_gain_fails_even_when_archived_exact_is_present(self):
        transitions = np.full(8, 0.8)
        row = base_row("wrong-gain", "native") | exact_fields(transitions)
        row["mask_kappa"] *= 1.1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_unit(root, [row])
            report, details, exit_code = audit.run_audit(root, ("mamba2",), require_full_length_scope=False)

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("stored_power_gain_disagrees_with_exact_svd", details[0]["issues"])

    def test_missing_power_metadata_is_selected_and_fails(self):
        transitions = np.full(8, 0.8)
        row = base_row("missing-power", "native") | exact_fields(transitions)
        row["mask_kappa_power_iterations"] = None
        row["mask_kappa_power_residual"] = None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_unit(root, [row])
            report, details, exit_code = audit.run_audit(root, ("mamba2",), require_full_length_scope=False)

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["coverage"]["selected_mask_rows"], 1)
        self.assertIn("missing_or_invalid_power_iteration_count", details[0]["issues"])

    def test_archived_exact_values_require_production_pass_flag(self):
        transitions = np.full(8, 0.8)
        row = base_row("missing-pass-flag", "native") | exact_fields(transitions)
        row.pop("mask_exact_validation_passed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_unit(root, [row])
            report, details, exit_code = audit.run_audit(root, ("mamba2",), require_full_length_scope=False)

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("archived_exact_fields_lack_validation_pass_flag", details[0]["issues"])

    def test_strict_independent_mode_rejects_archival_only_coverage(self):
        transitions = np.full(8, 0.8)
        row = base_row("archival-strict", "native") | exact_fields(transitions)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_unit(root, [row])
            report, _, exit_code = audit.run_audit(
                root,
                ("mamba2",),
                require_independent=True,
                require_full_length_scope=False,
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(report["status"], "INCOMPLETE_INDEPENDENT_COVERAGE")

    def test_production_scope_rejects_missing_1024_rows(self):
        transitions = np.full(8, 0.8)
        row = base_row("short-only", "native") | exact_fields(transitions)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_unit(root, [row])
            with self.assertRaisesRegex(audit.AuditInputError, "length 1024"):
                audit.run_audit(root, ("mamba2",))


if __name__ == "__main__":
    unittest.main()
