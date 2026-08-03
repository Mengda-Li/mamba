"""Model-free tests for deterministic Mamba transition replay identities."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import replay_mamba_memory_transitions as replay


def sample(length: int = 64) -> dict:
    return {
        "checkpoint": "mamba2",
        "document_id": "doc-0",
        "article_index": 7,
        "sample_offset": 0,
        "input_condition": "native-prefix",
        "length": length,
        "token_ids": list(range(length)),
    }


class ReplayMambaTransitionsTests(unittest.TestCase):
    def test_observation_ids_are_variant_specific_and_stable(self):
        native = replay.stable_observation_id(sample(), 31, 16, "native")
        shuffled = replay.stable_observation_id(sample(), 31, 16, "transition_shuffled")
        self.assertEqual(native, replay.stable_observation_id(sample(), 31, 16, "native"))
        self.assertNotEqual(native, shuffled)
        self.assertEqual(len(native), 32)

    def test_archive_resume_requires_exact_keys_and_completed_metrics(self):
        item = sample()
        keys = replay.expected_keys(item, 0)
        metric_ids = {key.removeprefix("transition-") for key in keys}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unit.npz"
            np.savez_compressed(path, **{key: np.ones(item["length"]) for key in keys})
            self.assertTrue(replay.valid_archive(path, keys, metric_ids))
            self.assertFalse(replay.valid_archive(path, keys, set()))
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files if key != sorted(keys)[0]}
            np.savez_compressed(path, **arrays)
            self.assertFalse(replay.valid_archive(path, keys, metric_ids))

    def test_selected_samples_keep_only_required_protocol_rows(self):
        rows = [
            sample(64),
            sample(768),
            sample(1024),
            sample(64) | {"checkpoint": "retnet"},
            sample(64) | {"input_condition": "explicit-boundary"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            selected = replay.selected_samples(path)
        self.assertEqual([row["length"] for row in selected], [64, 1024])

    def test_pending_units_skip_complete_archives(self):
        complete = sample(64)
        pending = sample(128)
        layers = (0,)
        complete_keys = replay.expected_keys(complete, 0)
        pending_keys = replay.expected_keys(pending, 0)
        metric_ids = {
            key.removeprefix("transition-")
            for key in complete_keys | pending_keys
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = replay.archive_path(output, complete, 0)
            path.parent.mkdir(parents=True)
            np.savez_compressed(
                path,
                **{key: np.ones(complete["length"]) for key in complete_keys},
            )
            units = replay.pending_units(
                [complete, pending],
                output,
                metric_ids,
                layers,
            )
        self.assertEqual(units, [("doc-0", 128)])

    def test_isolated_worker_command_forwards_arguments(self):
        command = replay.worker_command(
            "/venv/bin/python",
            Path("/code/replay.py"),
            Path("/data/manifest.jsonl"),
            Path("/data/results"),
            "cuda:0",
            ("document with spaces", 512),
        )
        self.assertEqual(command[0:2], ["/venv/bin/python", "/code/replay.py"])
        self.assertEqual(command[command.index("--manifest") + 1], "/data/manifest.jsonl")
        self.assertEqual(command[command.index("--out-dir") + 1], "/data/results")
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(command[command.index("--document-id") + 1], "document with spaces")
        self.assertEqual(command[command.index("--length") + 1], "512")
        self.assertIn("--worker", command)

    def test_isolated_runner_stops_on_worker_failure(self):
        calls = []

        def failing_runner(command, *, check, env):
            calls.append((command, check, env))
            raise subprocess.CalledProcessError(9, command)

        with self.assertRaises(subprocess.CalledProcessError):
            replay.run_isolated_units(
                [("doc-0", 64), ("doc-1", 128)],
                "/venv/bin/python",
                Path("/code/replay.py"),
                Path("/data/manifest.jsonl"),
                Path("/data/results"),
                "cuda",
                runner=failing_runner,
            )
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1])
        self.assertEqual(
            calls[0][2]["PYTORCH_CUDA_ALLOC_CONF"],
            "expandable_segments:True",
        )


if __name__ == "__main__":
    unittest.main()
