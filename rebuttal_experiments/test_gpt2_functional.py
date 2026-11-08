"""Persistence tests and a no-download tiny GPT-2 end-to-end validation.

Run on the GPU server with its existing torch/transformers environment:
  python -m unittest discover -s rebuttal_experiments -p test_gpt2_functional.py -v
"""

from __future__ import annotations

from repo_paths import REPO_ROOT, ARTIFACT_ROOT

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_gpt2_functional as runner


class ManifestAndResumeTests(unittest.TestCase):
    def test_document_shards_are_disjoint_and_completeness_checked(self):
        source = [dict(checkpoint="gpt2", document_id=d, input_condition="native-prefix",
                       length=n, token_ids=list(range(n)))
                  for d in ("doc-z", "doc-a", "doc-b") for n in (4, 8)]
        args = SimpleNamespace(document_start=0, document_stop=2, conditions="native-prefix", lengths=(4, 8))
        first = runner.select_samples(source, args)
        self.assertEqual({s["document_id"] for s in first}, {"doc-a", "doc-b"})
        args.document_start, args.document_stop = 2, 3
        second = runner.select_samples(source, args)
        self.assertFalse({s["document_id"] for s in first} & {s["document_id"] for s in second})
        with self.assertRaises(ValueError):
            runner.select_samples([s for s in source if not (s["document_id"] == "doc-z" and s["length"] == 8)], args)

    def test_resume_requires_metrics_and_validation_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {"metrics": root / "metrics" / "s.jsonl", "validation": root / "validation" / "s.json"}
            for path in files.values():
                runner.atomic_text(path, "{}\n")
            marker = {"sample_id": "s", "config_id": "c"}
            for kind, path in files.items():
                marker[kind + "_path"] = str(path.relative_to(root))
                marker[kind + "_sha256"] = runner.file_hash(path)
            runner.atomic_json(root / "completed" / "s.json", marker)
            self.assertTrue(runner.completed(root, "s", "c"))
            self.assertFalse(runner.completed(root, "s", "other"))
            files["metrics"].write_text("corrupt\n")
            self.assertFalse(runner.completed(root, "s", "c"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "Requires server torch/transformers; no package download")
class TinyGpt2FunctionalTests(unittest.TestCase):
    def test_pre_projection_ablation_matches_native_head_mask_and_preserves_bias(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        runner.torch = torch
        helpers = REPO_ROOT
        sys.path.insert(0, str(helpers))
        import gpt2_experiments as helper

        torch.manual_seed(31)
        config = GPT2Config(vocab_size=31, n_positions=16, n_ctx=16, n_embd=16, n_layer=2,
                            n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).float().eval()
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.bias.copy_(torch.randn_like(block.attn.c_proj.bias) * .05)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        sample = dict(document_id="tiny", article_index=0, sample_offset=0,
                      input_condition="native-prefix", length=8, token_ids=[1, 3, 5, 8, 6, 7, 4, 2])
        args = SimpleNamespace(device="cpu", svd_device="cpu", layers=(0, 1), heads="all",
            revision="random-local-test", modes="ablate,sink-neutralized", reconstruction_tolerance=2e-5,
            zero_logits_tolerance=2e-5, zero_loss_tolerance=1e-6)
        with torch.inference_mode():
            rows, checks = runner.analyze_sample(model, helper, sample, args, "tiny-test")
            ids = torch.tensor(sample["token_ids"]).unsqueeze(0)
            for row in rows:
                self.assertTrue(math.isfinite(row["delta_loss"]))
                self.assertEqual(row["loss_token_count"], 7)
                self.assertTrue(row["causality_preserved"])
                if row["intervention"] == "ablate":
                    mask = torch.ones(2, 2)
                    mask[row["layer"], row["head_or_channel"]] = 0
                    logits = model(input_ids=ids, head_mask=mask, use_cache=False).logits
                    expected_sum, count = runner.nll(logits, ids)
                    self.assertAlmostEqual(row["intervention_loss"], expected_sum / count, places=6)
        self.assertEqual(len(rows), 8)
        self.assertLessEqual(checks["zero_logits_max_abs_error"], args.zero_logits_tolerance)
        self.assertTrue(all(torch.equal(before[name], p) for name, p in model.named_parameters()))
        self.assertTrue(all(not module._forward_pre_hooks for block in model.transformer.h
                            for module in (block, block.attn.c_proj)))


if __name__ == "__main__":
    unittest.main()
