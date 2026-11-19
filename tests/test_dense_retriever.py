"""Tests for the CodeBERT baseline that run without the model."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "baselines"))
import codebert  # noqa: E402


class NearestTest(unittest.TestCase):
    def test_finds_itself_first(self):
        rng = np.random.default_rng(0)
        vecs = codebert.normalize(rng.normal(size=(50, 16)).astype(np.float32))
        ids = [f"d{i}" for i in range(50)]
        for i in range(5):
            hits = codebert.nearest(vecs[i], ids, vecs, 3)
            self.assertEqual(hits[0][0], f"d{i}")
            self.assertAlmostEqual(hits[0][1], 1.0, places=4)
            scores = [s for _, s in hits]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_normalize(self):
        x = codebert.normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(x[0])), 1.0, places=5)
        self.assertTrue(np.isfinite(x).all())


class PipelineInputTest(unittest.TestCase):
    def test_problem_of(self):
        self.assertEqual(codebert.problem_of("p00012/s123456789"), "p00012")

    def test_load_text_normalizes_line_endings(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.py"
            path.write_bytes(b"x = 1\r\ny = 2\rz = 3\n")
            self.assertEqual(codebert.load_text(path), "x = 1\ny = 2\nz = 3\n")

    def test_load_text_honors_declared_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.py"
            path.write_bytes("# -*- coding: latin-1 -*-\ns = 'é'\n".encode("latin-1"))
            self.assertIn("é", codebert.load_text(path))

    def test_read_config_paths(self):
        cfg = codebert.read_config(REPO / "configs" / "full_benchmark.json")
        self.assertEqual(cfg["seed"], 881)
        self.assertEqual(cfg["output"], REPO / "artifacts" / "full_benchmark")
        self.assertEqual(cfg["source"], REPO / "src" / "data" / "Project_CodeNet_Python800")


try:
    import torch
except ImportError:  # the test is skipped where torch is missing
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class EmbedTest(unittest.TestCase):
    """embed() with a stand-in model keeps the input order."""

    class Tok:
        def __call__(self, texts, **kwargs):
            n = max(len(t) for t in texts)
            ids = torch.tensor([[float(len(t)), 1.0] + [0.0] * (n - 1) for t in texts])
            out = {"input_ids": ids, "attention_mask": torch.ones(len(texts), n + 1)}

            class Batch(dict):
                def to(self, device):
                    return self
            return Batch(out)

    class Model:
        def __init__(self):
            self.config = type("C", (), {"hidden_size": 3})()

        def __call__(self, input_ids, attention_mask):
            first = input_ids[:, 0:1]  # the program's length, so each program has a distinct vector
            hidden = torch.stack([first, first * 2 + 1, torch.ones_like(first)], dim=-1)
            hidden = hidden.expand(-1, attention_mask.shape[1], -1)
            return type("O", (), {"last_hidden_state": hidden})()

    def test_order(self):
        codes = ["a" * n for n in (5, 1, 9, 3, 7, 2, 8, 4)]
        batched = codebert.embed(codes, self.Tok(), self.Model(), "cpu", batch_size=3)
        for i, code in enumerate(codes):
            alone = codebert.embed([code], self.Tok(), self.Model(), "cpu")
            np.testing.assert_allclose(batched[0][i], alone[0][0], atol=1e-6)
            np.testing.assert_allclose(batched[1][i], alone[1][0], atol=1e-6)


class EvaluateTest(unittest.TestCase):
    def test_matches_pipeline_metrics_on_separable_clusters(self):
        rng = np.random.default_rng(1)
        centers = codebert.normalize(rng.normal(size=(3, 8)).astype(np.float32))
        gallery_problems = np.repeat(["a", "b", "c"], 10)
        gallery = codebert.normalize(centers[np.repeat(np.arange(3), 10)]
                                     + 0.01 * rng.normal(size=(30, 8)).astype(np.float32))
        query_problems = ["a", "b", "c"]
        queries = codebert.normalize(centers + 0.01 * rng.normal(size=(3, 8)).astype(np.float32))
        res = codebert.evaluate_embeddings(queries, query_problems, gallery, gallery_problems, seed=881)
        self.assertEqual(res["precision@1"], 1.0)
        self.assertEqual(res["precision@10"], 1.0)
        self.assertEqual(res["mrr"], 1.0)
        self.assertEqual(res["map"], 1.0)
        self.assertAlmostEqual(res["recall@10"], 1.0)

    def test_skips_queries_without_relevant_gallery_items(self):
        gallery = codebert.normalize(np.eye(4, dtype=np.float32))
        res = codebert.evaluate_embeddings(gallery[:2], ["a", "z"], gallery, ["a", "a", "b", "b"], seed=0)
        self.assertEqual(res["queries"], 1)
        self.assertEqual(res["skipped_queries"], 1)


if __name__ == "__main__":
    unittest.main()
