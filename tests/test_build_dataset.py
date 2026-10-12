"""Builds a tiny fake CodeNet tree and checks the filters. Run: python -m unittest discover tests"""
import csv
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
import build_dataset as bd  # noqa: E402

SHARED = "print(1+2+3+4+5+6+7+8+9+10+11+12)\n"  # same text under two problems


def make_fixture(root: Path) -> None:
    meta = root / "metadata"
    meta.mkdir(parents=True)
    for p in range(1, 5):
        pid = f"p{p:05d}"
        d = root / "data" / pid / "Python"
        d.mkdir(parents=True)
        rows = []
        for i in range(60):
            sid = f"s{p}{i:04d}"
            if i % 10 == 0:
                code = f'print "py2 style {i}"\nx = {i}\n' + "# pad\n" * 8   # does not parse
            elif i == 5:
                code = (d / f"s{p}0004.py").read_text()                      # duplicate of i == 4
            elif i == 7 and p in (1, 2):
                code = SHARED
            else:
                code = f"n = int(input())\nprint(n * {i} + {p})\n" + "# pad\n" * 6
            (d / f"{sid}.py").write_text(code)
            status = "Wrong Answer" if i % 7 == 3 else "Accepted"
            user = "heavy" if i % 6 == 1 else f"u{i}"                        # heavy submits a lot
            rows.append({"submission_id": sid, "problem_id": pid, "user_id": user,
                         "language": "Python", "status": status})
        with (meta / f"{pid}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def args_for(tmp: Path, out: Path, n_problems=4, seed=0) -> Namespace:
    return Namespace(root=str(tmp / "raw"), metadata_dir=None, out=str(out), n_problems=n_problems,
                     per_problem=30, max_per_user=3, min_chars=40, max_chars=6000,
                     n_test=5, n_val=5, min_corpus=5, seed=seed)


class BuildTest(unittest.TestCase):
    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_fixture(tmp / "raw")
            out = tmp / "out" / "dataset.parquet"
            bd.build(args_for(tmp, out))
            df = pd.read_parquet(out)
            card = (out.parent / "data_card.json").read_text()

            self.assertTrue(df["id"].is_unique)
            self.assertFalse(df["code"].str.contains('print "py2').any())
            self.assertFalse((df["code"] == SHARED).any())
            self.assertEqual(set(df["split"]), {"test", "val", "corpus"})
            for _, g in df.groupby("problem_id"):
                self.assertEqual((g["split"] == "test").sum(), 5)
                self.assertEqual((g["split"] == "val").sum(), 5)
                self.assertGreaterEqual((g["split"] == "corpus").sum(), 5)
                self.assertLessEqual(g["user_id"].value_counts().max(), 3)
                queries = set(g[g["split"] != "corpus"]["user_id"])
                corpus = set(g[g["split"] == "corpus"]["user_id"])
                self.assertFalse(queries & corpus)                            # no same-user leak
            for key in ("syntax_error", "not_accepted", "duplicate_in_problem", "dropped_user_cap"):
                self.assertIn(key, card)

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_fixture(tmp / "raw")
            frames = []
            for name in ("a", "b"):
                out = tmp / name / "dataset.parquet"
                bd.build(args_for(tmp, out, n_problems=3, seed=1))
                frames.append(pd.read_parquet(out))
            pd.testing.assert_frame_equal(frames[0], frames[1])


if __name__ == "__main__":
    unittest.main()
