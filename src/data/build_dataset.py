#!/usr/bin/env python3
"""Prepar the CodeNet dataset and split it into test, validation, and corpus.
Run with --root and optionally --metadata-dir.
Without metadata, acceptance status and actual user IDs cannot be checked.
Queries come only from users with a single program in that problem, so a query 
never has another program by the same user in the corpus.
"""
import argparse
import ast
import csv
import hashlib
import json
import random
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROBLEM_RE = re.compile(r"^p\d{5}$")


def find_problem_dirs(root: Path) -> list[Path]:
    for cand in (root / "data", root):
        if cand.is_dir():
            dirs = sorted(d for d in cand.iterdir() if d.is_dir() and PROBLEM_RE.match(d.name))
            if dirs:
                return dirs
    raise SystemExit(f"No problem directories (p00001, ...) found under {root}")


def find_metadata_dir(root: Path, given: str | None) -> Path | None:
    if given:
        return Path(given).expanduser()
    for cand in (root / "metadata", root.parent / "metadata"):
        if cand.is_dir():
            return cand
    return None


def load_metadata(meta_dir: Path | None, problem_id: str) -> dict:
    """Read the user, status, and language for each submission."""
    if meta_dir is None:
        return {}
    path = meta_dir / f"{problem_id}.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            out[row["submission_id"]] = (
                row.get("user_id", ""),
                row.get("status", ""),
                row.get("language", ""),
            )
    return out


def parses(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False


def norm_hash(code: str) -> str:
    return hashlib.sha1(" ".join(code.split()).encode("utf-8")).hexdigest()


def candidates_for_problem(pdir: Path, meta: dict, args, stages: Counter) -> list[dict]:
    seen = set()
    out = []
    for f in sorted(pdir.rglob("*.py")):
        stages["files_found"] += 1
        sid = f.stem
        if meta:
            if sid not in meta:
                stages["no_metadata_row"] += 1
                continue
            user, status, lang = meta[sid]
            if status != "Accepted":
                stages["not_accepted"] += 1
                continue
            if lang and lang != "Python":
                stages["not_python"] += 1
                continue
        else:
            user = sid  # Use the submission ID when the user is unknown.
        code = f.read_text(encoding="utf-8", errors="ignore")
        if not (args.min_chars <= len(code) <= args.max_chars):
            stages["bad_length"] += 1
            continue
        if not parses(code):
            stages["syntax_error"] += 1
            continue
        h = norm_hash(code)
        if h in seen:
            stages["duplicate_in_problem"] += 1
            continue
        seen.add(h)
        out.append({"id": f"{pdir.name}_{sid}", "problem_id": pdir.name,
                    "user_id": user or sid, "code": code, "hash": h})
    return out


def cap_per_user_and_problem(cands: list[dict], args, rng: random.Random, stages: Counter) -> list[dict]:
    rng.shuffle(cands)
    per_user = Counter()
    sel = []
    for c in cands:
        if len(sel) >= args.per_problem:
            break
        if per_user[c["user_id"]] >= args.max_per_user:
            stages["dropped_user_cap"] += 1
            continue
        per_user[c["user_id"]] += 1
        sel.append(c)
    return sel


def singleton_users(rows: list[dict]) -> list[dict]:
    """Programs whose user has no other program in this problem."""
    n = Counter(c["user_id"] for c in rows)
    return [c for c in rows if n[c["user_id"]] == 1]


def too_small(rows: list[dict], args, min_per_problem: int) -> bool:
    return len(rows) < min_per_problem or len(singleton_users(rows)) < args.n_test + args.n_val


def build(args) -> None:
    rng = random.Random(args.seed)
    root = Path(args.root).expanduser()
    meta_dir = find_metadata_dir(root, args.metadata_dir)
    problems = find_problem_dirs(root)
    rng.shuffle(problems)
    min_per_problem = args.n_test + args.n_val + args.min_corpus

    stages: Counter = Counter()
    selected: dict[str, list[dict]] = {}
    for pdir in problems:
        if len(selected) >= args.n_problems:
            break
        stages["problems_visited"] += 1
        cands = candidates_for_problem(pdir, load_metadata(meta_dir, pdir.name), args, stages)
        sel = cap_per_user_and_problem(cands, args, rng, stages)
        if too_small(sel, args, min_per_problem):
            stages["problems_too_small"] += 1
            continue
        selected[pdir.name] = sel

    # Remove code shared by different problems.
    hash_problems = defaultdict(set)
    for pid, sel in selected.items():
        for c in sel:
            hash_problems[c["hash"]].add(pid)
    rows = []
    for pid, sel in selected.items():
        keep = [c for c in sel if len(hash_problems[c["hash"]]) == 1]
        stages["dropped_cross_problem_duplicate"] += len(sel) - len(keep)
        if too_small(keep, args, min_per_problem):
            stages["problems_dropped_after_dedupe"] += 1
            continue
        queries = singleton_users(keep)
        rng.shuffle(queries)
        test_ids = {c["id"] for c in queries[:args.n_test]}
        val_ids = {c["id"] for c in queries[args.n_test:args.n_test + args.n_val]}
        for c in keep:
            if c["id"] in test_ids:
                split = "test"
            elif c["id"] in val_ids:
                split = "val"
            else:
                split = "corpus"
            rows.append({"id": c["id"], "problem_id": pid, "user_id": c["user_id"],
                         "code": c["code"], "split": split, "n_chars": len(c["code"])})
    if not rows:
        raise SystemExit("No programs survived filtering. Check --root and the filter settings.")

    df = pd.DataFrame(rows).sort_values(["problem_id", "split", "id"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_card(df, stages, args, out.parent, meta_dir is not None)
    print(f"Wrote {len(df)} programs from {df['problem_id'].nunique()} problems to {out}")


def write_card(df: pd.DataFrame, stages: Counter, args, out_dir: Path, has_meta: bool) -> None:
    per_problem = df.groupby("problem_id").size()
    q = df["n_chars"].quantile([0.05, 0.5, 0.95])
    final = {
        "problems": int(df["problem_id"].nunique()),
        "programs": int(len(df)),
        "by_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "programs_per_problem": {"min": int(per_problem.min()),
                                 "median": float(per_problem.median()),
                                 "max": int(per_problem.max())},
        "chars": {"p05": float(q[0.05]), "median": float(q[0.5]), "p95": float(q[0.95])},
        "distinct_users": int(df["user_id"].nunique()),
    }
    card = {"args": vars(args), "metadata_used": has_meta, "stages": dict(stages), "final": final}
    (out_dir / "data_card.json").write_text(json.dumps(card, indent=2, default=str))
    lines = ["# Data card", ""]
    if not has_meta:
        lines += ["No metadata found: status and user filters were skipped.", ""]
    lines += ["## Filtering stages", ""] + [f"- {k}: {v}" for k, v in stages.items()]
    lines += ["", "## Final dataset", "",
              f"- problems: {final['problems']}",
              f"- programs: {final['programs']}",
              f"- by split: {final['by_split']}",
              f"- programs per problem (min / median / max): "
              f"{final['programs_per_problem']['min']} / {final['programs_per_problem']['median']} / "
              f"{final['programs_per_problem']['max']}",
              f"- characters (5th / median / 95th percentile): "
              f"{final['chars']['p05']:.0f} / {final['chars']['median']:.0f} / {final['chars']['p95']:.0f}",
              f"- distinct users: {final['distinct_users']}", ""]
    (out_dir / "data_card.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="extracted CodeNet dir (contains p00001, ... or data/)")
    p.add_argument("--metadata-dir", default=None, help="CodeNet metadata dir (auto-detected if next to root)")
    p.add_argument("--out", default="data/processed/dataset.parquet")
    p.add_argument("--n-problems", type=int, default=200)
    p.add_argument("--per-problem", type=int, default=100)
    p.add_argument("--max-per-user", type=int, default=3)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--n-val", type=int, default=10)
    p.add_argument("--min-corpus", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    build(p.parse_args())


if __name__ == "__main__":
    main()
