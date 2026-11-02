import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .vectorize import FEATURE_FIELDS

VIEW_LABELS = {"lexical_raw": "Raw tokens", "lexical_masked": "Masked tokens", "ast_features": "AST"}


def retrieval_metrics(relevant: np.ndarray, ks: tuple[int, ...]) -> dict:
    hits = np.flatnonzero(relevant) + 1
    if not len(hits):
        raise ValueError("Each query must have relevant gallery candidates")
    if any(k < 1 or k > len(relevant) for k in ks):
        raise ValueError("Cutoffs must be between 1 and the gallery size")
    metrics = {}
    for k in ks:
        found = np.count_nonzero(hits <= k)
        metrics[f"precision@{k}"] = float(found / k)
        metrics[f"recall@{k}"] = float(found / len(hits))
    metrics["reciprocal_rank"] = float(1 / hits[0])
    metrics["average_precision"] = float(np.mean(np.arange(1, len(hits) + 1) / hits))
    return metrics


def evaluate_matrices(matrix_dir: Path, memberships: dict, problems: dict, seed: int) -> tuple[dict, list]:
    started = time.perf_counter()
    row_ids = json.loads((matrix_dir / "program_ids.json").read_text())
    fit_ids = json.loads((matrix_dir / "fit_ids.json").read_text())
    query_ids = sorted(memberships["query_ids"])
    gallery_ids = sorted(memberships["gallery_ids"])
    combined = query_ids + gallery_ids
    if (not query_ids or not gallery_ids or len(set(combined)) != len(combined)
            or len(set(row_ids)) != len(row_ids) or set(row_ids) != set(combined)
            or sorted(fit_ids) != gallery_ids):
        raise ValueError("Matrix rows and fitted gallery must match disjoint query/gallery memberships")

    # [Shuffle once so stable score sorting does not break ties by problem ID.]
    permutation = np.random.default_rng(seed).permutation(len(gallery_ids))
    gallery_ids = [gallery_ids[i] for i in permutation]
    index = {program_id: i for i, program_id in enumerate(row_ids)}
    query_rows = [index[p] for p in query_ids]
    gallery_rows = [index[p] for p in gallery_ids]
    query_problems = np.array([problems[p] for p in query_ids])
    gallery_problems = np.array([problems[p] for p in gallery_ids])
    ks = tuple(k for k in (1, 5, 10) if k <= len(gallery_ids))
    summary = {
        "query_count": len(query_ids), "gallery_count": len(gallery_ids), "ks": list(ks),
        "seed": seed, "scope": memberships["scope"], "scoring": "cosine similarity",
        "relevance": "same problem_id; acceptance and problem-family verification pending",
        "ties": "fixed seeded gallery permutation, then stable descending score sort",
        "averaging": "arithmetic mean over queries; MRR and MAP use the full gallery ranking",
        "views": {}, "view_elapsed_seconds": {},
    }
    per_query = []
    for view in FEATURE_FIELDS:
        view_started = time.perf_counter()
        matrix = sparse.load_npz(matrix_dir / f"{view}.npz")
        if matrix.shape[0] != len(row_ids):
            raise ValueError(f"{view}: matrix row count does not match program_ids")
        gallery = matrix[gallery_rows].T.tocsc()
        view_rows = []
        for start in range(0, len(query_rows), 128):
            scores = (matrix[query_rows[start:start + 128]] @ gallery).toarray()
            for offset, row_scores in enumerate(scores):
                q = start + offset
                ranking = np.argsort(-row_scores, kind="stable")
                relevant = gallery_problems[ranking] == query_problems[q]
                metrics = retrieval_metrics(relevant, ks)
                view_rows.append({"view": view, "query_id": query_ids[q],
                                  "relevant_gallery_count": int(relevant.sum()), **metrics})
            completed = min(start + 128, len(query_rows))
            if len(query_rows) >= 1000 and (completed % 512 == 0 or completed == len(query_rows)):
                print(f"Evaluated {view}: {completed:,}/{len(query_rows):,} queries", flush=True)
        means = {name: float(np.mean([row[name] for row in view_rows])) for name in metrics}
        means["mrr"] = means.pop("reciprocal_rank")
        means["map"] = means.pop("average_precision")
        summary["views"][view] = means
        summary["view_elapsed_seconds"][view] = round(time.perf_counter() - view_started, 3)
        per_query.extend(view_rows)
        del matrix, gallery, scores
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return summary, per_query


def format_metrics(summary: dict) -> list[str]:
    keys = [f"{metric}@{k}" for metric in ("precision", "recall") for k in summary["ks"]]
    keys += ["mrr", "map"]
    labels = [key.replace("precision", "P").replace("recall", "R").upper() for key in keys]
    lines = []
    for view in FEATURE_FIELDS:
        values = ", ".join(f"{label}={summary['views'][view][key]:.4f}" for key, label in zip(keys, labels))
        lines.append(f"{VIEW_LABELS[view]}: {values}")
    return lines
