"""CodeBERT baseline, scored on the same queries and gallery as the TF-IDF views.

    python src/baselines/codebert.py embed [--limit N]
    python src/baselines/codebert.py evaluate
"""
import argparse
import hashlib
import io
import json
import sys
import tokenize
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from codenet_pipeline.evaluate import retrieval_metrics

MODEL = "microsoft/codebert-base"
CHUNK = 4096


def normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def problem_of(program_id):
    return program_id.split("/", 1)[0]


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    root = (path.parent / config["project_root"]).resolve()
    return {"source": root / config["source_root"], "output": root / config["output_root"],
            "seed": config["seed"]}


def load_memberships(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"query_ids": sorted(m["query_ids"]), "gallery_ids": sorted(m["gallery_ids"])}


def load_text(path):
    raw = Path(path).read_bytes()
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding)
    except (SyntaxError, LookupError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    return tokenizer, model, device


def embed(codes, tokenizer, model, device, batch_size=32, max_length=512):
    import torch

    order = np.argsort([len(c) for c in codes])  # similar lengths together, less padding
    mean = np.zeros((len(codes), model.config.hidden_size), dtype=np.float32)
    cls = np.zeros_like(mean)
    with torch.no_grad():
        for start in range(0, len(codes), batch_size):
            idx = order[start:start + batch_size]
            batch = tokenizer([codes[i] for i in idx], truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean[idx] = ((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
            cls[idx] = hidden[:, 0].float().cpu().numpy()
            if start % (batch_size * 50) == 0:
                print(f"  {start}/{len(codes)}", flush=True)
    return normalize(mean), normalize(cls)


def nearest(query_vec, ids, vectors, k=10):
    scores = vectors @ query_vec
    top = np.argsort(-scores)[:k]
    return [(str(ids[i]), float(scores[i])) for i in top]


def evaluate_embeddings(query_vecs, query_problems, gallery_vecs, gallery_problems, seed,
                        ks=(1, 5, 10), batch=128):
    query_problems = np.asarray(query_problems)
    # shuffle the gallery once so ties are not broken by problem order, same as the pipeline
    perm = np.random.default_rng(seed).permutation(len(gallery_problems))
    gallery = np.ascontiguousarray(gallery_vecs[perm], dtype=np.float32)
    gallery_problems = np.asarray(gallery_problems)[perm]
    ks = tuple(k for k in ks if k <= len(gallery_problems))
    rows, skipped = [], 0
    for start in range(0, len(query_vecs), batch):
        scores = query_vecs[start:start + batch] @ gallery.T
        for offset, row in enumerate(scores):
            ranking = np.argsort(-row, kind="stable")
            relevant = gallery_problems[ranking] == query_problems[start + offset]
            if not relevant.any():
                skipped += 1
                continue
            rows.append(retrieval_metrics(relevant, ks))
    means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    means["mrr"] = means.pop("reciprocal_rank")
    means["map"] = means.pop("average_precision")
    means["queries"] = len(rows)
    means["skipped_queries"] = skipped
    return means


def format_row(name, m, ks=(1, 5, 10)):
    cols = [f"P@{k}={m[f'precision@{k}']:.4f}" for k in ks]
    cols += [f"R@{k}={m[f'recall@{k}']:.4f}" for k in ks]
    cols += [f"MRR={m['mrr']:.4f}", f"MAP={m['map']:.4f}"]
    return f"{name}: " + ", ".join(cols)


def run_embed(args):
    cfg = read_config(args.config)
    m = load_memberships(cfg["output"] / "memberships.json")
    ids = m["query_ids"] + m["gallery_ids"]
    if args.limit:
        ids = sorted(ids)[:args.limit]
    out = Path(args.out_dir)
    chunks = out / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)

    # chunks from a run on a different set of programs (e.g. --limit) must not be reused
    digest = hashlib.sha1("\n".join(ids).encode()).hexdigest()
    stamp = chunks / "ids.sha1"
    if stamp.exists() and stamp.read_text() != digest:
        sys.exit(f"{chunks} has chunks for a different set of programs, use another --out-dir")
    stamp.write_text(digest)

    tokenizer, model, device = load_encoder()
    n_chunks = -(-len(ids) // CHUNK)
    print(f"device: {device}, programs: {len(ids)}")
    for n in range(n_chunks):
        part = chunks / f"{n:04d}.npz"
        if part.exists():
            continue
        print(f"chunk {n + 1}/{n_chunks}", flush=True)
        codes = [load_text(cfg["source"] / f"{i}.py") for i in ids[n * CHUNK:(n + 1) * CHUNK]]
        mean, cls = embed(codes, tokenizer, model, device)
        tmp = part.with_suffix(".tmp.npz")
        np.savez(tmp, mean=mean, cls=cls)
        tmp.replace(part)

    parts = [np.load(chunks / f"{n:04d}.npz") for n in range(n_chunks)]
    np.save(out / "ids.npy", np.array(ids, dtype=str))
    np.save(out / "mean.npy", np.concatenate([p["mean"] for p in parts]))
    np.save(out / "cls.npy", np.concatenate([p["cls"] for p in parts]))
    print(f"saved to {out}")


def run_evaluate(args):
    cfg = read_config(args.config)
    m = load_memberships(cfg["output"] / "memberships.json")
    out = Path(args.out_dir)
    pos = {i: n for n, i in enumerate(np.load(out / "ids.npy"))}
    q = [i for i in m["query_ids"] if i in pos]
    g = [i for i in m["gallery_ids"] if i in pos]
    print(f"{len(q)} queries, {len(g)} gallery programs")

    lines = [f"CodeBERT ({MODEL}), cosine similarity, same queries and gallery as the TF-IDF views"]
    summary = {}
    for name in ("mean", "cls"):
        vecs = np.load(out / f"{name}.npy")
        res = evaluate_embeddings(vecs[[pos[i] for i in q]], [problem_of(i) for i in q],
                                  vecs[[pos[i] for i in g]], [problem_of(i) for i in g], cfg["seed"])
        summary[name] = res
        lines.append(format_row(f"CodeBERT {name}", res) +
                     f"  [{res['queries']} queries, {res['skipped_queries']} skipped]")
        print(lines[-1], flush=True)
    (out / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'RESULTS.txt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("embed", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=str(REPO / "configs" / "full_benchmark.json"))
        p.add_argument("--out-dir", default=str(REPO / "artifacts" / "codebert"))
        if name == "embed":
            p.add_argument("--limit", type=int, default=0, help="only the first N programs (smoke test)")
    args = parser.parse_args()
    run_embed(args) if args.cmd == "embed" else run_evaluate(args)


if __name__ == "__main__":
    main()
