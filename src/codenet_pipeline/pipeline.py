from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .corpus import classify_record, iter_sources, load_metadata

MANIFEST_FIELDS = (
    "program_id", "problem_id", "submission_id", "source_path", "source_bytes",
    "metadata_status", "acceptance_status", "dialect_status", "eligibility_status",
    "status", "language", "original_language", "user_id", "exclusion_reason",
)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = read_json(path)
    if platform.python_version() != config["required_python"]:
        raise ValueError(f"Use Python {config['required_python']}; running {platform.python_version()}. "
                         "Changing the parser requires a new configuration and output directory.")
    for key in ("pilot_problems", "programs_per_problem", "queries_per_problem"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["queries_per_problem"] >= config["programs_per_problem"]:
        raise ValueError("Leave at least one retrieval candidate per problem")
    if not (0 < config["development_fraction"] < 1
            and 0 <= config["validation_fraction"] < 1
            and config["development_fraction"] + config["validation_fraction"] < 1):
        raise ValueError("Invalid problem partition fractions")
    root = (path.parent / config["project_root"]).resolve()
    config["_project_root"] = root
    config["_source"] = (root / config["source_root"]).resolve()
    config["_output"] = (root / config["output_root"]).resolve()
    config["_metadata"] = ((root / config["metadata_root"]).resolve()
                            if config.get("metadata_root") else None)
    return config


def current_environment() -> dict:
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "executable": sys.executable, "platform": platform.platform(),
    }


def snapshot(config: dict) -> None:
    out = config["_output"]
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in config.items() if not k.startswith("_")}
    existing = out / "configuration.json"
    if existing.exists() and read_json(existing) != settings:
        raise ValueError("Output directory has a different configuration; choose a new output_root")
    write_json(existing, settings)
    # [Keep the original run environment; later stages have their own records]
    if not (out / "environment.json").exists():
        write_json(out / "environment.json", current_environment())


def record_stage(config: dict, stage: str, elapsed_seconds: float | None = None) -> None:
    inputs = {
        "inventory": [],
        "process": ["inventory.csv.gz"],
        "vectorize": ["features.jsonl.gz", "memberships.json", "provenance/process.json"],
        "evaluate": ["memberships.json", "processing_audit.csv", "matrices/program_ids.json",
                     "matrices/fit_ids.json", "matrices/lexical_raw.npz",
                     "matrices/lexical_masked.npz", "matrices/ast_features.npz"],
        "report": ["collection_summary.json", "processing_summary.json",
                   "matrices/vectorization_summary.json", "provenance/process.json",
                   "evaluation_summary.json", "processing_audit.csv", "memberships.json",
                   "per_query_metrics.csv"],
    }
    out = config["_output"]
    write_json(out / "provenance" / f"{stage}.json", {
        "stage": stage, "producer": current_environment(),
        "inputs": [name for name in inputs[stage] if (out / name).exists()],
        "elapsed_seconds": elapsed_seconds,
    })


def inventory(config: dict) -> dict:
    started = time.perf_counter()
    source_records = list(iter_sources(config["_source"]))
    lookup, metadata_summary = None, {"available": False}
    if config["_metadata"] is not None:
        lookup, metadata_summary = load_metadata(
            config["_metadata"], {r["problem_id"] for r in source_records},
            submission_ids=((r["problem_id"], r["submission_id"]) for r in source_records),
        )
        metadata_summary["available"] = True
    problems, acceptance, eligibility, dialect, matches = (Counter() for _ in range(5))
    source_bytes = 0
    path = config["_output"] / "inventory.csv.gz"
    with gzip.open(path.with_suffix(".gz.tmp"), "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for source in source_records:
            row = classify_record(source, lookup, assume_accepted=config["assume_accepted"])
            row["program_id"] = row["problem_id"] + "/" + row["submission_id"]
            writer.writerow(row)
            problems[row["problem_id"]] += 1
            acceptance[row["acceptance_status"]] += 1
            eligibility[row["eligibility_status"]] += 1
            dialect[row["dialect_status"]] += 1
            matches[row["metadata_status"]] += 1
            source_bytes += row["source_bytes"]
    path.with_suffix(".gz.tmp").replace(path)
    summary = {
        "programs": len(source_records), "problems": len(problems),
        "source_bytes": source_bytes, "programs_per_problem": dict(sorted(problems.items())),
        "acceptance_status": dict(acceptance), "eligibility_status": dict(eligibility),
        "dialect_status": dict(dialect), "metadata_status": dict(matches),
        "metadata_audit": metadata_summary,
        "acceptance_assumption": "Acceptance assumed pending metadata verification."
                                 if lookup is None and config["assume_accepted"] else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(config["_output"] / "collection_summary.json", summary)
    print(f"Inventoried {len(source_records):,} sources across {len(problems)} problems.", flush=True)
    return summary


def select_pilot(config: dict) -> list[dict]:
    groups = defaultdict(list)
    all_problems = set()
    with gzip.open(config["_output"] / "inventory.csv.gz", "rt", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            all_problems.add(row["problem_id"])
            if row["eligibility_status"] in {"candidate", "provisional_candidate"}:
                row["source_bytes"] = int(row["source_bytes"])
                groups[row["problem_id"]].append(row)
    shuffled = sorted(all_problems)
    random.Random(config["seed"]).shuffle(shuffled)
    dev_end = int(len(shuffled) * config["development_fraction"])
    val_end = dev_end + int(len(shuffled) * config["validation_fraction"])
    partitions = {
        "development": sorted(shuffled[:dev_end]),
        "validation": sorted(shuffled[dev_end:val_end]), "test": sorted(shuffled[val_end:]),
        "status": "provisional: identical-problem-family metadata has not been incorporated",
        "relevance_label": "original problem_id, provisional pending family verification",
        "seed": config["seed"],
    }
    full = config.get("full_corpus", False)
    candidates = [p for p in shuffled[:dev_end] if len(groups[p]) >= config["programs_per_problem"]]
    selected_problems = sorted(groups) if full else sorted(candidates[:config["pilot_problems"]])
    if not selected_problems:
        raise ValueError("No development problems have enough candidates for the configured pilot")
    chosen = []
    for problem in selected_problems:
        rows = sorted(groups[problem], key=lambda r: r["program_id"])
        random.Random(f"{config['seed']}:{problem}:sources").shuffle(rows)
        chosen.extend(rows if full else rows[:config["programs_per_problem"]])
    chosen.sort(key=lambda r: r["program_id"])
    partitions["pilot_problems"] = selected_problems
    write_json(config["_output"] / "problem_splits.json", partitions)
    write_json(config["_output"] / "pilot_selection.json", {
        "scope": "full corpus" if full else "development sample",
        "requested_problems": config["pilot_problems"], "actual_problems": len(selected_problems),
        "attempted_programs": len(chosen), "records": chosen,
    })
    return chosen


def conservative_tokens(row: dict) -> str:
    tokens = row.get("conservative_tokens")
    if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
        raise ValueError(f"{row['program_id']}: invalid conservative_tokens")
    return json.dumps(tokens, ensure_ascii=True, separators=(",", ":"))


def group_and_assign(records: list[dict], config: dict) -> tuple[list[dict], dict]:
    duplicate_groups = defaultdict(list)
    seen = set()
    for row in records:
        program_id = row.get("program_id")
        if not isinstance(program_id, str) or not program_id:
            raise ValueError("Each processed record needs a nonempty program_id")
        if program_id in seen:
            raise ValueError(f"Duplicate processed program_id: {program_id}")
        seen.add(program_id)
        if row["status"] == "ok":
            key = row["_conservative_key"] if "_conservative_key" in row else conservative_tokens(row)
            duplicate_groups[key].append(row)
        else:
            row["processing_disposition"] = row["status"]
    group_audit = []
    usable = []
    for members in duplicate_groups.values():
        members.sort(key=lambda row: row["program_id"])
        if len({r["problem_id"] for r in members}) > 1:
            for row in members:
                row["processing_disposition"] = "cross_problem_duplicate_conflict"
        else:
            members[0]["processing_disposition"] = "retained"
            usable.append(members[0])
            for row in members[1:]:
                row["processing_disposition"] = "duplicate_nonrepresentative"
        if len(members) > 1:
            group_audit.append({"program_ids": [r["program_id"] for r in members],
                                "cross_problem_conflict": len({r["problem_id"] for r in members}) > 1})
    group_audit.sort(key=lambda group: group["program_ids"])
    by_problem = defaultdict(list)
    for row in usable:
        by_problem[row["problem_id"]].append(row)
    queries, gallery, unused = [], [], []
    for problem, rows in sorted(by_problem.items()):
        rows.sort(key=lambda r: r["program_id"])
        random.Random(f"{config['seed']}:{problem}:roles").shuffle(rows)
        if len(rows) <= config["queries_per_problem"]:
            unused.extend(r["program_id"] for r in rows)
            continue
        queries.extend(r["program_id"] for r in rows[:config["queries_per_problem"]])
        gallery.extend(r["program_id"] for r in rows[config["queries_per_problem"]:])
    memberships = {
        "query_ids": sorted(queries), "gallery_ids": sorted(gallery), "unused_ids": sorted(unused),
        "scope": ("full corpus" if config.get("full_corpus") else "development sample")
                 + "; problem-family verification pending",
        "deduplication_scope": "conservative-token duplicates across all selected submissions",
        "duplicate_groups": group_audit,
    }
    return sorted(usable, key=lambda r: r["program_id"]), memberships


def distribution(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    def quantile(q):
        location = (len(ordered) - 1) * q
        lo, hi = int(location), min(int(location) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (location - lo)
    return {"count": len(values), "min": ordered[0], "median": quantile(.5),
            "p90": quantile(.9), "p95": quantile(.95), "max": ordered[-1]}


def process(config: dict) -> dict:
    from .features import extract_features
    started = time.perf_counter()
    selected = select_pilot(config)
    out = config["_output"]
    audit_fields = ("program_id", "problem_id", "source_path", "status", "processing_disposition",
                    "acceptance_status", "metadata_status", "dialect_status",
                    "source_bytes", "encoding", "loc", "token_count",
                    "ast_node_count", "ast_max_depth", "docstring_count", "error_type", "error_message")
    records = []
    warning_categories = Counter()
    warning_programs = 0
    pending = out / "features.pending.jsonl.gz"
    with gzip.open(pending, "wt", encoding="utf-8", compresslevel=1) as handle:
        for index, row in enumerate(selected, 1):
            try:
                raw = (config["_source"] / row["source_path"]).read_bytes()
                features = extract_features(raw, filename=row["source_path"])
            except OSError as error:
                features = {"status": "read_error", "error_type": type(error).__name__,
                            "error_message": str(error), "source_bytes": row["source_bytes"]}
            record = {**features, "program_id": row["program_id"], "problem_id": row["problem_id"],
                      "submission_id": row["submission_id"], "source_path": row["source_path"],
                      "acceptance_status": row["acceptance_status"], "metadata_status": row["metadata_status"],
                      "dialect_status": row["dialect_status"]}
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
            light = {key: record.get(key) for key in audit_fields if key != "processing_disposition"}
            if record["status"] == "ok":
                light["_conservative_key"] = conservative_tokens(record)
            records.append(light)
            warnings = record.get("parser_warnings", [])
            warning_programs += bool(warnings)
            warning_categories.update(w["category"] for w in warnings)
            if index % (1000 if config.get("full_corpus") else 100) == 0:
                print(f"Processed {index:,}/{len(selected):,} submissions", flush=True)
    del selected
    usable, memberships = group_and_assign(records, config)
    for row in records:
        row.pop("_conservative_key", None)
    dispositions = {row["program_id"]: row["processing_disposition"] for row in records}
    with gzip.open(pending, "rt", encoding="utf-8") as source, \
            gzip.open(out / "features.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as target:
        for line in source:
            record = json.loads(line)
            record["processing_disposition"] = dispositions[record["program_id"]]
            target.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    pending.unlink()
    write_json(out / "memberships.json", memberships)
    with (out / "processing_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    ok = [r for r in records if r["status"] == "ok"]
    retained_per_problem = Counter({row["problem_id"]: 0 for row in records})
    retained_per_problem.update(row["problem_id"] for row in usable)
    duplicate_groups = memberships["duplicate_groups"]
    summary = {
        "scope": "full corpus" if config.get("full_corpus") else "development sample",
        "attempted": len(records), "status_counts": dict(Counter(r["status"] for r in records)),
        "error_type_counts": dict(Counter(r["error_type"] for r in records if r["status"] != "ok")),
        "encoding_counts": dict(Counter(r["encoding"] for r in records if r["encoding"])),
        "parser_warning_programs": warning_programs, "parser_warning_counts": dict(warning_categories),
        "programs_with_docstrings": sum(r["docstring_count"] > 0 for r in ok),
        "docstrings_removed": sum(r["docstring_count"] for r in ok),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_group_members": sum(len(g["program_ids"]) for g in duplicate_groups),
        "cross_problem_duplicate_groups": sum(g["cross_problem_conflict"] for g in duplicate_groups),
        "retained_per_problem": dict(sorted(retained_per_problem.items())),
        "retained_per_problem_distribution": distribution(list(retained_per_problem.values())),
        "disposition_counts": dict(Counter(r["processing_disposition"] for r in records)),
        "problems_attempted": len({r["problem_id"] for r in records}),
        "unique_parseable_retained": len(usable), "query_count": len(memberships["query_ids"]),
        "gallery_count": len(memberships["gallery_ids"]), "unused_count": len(memberships["unused_ids"]),
        "distributions": {key: distribution([r[key] for r in ok])
                          for key in ("source_bytes", "loc", "token_count", "ast_node_count", "ast_max_depth")},
        "acceptance_status": dict(Counter(r["acceptance_status"] for r in records)),
        "parser": platform.python_version(),
        "coverage_note": "All selected submissions. Parsing checks syntax; acceptance and original dialect require metadata.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(out / "processing_summary.json", summary)
    examples = []
    for row in sorted(usable, key=lambda r: (r["token_count"], r["program_id"]))[:3]:
        raw = (config["_source"] / row["source_path"]).read_bytes()
        features = extract_features(raw, filename=row["source_path"])
        examples.append({"program_id": row["program_id"],
                         "source": raw.decode(row["encoding"]), "raw_tokens": features["raw_tokens"],
                         "masked_tokens": features["masked_tokens"],
                         "ast_features": features["ast_features"], "feature_version": features["feature_version"]})
    write_json(out / "worked_examples.json", examples)
    print(f"Extracted: {len(ok):,}; retained after deduplication: {len(usable):,}", flush=True)
    return summary


def vectorize(config: dict) -> dict:
    from .vectorize import FEATURE_FIELDS, fit_feature_matrices
    memberships = read_json(config["_output"] / "memberships.json")
    if not isinstance(memberships, dict):
        raise ValueError("Memberships must be an object")
    roles = {}
    assigned = set()
    for role in ("query_ids", "gallery_ids", "unused_ids"):
        ids = memberships.get(role, [] if role == "unused_ids" else None)
        if (not isinstance(ids, list)
                or any(not isinstance(value, str) or not value for value in ids)):
            raise ValueError(f"{role} must be a list of nonempty program IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{role} contains duplicate IDs")
        if assigned.intersection(ids):
            raise ValueError("Query, retrieval corpus and unused memberships must be disjoint")
        roles[role] = set(ids)
        assigned.update(ids)
    if not roles["gallery_ids"]:
        raise ValueError("Retrieval corpus is empty; no candidates remain")
    wanted = roles["query_ids"] | roles["gallery_ids"]
    seen = set()
    problems = {"query_ids": set(), "gallery_ids": set()}
    token_owners = {}
    records = []
    with gzip.open(config["_output"] / "features.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Feature record {line_number}: invalid JSON") from error
            program_id = row.get("program_id") if isinstance(row, dict) else None
            if not isinstance(program_id, str) or not program_id:
                raise ValueError(f"Feature record {line_number}: invalid program_id")
            if program_id in seen:
                raise ValueError(f"Duplicate feature record: {program_id}")
            seen.add(program_id)
            if program_id not in assigned:
                continue
            if row.get("status") != "ok" or row.get("processing_disposition") != "retained":
                raise ValueError(f"{program_id}: membership references a failed or excluded submission")
            if program_id not in wanted:
                continue
            problem_id = row.get("problem_id")
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError(f"{program_id}: invalid problem_id")
            tokens = conservative_tokens(row)
            if tokens in token_owners:
                raise ValueError(f"Duplicate conservative_tokens in query/retrieval corpus: {token_owners[tokens]} and {program_id}")
            token_owners[tokens] = program_id
            role = "query_ids" if program_id in roles["query_ids"] else "gallery_ids"
            problems[role].add(problem_id)
            if not config.get("stream_features"):
                records.append({"program_id": program_id, **{field: row.get(field) for field in FEATURE_FIELDS}})
    missing = assigned.difference(seen)
    if missing:
        raise ValueError(f"Membership IDs missing from feature records: {sorted(missing)[:5]}")
    missing_problems = problems["query_ids"].difference(problems["gallery_ids"])
    if missing_problems:
        raise ValueError(f"Query problems have no retrieval candidates: {sorted(missing_problems)[:5]}")
    del token_owners
    if config.get("stream_features"):
        from .vectorize import fit_feature_matrices_from_file
        summary = fit_feature_matrices_from_file(
            config["_output"] / "features.jsonl.gz", config["_output"] / "matrices",
            memberships["gallery_ids"], sorted(wanted))
    else:
        summary = fit_feature_matrices(records, config["_output"] / "matrices",
                                       memberships["gallery_ids"], sorted(wanted))
    for name in ("evaluation_summary.json", "per_query_metrics.csv", "provenance/evaluate.json"):
        (config["_output"] / name).unlink(missing_ok=True)
    print(f"Saved 3 feature matrices, {len(wanted):,} rows each", flush=True)
    return summary


def evaluate(config: dict) -> dict:
    from .evaluate import evaluate_matrices, format_metrics
    out = config["_output"]
    with (out / "processing_audit.csv").open(encoding="utf-8", newline="") as handle:
        problems = {row["program_id"]: row["problem_id"] for row in csv.DictReader(handle)}
    summary, per_query = evaluate_matrices(
        out / "matrices", read_json(out / "memberships.json"), problems, config["seed"])
    write_json(out / "evaluation_summary.json", summary)
    with (out / "per_query_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    print(f"Evaluated {summary['query_count']:,} queries against {summary['gallery_count']:,} candidate submissions.")
    print("\n".join(format_metrics(summary)), flush=True)
    return summary


def report(config: dict) -> None:
    import re

    out = config["_output"]
    collection = read_json(out / "collection_summary.json")
    processed = read_json(out / "processing_summary.json")
    memberships = read_json(out / "memberships.json")
    matrix_path = out / "matrices" / "vectorization_summary.json"
    matrix = read_json(matrix_path) if matrix_path.exists() else None
    evaluation_path = out / "evaluation_summary.json"
    evaluation = read_json(evaluation_path) if evaluation_path.exists() else None
    full = processed.get("scope") == "full corpus" or config.get("full_corpus", False)
    metadata_joined = collection["metadata_audit"]["available"]
    acceptance = "; ".join(f"{count:,} {status.replace('_', ' ')}"
                           for status, count in sorted(collection["acceptance_status"].items()))
    statuses = processed["status_counts"]
    dispositions = processed["disposition_counts"]
    error_statuses = ("read_error", "decode_error", "tokenize_error", "parse_error")
    duplicate_groups = memberships["duplicate_groups"]
    duplicate_ids = {
        program_id: {"group_id": index, "group_size": len(group["program_ids"]),
                     "cross_problem_conflict": group["cross_problem_conflict"],
                     "retained_program_id": "" if group["cross_problem_conflict"] else min(group["program_ids"])}
        for index, group in enumerate(duplicate_groups, 1) for program_id in group["program_ids"]
    }
    query_ids, gallery_ids = set(memberships["query_ids"]), set(memberships["gallery_ids"])
    per_problem = defaultdict(Counter)
    error_reasons = Counter()
    duplicate_fields = ("group_id", "group_size", "cross_problem_conflict", "program_id", "problem_id",
                        "source_path", "processing_disposition", "retained_program_id")
    with (out / "processing_audit.csv").open(encoding="utf-8", newline="") as source, \
            (out / "errors.csv").open("w", encoding="utf-8", newline="") as errors, \
            (out / "duplicates.csv").open("w", encoding="utf-8", newline="") as duplicates:
        reader = csv.DictReader(source)
        error_writer = csv.DictWriter(errors, fieldnames=reader.fieldnames)
        duplicate_writer = csv.DictWriter(duplicates, fieldnames=duplicate_fields, extrasaction="ignore")
        error_writer.writeheader()
        duplicate_writer.writeheader()
        for row in reader:
            counts = per_problem[row["problem_id"]]
            counts["attempted"] += 1
            status = row["status"]
            counts["successful"] += status == "ok"
            counts[status if status in error_statuses else "other_errors"] += status != "ok"
            counts[row["processing_disposition"]] += row["processing_disposition"] in {
                "retained", "duplicate_nonrepresentative", "cross_problem_duplicate_conflict"}
            counts["queries"] += row["program_id"] in query_ids
            counts["gallery"] += row["program_id"] in gallery_ids
            if status != "ok":
                error_writer.writerow(row)
                reason = re.sub(r"\s*\([^()]*, line \d+\)$", "", row["error_message"])
                reason = re.sub(r", \(\d+, \d+\)\)$", ")", reason)
                error_reasons[(status, row["error_type"], reason)] += 1
            if row["program_id"] in duplicate_ids:
                duplicate_writer.writerow({**row, **duplicate_ids[row["program_id"]]})

    metric_names, metric_sums, metric_counts = set(), defaultdict(Counter), Counter()
    query_path = out / "per_query_metrics.csv"
    if evaluation and query_path.exists():
        with query_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                problem_id = row["query_id"].split("/", 1)[0]
                view = row["view"]
                metric_counts[(problem_id, view)] += 1
                for name, value in row.items():
                    if name in {"view", "query_id", "relevant_gallery_count"}:
                        continue
                    name = {"reciprocal_rank": "mrr", "average_precision": "map"}.get(name, name)
                    column = f"{view}_{name}"
                    metric_names.add(column)
                    metric_sums[problem_id][column] += float(value)
    count_fields = ("attempted", "successful", *error_statuses, "other_errors", "duplicate_nonrepresentative",
                    "cross_problem_duplicate_conflict", "retained", "queries", "gallery")
    with (out / "per_problem_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["problem_id", *count_fields, *sorted(metric_names)])
        writer.writeheader()
        for problem_id, counts in sorted(per_problem.items()):
            row = {"problem_id": problem_id, **{key: counts[key] for key in count_fields}}
            for view in evaluation["views"] if evaluation else ():
                n = metric_counts[(problem_id, view)]
                if n:
                    row.update({key: value / n for key, value in metric_sums[problem_id].items()
                                if key.startswith(view + "_")})
            writer.writerow(row)

    scope = "Full corpus" if full else "Development sample"
    lines = ["Python800 results", "",
             f"Input: {collection['programs']:,} submissions, {collection['problems']:,} problems, "
             f"{collection['source_bytes']:,} source bytes.",
             f"Scope: {scope.lower()}; {processed['attempted']:,} submissions from {processed['problems_attempted']:,} problems.",
             f"Parser: CPython {processed['parser']}.",
             f"Official metadata: {'joined' if metadata_joined else 'not joined'}.",
             f"Acceptance: {acceptance}.", "",
             "Processing"]
    for name, count in (
        ("Successful extraction before deduplication", statuses.get("ok", 0)),
        ("Read errors", statuses.get("read_error", 0)),
        ("Decoding errors", statuses.get("decode_error", 0)),
        ("Tokenization errors", statuses.get("tokenize_error", 0)),
        ("Parse errors", statuses.get("parse_error", 0)),
        ("Other errors", sum(value for key, value in statuses.items() if key not in {"ok", *error_statuses})),
        ("Retained after duplicate filtering", processed["unique_parseable_retained"]),
    ):
        lines.append(f"{name}: {count:,}")
    lines.append("Error counts record the first failing stage. Details: errors.csv.")
    if error_reasons:
        lines.extend(["", "Frequent errors (file names and locations omitted)"])
        for (stage, exception, reason), count in error_reasons.most_common(15):
            reason = reason.replace("\n", " ").replace("\r", " ")
            lines.append(f"{stage.replace('_', ' ').capitalize()}: {exception}: {reason} ({count:,} submissions)")
    within_groups = sum(not group["cross_problem_conflict"] for group in duplicate_groups)
    lines.extend(["", "Duplicates",
                  "Compared conservative token sequences: names, literals and docstrings retained; "
                  "comments and nonstructural whitespace ignored.",
                  f"Duplicate groups: {len(duplicate_groups):,}",
                  f"Submissions in duplicate groups, including representatives: {len(duplicate_ids):,}",
                  f"Within-problem duplicate groups: {within_groups:,}",
                  f"Within-problem nonrepresentatives removed: {dispositions.get('duplicate_nonrepresentative', 0):,}",
                  f"Cross-problem duplicate groups: {len(duplicate_groups) - within_groups:,}",
                  f"Cross-problem submissions quarantined: {dispositions.get('cross_problem_duplicate_conflict', 0):,}",
                  f"Queries: {processed['query_count']:,}; candidate submissions: {processed['gallery_count']:,}; "
                  f"retained but unused: {processed['unused_count']:,}.",
                  f"Problems with queries: {sum(counts['queries'] > 0 for counts in per_problem.values()):,}. "
                  f"Problems with no retained submissions: {sum(counts['retained'] == 0 for counts in per_problem.values()):,}.",
                  "Duplicate details: duplicates.csv",
                  "Per-problem statistics: per_problem_statistics.csv"])
    retained = distribution([counts["retained"] for counts in per_problem.values()])
    if retained["count"]:
        lines.append(f"Retained submissions per attempted problem: min {retained['min']:,}, "
                     f"median {retained['median']:g}, p95 {retained['p95']:g}, max {retained['max']:,}.")
    lines.extend(["", "Program sizes before duplicate filtering (successfully parsed submissions)"])
    size_labels = {"ast_max_depth": "AST depth", "ast_node_count": "AST nodes", "loc": "Lines",
                   "source_bytes": "Source bytes", "token_count": "Tokens"}
    for key, value in processed["distributions"].items():
        if value["count"]:
            lines.append(f"{size_labels.get(key, key)}: min={value['min']:,}, median={value['median']:g}, "
                         f"p95={value['p95']:g}, max={value['max']:,}")
    if "encoding_counts" in processed:
        lines.extend(["", "Encodings (includes submissions that failed later)"])
        lines.extend(f"{encoding}: {count:,}" for encoding, count in sorted(processed["encoding_counts"].items()))
        warnings = processed.get("parser_warning_counts", {})
        warning_details = "; ".join(f"{category}: {count:,}" for category, count in sorted(warnings.items())) or "none"
        lines.extend(["", f"Parser warnings: {sum(warnings.values()):,} across "
                      f"{processed.get('parser_warning_programs', 0):,} submissions ({warning_details}).",
                      f"Docstrings removed from features: {processed.get('docstrings_removed', 0):,} "
                      f"across {processed.get('programs_with_docstrings', 0):,} successfully parsed submissions."])
    if matrix:
        lines.extend(["", "Feature matrices",
                      "Vocabulary and IDF fitted on the retrieval corpus; float32 CSR storage.",
                      "Query OOV counts include overlapping n-gram occurrences."])
        view_labels = {"ast_features": "AST", "lexical_raw": "Raw tokens", "lexical_masked": "Masked tokens"}
        for view, details in matrix["views"].items():
            all_rows, query_rows = details["all_rows"], details["nonfit_rows"]
            size = (out / "matrices" / f"{view}.npz").stat().st_size / 2**20
            label = view_labels.get(view, view)
            lines.append(f"{label}: {all_rows['rows']:,} rows, {details['vocabulary_size']:,} features, "
                         f"{all_rows['nonzero_values']:,} nonzero entries, {all_rows['zero_rows']:,} empty rows, "
                         f"{size:.2f} MiB on disk.")
            lines.append(f"{label} queries: {query_rows['oov_occurrences']:,}/{query_rows['feature_occurrences']:,} "
                         f"OOV feature occurrences ({query_rows['oov_occurrence_fraction']:.4%}), "
                         f"{query_rows['rows_with_oov']:,} rows with OOV, {query_rows['zero_rows']:,} empty rows.")
    if evaluation:
        from .evaluate import format_metrics
        lines.extend(["", "TF-IDF retrieval",
                      f"{evaluation['query_count']:,} queries ranked against all "
                      f"{evaluation['gallery_count']:,} candidate submissions using cosine similarity.",
                      "Relevance: same problem ID. Metrics averaged over queries.",
                      "MRR and MAP use the full ranking; recall uses each query's relevant-candidate count.",
                      f"Exact ties use a fixed candidate shuffle (seed {evaluation['seed']}).",
                      *format_metrics(evaluation), ""])
        lines.append("Includes all provisional development, validation and test partitions; "
                     "this is a descriptive evaluation, not a blind held-out evaluation."
                     if full else "Development-sample evaluation.")
        lines.append("Pending verification: acceptance and original problem families."
                     if not metadata_joined else "Pending verification: original problem families.")
    timings = {}
    for stage, fallback in (("inventory", collection), ("process", processed), ("vectorize", matrix), ("evaluate", evaluation)):
        path = out / "provenance" / f"{stage}.json"
        seconds = read_json(path).get("elapsed_seconds") if path.exists() else None
        seconds = seconds if seconds is not None else (fallback or {}).get("elapsed_seconds")
        if seconds is not None:
            timings[stage] = seconds
    if timings:
        lines.extend(["", "Runtime"])
        lines.extend(f"{stage.capitalize()}: {seconds:,.3f} seconds ({seconds / 60:,.2f} minutes)"
                     for stage, seconds in timings.items())
    path = out / "RESULTS.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "RESULTS.md").unlink(missing_ok=True)
    print(f"Saved report to {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Python800 and evaluate code retrieval.")
    parser.add_argument("stage", choices=("run", "inventory", "process", "vectorize", "evaluate", "report"))
    parser.add_argument("--config", type=Path, default=Path("configs/full_benchmark.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    snapshot(config)
    stages = {"inventory": inventory, "process": process, "vectorize": vectorize,
              "evaluate": evaluate, "report": report}
    selected = tuple(stages) if args.stage == "run" else (("evaluate", "report") if args.stage == "evaluate" else (args.stage,))
    for name in selected:
        started = time.perf_counter()
        stages[name](config)
        record_stage(config, name, round(time.perf_counter() - started, 3))

