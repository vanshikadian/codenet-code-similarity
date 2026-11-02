from __future__ import annotations

import csv
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


PROBLEM_PATTERN = re.compile(r"p\d{5}\Z", re.ASCII)
SOURCE_PATTERN = re.compile(r"(s\d{9})\.py\Z", re.ASCII)
SUBMISSION_PATTERN = re.compile(r"s\d{9}\Z", re.ASCII)
METADATA_FIELDS = (
    "submission_id", "problem_id", "user_id", "date", "language",
    "original_language", "filename_ext", "status", "cpu_time", "memory",
    "code_size", "accuracy",
)
REQUIRED_METADATA_FIELDS = (
    "submission_id", "problem_id", "user_id", "status", "language",
    "original_language",
)


def _entries(path: Path) -> list[os.DirEntry[str]]:
    with os.scandir(path) as entries:
        return sorted(entries, key=lambda entry: entry.name)


def iter_sources(root: Path) -> Iterator[dict[str, Any]]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Source root must be a real directory, not a symlink: {root}")
    problems = _entries(root)
    if not problems:
        raise ValueError(f"Source root contains no problem directories: {root}")
    for problem in problems:
        if (not PROBLEM_PATTERN.fullmatch(problem.name)
                or problem.is_symlink()
                or not problem.is_dir(follow_symlinks=False)):
            raise ValueError(f"Unexpected problem entry: {problem.path}")
    for problem in problems:
        sources = _entries(Path(problem.path))
        if not sources:
            raise ValueError(f"Problem directory contains no source files: {problem.path}")
        for source in sources:
            match = SOURCE_PATTERN.fullmatch(source.name)
            if (match is None or source.is_symlink()
                    or not source.is_file(follow_symlinks=False)):
                raise ValueError(f"Unexpected source entry: {source.path}")
            yield {
                "problem_id": problem.name,
                "submission_id": match.group(1),
                "source_path": f"{problem.name}/{source.name}",
                "source_bytes": source.stat(follow_symlinks=False).st_size,
            }


def load_metadata(
    metadata_dir: Path,
    problem_ids: Iterable[str],
    *,
    submission_ids: Iterable[tuple[str, str]] | None = None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    metadata_dir = Path(metadata_dir)
    if metadata_dir.is_symlink() or not metadata_dir.is_dir():
        raise ValueError(f"Metadata root must be a real directory: {metadata_dir}")
    problems = sorted(set(problem_ids))
    for problem_id in problems:
        if not PROBLEM_PATTERN.fullmatch(problem_id):
            raise ValueError(f"Malformed problem identifier: {problem_id!r}")
    wanted = None if submission_ids is None else set(submission_ids)
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "metadata_directory": str(metadata_dir),
        "requested_problem_count": len(problems),
        "files_read_count": 0,
        "missing_problem_files": [],
        "missing_columns": {},
        "rows_read": 0,
        "rows_filtered_out": 0,
        "invalid_submission_id_rows": 0,
        "duplicate_key_rows": 0,
        "problem_id_mismatch_rows": 0,
        "records_retained": 0,
        "conflicted_records": 0,
    }
    for problem_id in problems:
        path = metadata_dir / f"{problem_id}.csv"
        if path.is_symlink():
            raise ValueError(f"Metadata CSV must not be a symlink: {path}")
        if not path.exists():
            summary["missing_problem_files"].append(problem_id)
            continue
        if not path.is_file():
            raise ValueError(f"Metadata CSV is not a regular file: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if len(fieldnames) != len(set(fieldnames)):
                raise ValueError(f"Duplicate metadata column names: {path}")
            missing = sorted(set(REQUIRED_METADATA_FIELDS) - set(fieldnames))
            if missing:
                summary["missing_columns"][problem_id] = missing
            if "submission_id" in missing or "problem_id" in missing:
                raise ValueError(f"Metadata CSV lacks identity columns {missing}: {path}")
            summary["files_read_count"] += 1
            for line_number, row in enumerate(reader, start=2):
                summary["rows_read"] += 1
                submission_id = (row.get("submission_id") or "").strip()
                if not SUBMISSION_PATTERN.fullmatch(submission_id):
                    summary["invalid_submission_id_rows"] += 1
                    continue
                key = (problem_id, submission_id)
                if wanted is not None and key not in wanted:
                    summary["rows_filtered_out"] += 1
                    continue
                record: dict[str, Any] = {
                    field: (row.get(field) or "").strip() for field in METADATA_FIELDS
                }
                conflicts: list[str] = []
                if record["problem_id"] != problem_id:
                    conflicts.append("problem_id_mismatch")
                    summary["problem_id_mismatch_rows"] += 1
                if None in row:
                    conflicts.append("extra_csv_values")
                if any(value is None for name, value in row.items() if name is not None):
                    conflicts.append("truncated_csv_row")
                record["_metadata_conflicts"] = conflicts
                record["_metadata_csv"] = path.name
                record["_metadata_line"] = line_number
                if key in lookup:
                    summary["duplicate_key_rows"] += 1
                    existing_conflicts = lookup[key]["_metadata_conflicts"]
                    for reason in ("duplicate_key", *conflicts):
                        if reason not in existing_conflicts:
                            existing_conflicts.append(reason)
                else:
                    lookup[key] = record
    summary["records_retained"] = len(lookup)
    summary["conflicted_records"] = sum(bool(row["_metadata_conflicts"]) for row in lookup.values())
    return lookup, summary


def _dialect_status(original_language: str) -> str:
    label = original_language.strip().casefold()
    # [PyPy (2.4.0) identifies the runtime version, not the Python dialect.]
    python_version = re.match(r"^python\s*\(?\s*([23])(?=\s|[.()]|$)", label)
    pypy_version = re.match(r"^pypy\s*([23])(?=\s|[.()]|$)", label)
    match = python_version or pypy_version
    if match is None:
        return "unknown"
    return f"python{match.group(1)}"


def classify_record(
    record: Mapping[str, Any],
    metadata_lookup: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    language_allowlist: Iterable[str] | None = None,
    *,
    assume_accepted: bool = False,
) -> dict[str, Any]:
    result = dict(record)
    for field in METADATA_FIELDS:
        if field not in ("problem_id", "submission_id"):
            result[field] = ""
    result.update({
        "metadata_status": "unverified",
        "metadata_conflicts": [],
        "dialect_status": "unknown",
        "acceptance_status": "unverified",
        "eligibility_status": "metadata_unverified",
        "exclusion_reason": "metadata_not_available",
    })
    if metadata_lookup is None:
        if assume_accepted:
            result.update(
                acceptance_status="assumed_accepted",
                eligibility_status="provisional_candidate",
                exclusion_reason="",
            )
        return result
    key = (str(record["problem_id"]), str(record["submission_id"]))
    metadata = metadata_lookup.get(key)
    if metadata is None:
        result.update(metadata_status="missing", exclusion_reason="metadata_row_missing")
        return result
    for field in METADATA_FIELDS:
        if field not in ("problem_id", "submission_id"):
            result[field] = metadata.get(field, "")
    conflicts = list(metadata.get("_metadata_conflicts", []))
    if metadata.get("problem_id", key[0]) != key[0] and "problem_id_mismatch" not in conflicts:
        conflicts.append("problem_id_mismatch")
    if metadata.get("submission_id", key[1]) != key[1]:
        conflicts.append("submission_id_mismatch")
    result["dialect_status"] = _dialect_status(str(result["original_language"]))
    result["metadata_conflicts"] = conflicts
    if conflicts:
        result.update(
            metadata_status="conflict", eligibility_status="rejected",
            exclusion_reason="metadata_conflict:" + ",".join(conflicts),
        )
        return result
    result["metadata_status"] = "matched"
    status = result["status"]
    if not status:
        result["exclusion_reason"] = "acceptance_status_missing"
        return result
    if status != "Accepted":
        result.update(
            acceptance_status="rejected", eligibility_status="rejected",
            exclusion_reason="submission_not_accepted",
        )
        return result
    result["acceptance_status"] = "verified_accepted"
    if result["dialect_status"] != "python3":
        result.update(
            eligibility_status="rejected",
            exclusion_reason=("python2_dialect" if result["dialect_status"] == "python2"
                              else "original_language_ambiguous"),
        )
        return result
    if language_allowlist is not None and result["original_language"] not in language_allowlist:
        result.update(eligibility_status="rejected", exclusion_reason="language_not_allowlisted")
        return result
    result.update(eligibility_status="candidate", exclusion_reason="")
    return result
