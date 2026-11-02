from __future__ import annotations

import json
import gzip
import math
import numbers
import platform
from array import array
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


FEATURE_FIELDS = ("lexical_raw", "lexical_masked", "ast_features")
SCHEMA_VERSION = 1
TFIDF_SETTINGS = {"sublinear_tf": True, "norm": "l2", "smooth_idf": True, "use_idf": True}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def _clean_counts(features: dict, program_id: str, field: str) -> dict:
    if not isinstance(features, dict):
        raise ValueError(f"{program_id}: {field} must be a feature-count dictionary")
    counts = {}
    for feature, count in features.items():
        if not isinstance(feature, str) or not feature:
            raise ValueError(f"{program_id}: {field} contains a non-string/empty feature")
        if (isinstance(count, bool) or not isinstance(count, numbers.Real)
                or not math.isfinite(count) or count < 0 or count != int(count)):
            raise ValueError(f"{program_id}: {field} counts must be finite nonnegative integers")
        if count:
            counts[feature] = int(count)
    return counts


def _index_records(records: list[dict]) -> dict[str, dict]:
    if not isinstance(records, (list, tuple)):
        raise ValueError("records must be a list of feature records")
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each feature record must be a dictionary")
        program_id = record.get("program_id")
        if not isinstance(program_id, str) or not program_id:
            raise ValueError("Each record needs a nonempty string program_id")
        if program_id in indexed:
            raise ValueError(f"Duplicate program_id: {program_id}")
        clean = {"program_id": program_id}
        for field in FEATURE_FIELDS:
            clean[field] = _clean_counts(record.get(field), program_id, field)
        indexed[program_id] = clean
    return indexed


def _ordered_ids(ids: list[str] | None, indexed: dict[str, dict], *, name: str, nonempty: bool = False) -> list[str]:
    if (ids is None and nonempty) or (ids is not None and not isinstance(ids, (list, tuple))):
        raise ValueError(f"{name} must be a list of IDs")
    requested = list(indexed) if ids is None else list(ids)
    if any(not isinstance(program_id, str) or not program_id for program_id in requested):
        raise ValueError(f"{name} must contain nonempty string IDs")
    if len(set(requested)) != len(requested):
        raise ValueError(f"{name} contains duplicate IDs")
    if nonempty and not requested:
        raise ValueError(f"{name} cannot be empty")
    missing = set(requested).difference(indexed)
    if missing:
        raise ValueError(f"{name} references unknown IDs: {sorted(missing)[:5]}")
    return sorted(requested)


def _coverage(records: list[dict], field: str, vocabulary: dict[str, int], matrix: sparse.csr_matrix) -> dict:
    occurrences = 0
    oov_occurrences = 0
    unique_features = set()
    unique_oov = set()
    oov_rows = 0
    for record in records:
        counts = record[field]
        occurrences += sum(counts.values())
        unknown = set(counts).difference(vocabulary)
        unique_features.update(counts)
        unique_oov.update(unknown)
        oov_occurrences += sum(counts[key] for key in unknown)
        oov_rows += bool(unknown)
    zero_rows = np.flatnonzero(np.diff(matrix.indptr) == 0).tolist()
    return {
        "rows": len(records),
        "columns": matrix.shape[1],
        "nonzero_values": matrix.nnz,
        "zero_rows": len(zero_rows),
        "zero_row_ids": [records[i]["program_id"] for i in zero_rows],
        "rows_with_oov": oov_rows,
        "feature_occurrences": occurrences,
        "oov_occurrences": oov_occurrences,
        "oov_occurrence_fraction": oov_occurrences / occurrences if occurrences else 0.0,
        "unique_features": len(unique_features),
        "unique_oov_features": len(unique_oov),
        "unique_oov_fraction": len(unique_oov) / len(unique_features) if unique_features else 0.0,
    }


def _validate_matrix(matrix: sparse.csr_matrix, shape: tuple[int, int]) -> None:
    if not sparse.isspmatrix_csr(matrix) or matrix.dtype != np.dtype("float32") or matrix.shape != shape:
        raise ValueError(f"Expected a float32 CSR matrix with shape {shape}")
    matrix.check_format(full_check=True)
    if not matrix.has_canonical_format:
        raise ValueError("TF-IDF matrix contains unsorted or duplicate column indices")
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError("TF-IDF values must be finite and nonnegative")
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    nonempty = np.diff(matrix.indptr) > 0
    if np.any(np.abs(norms[nonempty] - 1) > 1e-6):
        raise ValueError("Nonempty TF-IDF rows must have unit L2 norm")


def _transform(counts: list[dict], vocabulary: dict[str, int], idf: np.ndarray) -> sparse.csr_matrix:
    if not counts or not vocabulary:
        matrix = sparse.csr_matrix((len(counts), len(vocabulary)), dtype=np.float32)
    else:
        vectorizer = DictVectorizer(dtype=np.float64, sparse=True, sort=True)
        vectorizer.vocabulary_ = vocabulary
        vectorizer.feature_names_ = [key for key, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
        count_matrix = vectorizer.transform(counts)
        transformer = TfidfTransformer(**TFIDF_SETTINGS)
        transformer.idf_ = idf
        transformer.n_features_in_ = len(vocabulary)
        matrix = transformer.transform(count_matrix).astype(np.float32).tocsr()
        matrix.sort_indices()
    _validate_matrix(matrix, (len(counts), len(vocabulary)))
    return matrix


def _save_matrices(output_dir: Path, matrices: dict, program_ids: list[str], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for field, matrix in matrices.items():
        sparse.save_npz(output_dir / f"{field}.npz", matrix, compressed=True)
    _write_json(output_dir / "program_ids.json", program_ids)
    _write_json(output_dir / "vectorization_summary.json", summary)


def fit_feature_matrices(
    records: list[dict],
    output_dir: Path,
    fit_ids: list[str],
    transform_ids: list[str] | None = None,
) -> dict:
    indexed = _index_records(records)
    gallery_ids = _ordered_ids(fit_ids, indexed, name="fit_ids", nonempty=True)
    row_ids = _ordered_ids(transform_ids, indexed, name="transform_ids")
    gallery = [indexed[program_id] for program_id in gallery_ids]
    rows = [indexed[program_id] for program_id in row_ids]
    matrices = {}
    models = {}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "fit_rows": len(gallery_ids),
        "transform_rows": len(row_ids),
        "row_order": "lexicographic program_id",
        "vocabulary_and_idf_scope": "fit_ids only; no query fitting",
        "matrix_dtype": "float32",
        "idf_dtype": "float64",
        "settings": TFIDF_SETTINGS.copy(),
        "coverage_note": "Occurrences count all extracted features, including overlapping n-grams. Zero-denominator OOV fractions are 0.0.",
        "views": {},
    }
    gallery_id_set = set(gallery_ids)
    row_indices = {program_id: i for i, program_id in enumerate(row_ids)}
    heldout_indices = [i for i, program_id in enumerate(row_ids) if program_id not in gallery_id_set]
    for field in FEATURE_FIELDS:
        fit_counts = [record[field] for record in gallery]
        if any(fit_counts):
            vectorizer = DictVectorizer(dtype=np.float64, sparse=True, sort=True)
            gallery_counts = vectorizer.fit_transform(fit_counts)
            transformer = TfidfTransformer(**TFIDF_SETTINGS).fit(gallery_counts)
            vocabulary = {key: int(index) for key, index in vectorizer.vocabulary_.items()}
            idf = transformer.idf_.astype(np.float64)
        else:
            vocabulary = {}
            idf = np.empty(0, dtype=np.float64)
        matrix = _transform([record[field] for record in rows], vocabulary, idf)
        if gallery_id_set.issubset(row_indices):
            gallery_matrix = matrix[[row_indices[program_id] for program_id in gallery_ids]].tocsr()
        else:
            gallery_matrix = _transform(fit_counts, vocabulary, idf)
        matrices[field] = matrix
        models[field] = (vocabulary, idf)
        summary["views"][field] = {
            "vocabulary_size": len(vocabulary),
            "all_rows": _coverage(rows, field, vocabulary, matrix),
            "fit_rows": _coverage(gallery, field, vocabulary, gallery_matrix),
            "nonfit_rows": _coverage([rows[i] for i in heldout_indices], field, vocabulary, matrix[heldout_indices].tocsr()),
        }
    output_dir = Path(output_dir)
    _save_matrices(output_dir, matrices, row_ids, summary)
    _write_json(output_dir / "fit_ids.json", gallery_ids)
    for field, (vocabulary, idf) in models.items():
        _write_json(output_dir / f"{field}.vocabulary.json", vocabulary)
        np.save(output_dir / f"{field}.idf.npy", idf, allow_pickle=False)
    _write_json(output_dir / "model_metadata.json", {
        "schema_version": SCHEMA_VERSION,
        "feature_fields": list(FEATURE_FIELDS),
        "settings": TFIDF_SETTINGS.copy(),
        "fit_rows": len(gallery_ids),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    })
    return summary


def _file_counts(path: Path, field: str, wanted: set[str]):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            program_id = record["program_id"]
            if program_id in wanted:
                yield program_id, _clean_counts(record.get(field), program_id, field)


def _new_coverage() -> dict:
    return {"rows": 0, "nonzero_values": 0, "zero_row_ids": [], "rows_with_oov": 0,
            "feature_occurrences": 0, "oov_occurrences": 0, "known": set(), "unknown": set()}


def _add_coverage(coverage: dict, program_id: str, counts: dict, vocabulary: dict) -> None:
    known = {vocabulary[key] for key in counts if key in vocabulary}
    unknown = set(counts).difference(vocabulary)
    coverage["rows"] += 1
    coverage["nonzero_values"] += len(known)
    if not known:
        coverage["zero_row_ids"].append(program_id)
    coverage["rows_with_oov"] += bool(unknown)
    coverage["feature_occurrences"] += sum(counts.values())
    coverage["oov_occurrences"] += sum(counts[key] for key in unknown)
    coverage["known"].update(known)
    coverage["unknown"].update(unknown)


def _finish_coverage(coverage: dict, columns: int) -> dict:
    known = coverage.pop("known")
    unknown = coverage.pop("unknown")
    coverage["columns"] = columns
    coverage["zero_row_ids"].sort()
    coverage["zero_rows"] = len(coverage["zero_row_ids"])
    coverage["unique_features"] = len(known) + len(unknown)
    coverage["unique_oov_features"] = len(unknown)
    total = coverage["feature_occurrences"]
    coverage["oov_occurrence_fraction"] = coverage["oov_occurrences"] / total if total else 0.0
    total = coverage["unique_features"]
    coverage["unique_oov_fraction"] = len(unknown) / total if total else 0.0
    return coverage


def fit_feature_matrices_from_file(
    features_path: Path,
    output_dir: Path,
    fit_ids: list[str],
    transform_ids: list[str],
    batch_size: int = 1000,
) -> dict:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    for name, values in (("fit_ids", fit_ids), ("transform_ids", transform_ids)):
        if (not isinstance(values, (list, tuple))
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name} must contain unique nonempty string IDs")
    if not fit_ids:
        raise ValueError("fit_ids cannot be empty")
    gallery_ids = sorted(fit_ids)
    row_ids = sorted(transform_ids)
    gallery_set, row_set = set(gallery_ids), set(row_ids)
    wanted = gallery_set | row_set
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION, "fit_rows": len(gallery_ids), "transform_rows": len(row_ids),
        "row_order": "lexicographic program_id", "vocabulary_and_idf_scope": "fit_ids only; no query fitting",
        "matrix_dtype": "float32", "idf_dtype": "float64", "settings": TFIDF_SETTINGS.copy(),
        "coverage_note": "Occurrences count all extracted features, including overlapping n-grams. Zero-denominator OOV fractions are 0.0.",
        "views": {},
    }
    for field in FEATURE_FIELDS:
        print(f"Fitting {field} vocabulary from {len(gallery_ids):,} retrieval corpus rows", flush=True)
        frequencies = Counter()
        seen = set()
        for program_id, counts in _file_counts(features_path, field, wanted):
            if program_id in seen:
                raise ValueError(f"Duplicate program_id: {program_id}")
            seen.add(program_id)
            if program_id in gallery_set:
                frequencies.update(counts.keys())
        if seen != wanted:
            raise ValueError(f"Feature file is missing IDs: {sorted(wanted - seen)[:5]}")
        vocabulary = {feature: index for index, feature in enumerate(sorted(frequencies))}
        idf = np.array([frequencies[feature] for feature in vocabulary], dtype=np.float64)
        idf += 1
        np.divide(len(gallery_ids) + 1, idf, out=idf)
        np.log(idf, out=idf)
        idf += 1
        del frequencies, seen
        vectorizer = DictVectorizer(dtype=np.float64, sparse=True, sort=True)
        vectorizer.vocabulary_ = vocabulary
        vectorizer.feature_names_ = list(vocabulary)
        transformer = TfidfTransformer(**TFIDF_SETTINGS)
        transformer.idf_ = idf
        transformer.n_features_in_ = len(vocabulary)
        coverage = {scope: _new_coverage() for scope in ("all_rows", "fit_rows", "nonfit_rows")}
        values, columns, pointers = array("f"), array("i"), array("q", [0])
        file_order = []
        batch_counts, batch_ids = [], []

        def append_batch():
            if vocabulary:
                matrix = transformer.transform(vectorizer.transform(batch_counts)).astype(np.float32).tocsr()
                matrix.sort_indices()
            else:
                matrix = sparse.csr_matrix((len(batch_counts), 0), dtype=np.float32)
            _validate_matrix(matrix, (len(batch_counts), len(vocabulary)))
            values.frombytes(matrix.data.tobytes())
            columns.frombytes(matrix.indices.astype(np.int32, copy=False).tobytes())
            pointers.extend((matrix.indptr[1:] + pointers[-1]).tolist())
            file_order.extend(batch_ids)
            batch_counts.clear()
            batch_ids.clear()

        for program_id, counts in _file_counts(features_path, field, wanted):
            if program_id in gallery_set:
                _add_coverage(coverage["fit_rows"], program_id, counts, vocabulary)
            if program_id in row_set:
                _add_coverage(coverage["all_rows"], program_id, counts, vocabulary)
                if program_id not in gallery_set:
                    _add_coverage(coverage["nonfit_rows"], program_id, counts, vocabulary)
                batch_counts.append(counts)
                batch_ids.append(program_id)
                if len(batch_ids) == batch_size:
                    append_batch()
        if batch_ids:
            append_batch()
        matrix = sparse.csr_matrix((np.frombuffer(values, dtype=np.float32),
                                    np.frombuffer(columns, dtype=np.int32),
                                    np.frombuffer(pointers, dtype=np.int64)),
                                   shape=(len(row_ids), len(vocabulary)))
        if file_order != row_ids:
            positions = {program_id: index for index, program_id in enumerate(file_order)}
            matrix = matrix[[positions[program_id] for program_id in row_ids]].tocsr()
        del values, columns, pointers
        _validate_matrix(matrix, (len(row_ids), len(vocabulary)))
        sparse.save_npz(output_dir / f"{field}.npz", matrix, compressed=True)
        summary["views"][field] = {"vocabulary_size": len(vocabulary), **{
            scope: _finish_coverage(value, len(vocabulary)) for scope, value in coverage.items()}}
        _write_json(output_dir / f"{field}.vocabulary.json", vocabulary)
        np.save(output_dir / f"{field}.idf.npy", idf, allow_pickle=False)
        print(f"Saved {field}: {matrix.shape[0]:,} rows, {matrix.shape[1]:,} columns, {matrix.nnz:,} nonzeros", flush=True)
        del matrix, vocabulary, vectorizer, transformer, coverage
    _write_json(output_dir / "program_ids.json", row_ids)
    _write_json(output_dir / "fit_ids.json", gallery_ids)
    _write_json(output_dir / "vectorization_summary.json", summary)
    _write_json(output_dir / "model_metadata.json", {
        "schema_version": SCHEMA_VERSION, "feature_fields": list(FEATURE_FIELDS),
        "settings": TFIDF_SETTINGS.copy(), "fit_rows": len(gallery_ids),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    })
    return summary


def load_models(model_dir: Path) -> dict[str, tuple[dict[str, int], np.ndarray]]:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("settings") != TFIDF_SETTINGS):
        raise ValueError("Unsupported vectorization model schema or TF–IDF settings")
    if metadata.get("feature_fields") != list(FEATURE_FIELDS):
        raise ValueError("Model feature fields do not match this pipeline")
    if type(metadata.get("fit_rows")) is not int or metadata["fit_rows"] < 1:
        raise ValueError("Model fit_rows must be a positive integer")
    fit_ids = json.loads((model_dir / "fit_ids.json").read_text(encoding="utf-8"))
    if (not isinstance(fit_ids, list) or any(not isinstance(value, str) or not value for value in fit_ids)
            or len(set(fit_ids)) != len(fit_ids) or len(fit_ids) != metadata["fit_rows"]):
        raise ValueError("Model fit_ids must be unique IDs matching fit_rows")
    models = {}
    for field in FEATURE_FIELDS:
        vocabulary = json.loads((model_dir / f"{field}.vocabulary.json").read_text(encoding="utf-8"))
        if (not isinstance(vocabulary, dict)
                or any(not isinstance(key, str) or not key for key in vocabulary)
                or any(type(index) is not int for index in vocabulary.values())
                or sorted(vocabulary.values()) != list(range(len(vocabulary)))):
            raise ValueError(f"Invalid vocabulary for {field}")
        idf = np.load(model_dir / f"{field}.idf.npy", allow_pickle=False)
        if (idf.shape != (len(vocabulary),) or idf.dtype != np.dtype("float64")
                or not np.all(np.isfinite(idf)) or np.any(idf < 1)):
            raise ValueError(f"Invalid IDF array for {field}")
        models[field] = (vocabulary, idf.astype(np.float64))
    return models


def transform_records(
    records: list[dict],
    model_dir: Path,
    transform_ids: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict:
    indexed = _index_records(records)
    row_ids = _ordered_ids(transform_ids, indexed, name="transform_ids")
    rows = [indexed[program_id] for program_id in row_ids]
    models = load_models(model_dir)
    matrices = {}
    summary = {"schema_version": SCHEMA_VERSION, "transform_rows": len(rows), "row_order": "lexicographic program_id", "settings": TFIDF_SETTINGS.copy(), "views": {}}
    for field, (vocabulary, idf) in models.items():
        matrix = _transform([record[field] for record in rows], vocabulary, idf)
        matrices[field] = matrix
        summary["views"][field] = _coverage(rows, field, vocabulary, matrix)
    if output_dir is not None:
        if Path(output_dir).resolve() == Path(model_dir).resolve():
            raise ValueError("Transform output_dir must differ from model_dir to preserve fitted artifacts")
        _save_matrices(Path(output_dir), matrices, row_ids, summary)
    return {"program_ids": row_ids, "matrices": matrices, "summary": summary}
