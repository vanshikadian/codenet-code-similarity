from __future__ import annotations

import ast
from collections import Counter
import io
import json
import keyword
import tokenize
from typing import Any
import warnings


FEATURE_VERSION = "python-lexical-ast-v2"

_IGNORED_TOKEN_TYPES = frozenset(
    (tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT, tokenize.NL)
)
_LAYOUT_TOKEN_TYPES = frozenset(
    (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
)
_DOCSTRING_CONTAINERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _key(parts: list[str] | tuple[str, ...]) -> str:
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def _token_value(token: tokenize.TokenInfo, *, mask: bool = False) -> str:
    name = tokenize.tok_name[token.type]
    if token.type in _LAYOUT_TOKEN_TYPES:
        return name
    value = token.string
    if (
        mask
        and token.type == tokenize.NAME
        and not keyword.iskeyword(value)
        and not keyword.issoftkeyword(value)
    ):
        value = "ID"
    return f"{name}:{value}"


def _ngram_counts(tokens: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for size in (1, 2, 3):
        for index in range(len(tokens) - size + 1):
            counts[_key(tokens[index : index + size])] += 1
    return dict(sorted(counts.items()))


def _docstring_statements(tree: ast.AST) -> set[ast.Expr]:
    statements: set[ast.Expr] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_CONTAINERS) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            statements.add(first)
    return statements


def _without_docstrings(
    tokens: list[tokenize.TokenInfo], statements: set[ast.Expr], source: str
) -> list[tokenize.TokenInfo]:
    # [AST columns use UTF-8 bytes; tokenizer columns use Unicode characters.]
    lines = source.split("\n")

    def position(line: int, byte_column: int) -> tuple[int, int]:
        prefix = lines[line - 1].encode("utf-8")[:byte_column]
        return (line, len(prefix.decode("utf-8")))

    spans = sorted(
        (
            position(statement.lineno, statement.col_offset),
            position(statement.end_lineno, statement.end_col_offset),
        )
        for statement in statements
    )
    removed: set[int] = set()
    index = 0
    for start, end in spans:
        while index < len(tokens) and tokens[index].end <= start:
            index += 1
        while index < len(tokens) and tokens[index].start < end:
            removed.add(index)
            index += 1
        # [Remove the separator so docstrings add no comparison tokens.]
        following = index
        while following < len(tokens) and tokens[following].type in (
            tokenize.COMMENT,
            tokenize.NL,
        ):
            following += 1
        if following < len(tokens):
            token = tokens[following]
            if token.type == tokenize.NEWLINE or (
                token.type == tokenize.OP and token.string == ";"
            ):
                removed.add(following)
    return [token for index, token in enumerate(tokens) if index not in removed]


def _structural_counts(
    tree: ast.AST, docstrings: set[ast.Expr]
) -> tuple[dict[str, int], int, int]:
    counts: Counter[str] = Counter()
    # [Track ancestry per occurrence because Python shares operator/context objects.]
    stack = [(tree, 0, None, None, None, None)]
    node_count = 0
    max_depth = 0
    while stack:
        node, depth, parent, field, grandparent, parent_field = stack.pop()
        if node in docstrings:
            continue
        node_type = type(node).__name__
        node_count += 1
        max_depth = max(max_depth, depth)
        counts[_key(["node", node_type])] += 1
        if isinstance(node, ast.Constant):
            counts[_key(["literal_type", type(node.value).__name__])] += 1
        if parent is not None:
            counts[_key(["edge", parent, field, node_type])] += 1
        if grandparent is not None:
            counts[
                _key(["path2", grandparent, parent_field, parent, field, node_type])
            ] += 1
        for child_field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else (value,)
            for child in children:
                if isinstance(child, ast.AST):
                    stack.append((child, depth + 1, node_type, child_field, parent, field))
    return dict(sorted(counts.items())), node_count, max_depth


def _failure(record: dict[str, Any], status: str, error: Exception) -> dict[str, Any]:
    record.update(status=status, error_type=type(error).__name__, error_message=str(error))
    return record


def extract_features(source: bytes, filename: str = "<source>") -> dict[str, Any]:
    record: dict[str, Any] = {
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "source_bytes": len(source),
        "feature_version": FEATURE_VERSION,
        "encoding": None,
        "loc": 0,
        "token_count": 0,
        "conservative_token_count": 0,
        "ast_node_count": 0,
        "ast_max_depth": 0,
        "conservative_tokens": [],
        "raw_tokens": [],
        "masked_tokens": [],
        "lexical_raw": {},
        "lexical_masked": {},
        "ast_features": {},
        "docstring_count": 0,
        "parser_warnings": [],
    }
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
        record["encoding"] = encoding
        text = source.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
    except (UnicodeError, LookupError, SyntaxError) as error:
        return _failure(record, "decode_error", error)
    record["loc"] = text.count("\n") + int(bool(text) and not text.endswith("\n"))

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        bad_tokens = [token for token in tokens if token.type == tokenize.ERRORTOKEN]
        if bad_tokens:
            first = bad_tokens[0]
            raise tokenize.TokenError(
                f"Invalid token {first.string!r}", first.start
            )
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        return _failure(record, "tokenize_error", error)

    conservative_tokens = [
        _token_value(token) for token in tokens if token.type not in _IGNORED_TOKEN_TYPES
    ]
    record["conservative_token_count"] = len(conservative_tokens)

    parser_warnings = []
    try:
        with warnings.catch_warnings(record=True) as parser_warnings:
            warnings.simplefilter("always", SyntaxWarning)
            tree = ast.parse(text, filename=filename, mode="exec")
    except (SyntaxError, ValueError, RecursionError) as error:
        return _failure(record, "parse_error", error)
    finally:
        record["parser_warnings"] = [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
                "lineno": warning.lineno,
            }
            for warning in parser_warnings
        ]

    docstrings = _docstring_statements(tree)
    comparison_tokens = [
        token
        for token in _without_docstrings(tokens, docstrings, text)
        if token.type not in _IGNORED_TOKEN_TYPES
    ]
    raw_tokens = [_token_value(token) for token in comparison_tokens]
    masked_tokens = [_token_value(token, mask=True) for token in comparison_tokens]
    ast_features, ast_node_count, ast_max_depth = _structural_counts(tree, docstrings)
    record.update(
        conservative_tokens=conservative_tokens,
        token_count=len(raw_tokens),
        ast_node_count=ast_node_count,
        ast_max_depth=ast_max_depth,
        raw_tokens=raw_tokens,
        masked_tokens=masked_tokens,
        lexical_raw=_ngram_counts(raw_tokens),
        lexical_masked=_ngram_counts(masked_tokens),
        ast_features=ast_features,
        docstring_count=len(docstrings),
    )
    return record
