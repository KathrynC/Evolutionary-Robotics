#!/usr/bin/env python3
"""
apply_step2_commentary.py

Role:
    One-off utility that reads a CSV of docstring additions and applies them
    to Python source files. Used for batch-applying or regenerating module-level
    commentary (e.g., "Ludobots role" and "Beyond Ludobots" sections) across the
    codebase from a spreadsheet-driven workflow.

Pipeline:
    1. Read a CSV with columns "path" and "doc_header_to_add".
    2. For each listed file, parse the AST to find or insert a module docstring.
    3. Combine the existing docstring with the addition (skip if "Ludobots role:"
       already present to avoid duplication).
    4. Either write the files in-place (--write) or produce a unified diff patch.

Outputs:
    - A .patch file (default: step2_docstrings.patch) with all proposed changes.
    - When --write is passed, modified Python files are written in-place.

Usage:
    python3 apply_step2_commentary.py --csv step2_docs.csv                  # dry-run
    python3 apply_step2_commentary.py --csv step2_docs.csv --write          # apply
    python3 apply_step2_commentary.py --csv step2_docs.csv --patch out.patch

Notes:
    - Files with syntax errors are skipped (reported in the skip count).
    - The script preserves the original newline style (LF vs CRLF) of each file.
    - Shebang lines, encoding declarations, and leading comments are preserved
      above the inserted docstring.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
from pathlib import Path

TRIPLE = '"""'


def _inner_doc_from_csv(cell: str) -> str:
    """Extract the inner docstring text from a CSV cell value.

    Strips surrounding triple-quotes if present (CSV export artifacts),
    then trims leading/trailing newlines.

    Args:
        cell: Raw CSV cell content (may be None, empty, or triple-quoted).

    Returns:
        Cleaned docstring body text, or empty string if nothing to add.
    """
    s = (cell or "").strip()
    if not s:
        return ""
    if s.startswith(TRIPLE) and s.endswith(TRIPLE) and len(s) >= 6:
        s = s[len(TRIPLE) : -len(TRIPLE)]
    return s.strip("\n")


def _detect_newline(text: str) -> str:
    """Detect the dominant line-ending style in text (CRLF or LF)."""
    return "\r\n" if "\r\n" in text else "\n"


def _split_lines_keep(text: str):
    """Split text into lines while preserving line-ending characters."""
    return text.splitlines(True)


def _combine_docstrings(existing: str, addition: str, filename: str) -> str:
    """Merge an existing module docstring with new commentary text.

    Args:
        existing: Current docstring content (may be empty or None).
        addition: New text to append (e.g., Ludobots role section).
        filename: The source file's basename, used to deduplicate the first line
            if it just repeats the filename.

    Returns:
        Combined docstring. Returns existing unchanged if:
        - addition is empty, or
        - "Ludobots role:" is already present (idempotent guard).
    """
    existing = (existing or "").rstrip()
    addition = (addition or "").strip()
    if not addition:
        return existing
    if "Ludobots role:" in existing:
        return existing

    add_lines = addition.splitlines()
    if add_lines:
        first = add_lines[0].strip()
        if first == filename and existing.lstrip().startswith(filename):
            add_lines = add_lines[1:]
            while add_lines and not add_lines[0].strip():
                add_lines = add_lines[1:]
        addition = "\n".join(add_lines).strip()

    if not addition:
        return existing
    return existing + ("\n\n" if existing else "") + addition


def _render_module_doc(doc_text: str) -> str:
    """Wrap raw docstring text in triple-quote delimiters for module-level insertion."""
    return f'{TRIPLE}\n{doc_text.rstrip()}\n{TRIPLE}\n'


def _apply_to_file(path: Path, addition_inner: str) -> tuple[str, str, str]:
    """Apply a docstring addition to a single Python source file.

    Handles two cases:
        1. File already has a module docstring: append the addition inside it.
        2. File has no module docstring: insert a new one after shebang/comments.

    Args:
        path: Path to the Python source file.
        addition_inner: The docstring body text to add.

    Returns:
        Tuple of (old_text, new_text, action_tag) where action_tag is one of:
        - "skip:syntax_error" -- file could not be parsed
        - "noop:doc_present"  -- docstring already contains the addition
        - "update:doc_appended" -- existing docstring was extended
        - "insert:doc_added"  -- new docstring was inserted
    """
    old = path.read_text(encoding="utf-8", errors="replace")
    nl = _detect_newline(old)

    try:
        tree = ast.parse(old)
    except SyntaxError:
        return old, old, "skip:syntax_error"

    filename = path.name

    # If module has docstring, append content inside it
    if tree.body and isinstance(tree.body[0], ast.Expr):
        v = getattr(tree.body[0], "value", None)
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            old_doc = v.value
            new_doc = _combine_docstrings(old_doc, addition_inner, filename)
            if new_doc == old_doc:
                return old, old, "noop:doc_present"

            start = tree.body[0].lineno
            end = getattr(tree.body[0], "end_lineno", start)

            lines = old.splitlines(True)
            before = "".join(lines[: start - 1])
            after = "".join(lines[end:])

            new = before + _render_module_doc(new_doc) + after
            return old, new, "update:doc_appended"

    # Otherwise insert new docstring after shebang/encoding/comments/blanklines
    lines = old.splitlines(True)
    i = 0
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    if i < len(lines) and "coding" in lines[i]:
        i += 1
    while i < len(lines) and (lines[i].lstrip().startswith("#") or lines[i].strip() == ""):
        i += 1

    doc = _render_module_doc(addition_inner) + nl
    new = "".join(lines[:i]) + doc + "".join(lines[i:])
    return old, new, "insert:doc_added"


def main():
    """CLI entry point: read CSV, apply docstring additions, write patch and/or files.

    Side effects:
        - Reads the CSV file specified by --csv.
        - When --write is set, modifies Python files in-place.
        - Always writes a unified diff patch file (--patch, default step2_docstrings.patch).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--patch", default="step2_docstrings.patch")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    repo = Path(".").resolve()
    csv_path = Path(args.csv)

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Only touch files listed in the spreadsheet CSV
    mapping = {}
    for row in rows:
        p = (row.get("path") or "").strip()
        if not p:
            continue
        addition = _inner_doc_from_csv(row.get("doc_header_to_add") or "")
        if addition:
            mapping[p] = addition

    diffs = []
    changed = 0
    skipped = 0

    for rel, addition in sorted(mapping.items()):
        fp = repo / rel
        if not fp.exists():
            continue
        old, new, action = _apply_to_file(fp, addition)
        if new != old:
            changed += 1
            if args.write:
                fp.write_text(new, encoding="utf-8")
            diff = difflib.unified_diff(
                _split_lines_keep(old),
                _split_lines_keep(new),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
            diffs.extend(diff)
        else:
            if action.startswith("skip:"):
                skipped += 1

    Path(args.patch).write_text("".join(diffs), encoding="utf-8")
    print(f"Wrote patch: {args.patch}")
    print(f"Would change {changed} files. Skipped (syntax errors): {skipped}.")


if __name__ == "__main__":
    main()
