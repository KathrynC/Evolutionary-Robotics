#!/usr/bin/env python3
"""
pest_control.py — Background code health monitor

Watches for recently modified Python files and runs two tiers of analysis:

  Tier 1 (fast, no LLM): syntax check, import verification, common bug patterns
  Tier 2 (LLM-powered): semantic review, logic bugs, cross-file inconsistencies

Writes findings to artifacts/pest_control_report.json, which Claude Code
can read to catch issues early.

Usage:
    python pest_control.py                  # One-shot scan of recent changes
    python pest_control.py --watch          # Continuous monitoring (polls every 60s)
    python pest_control.py --full           # Deep scan of ALL Python files
    python pest_control.py --file foo.py    # Scan a specific file
"""

import ast
import json
import os
import subprocess
import sys
import time
import py_compile
import importlib
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/generate"

# Use the lightest model to minimize contention with experiments
LLM_MODEL = "llama3.1:latest"

REPORT_PATH = PROJECT / "artifacts" / "pest_control_report.json"

# Files to skip (generated, data, external)
SKIP_PATTERNS = {
    "pyrosim/", "artifacts/", "videos/", "twine/", "__pycache__",
    ".git/", "tools/telemetry/", "pest_control.py"  # don't review yourself
}


def should_skip(path):
    rel = str(path.relative_to(PROJECT))
    return any(skip in rel for skip in SKIP_PATTERNS)


def get_python_files(since_minutes=None):
    """Get Python files, optionally filtered by modification time."""
    files = sorted(PROJECT.glob("*.py")) + sorted(PROJECT.glob("tools/**/*.py"))
    files = [f for f in files if not should_skip(f)]

    if since_minutes:
        cutoff = time.time() - (since_minutes * 60)
        files = [f for f in files if f.stat().st_mtime > cutoff]

    return files


# ── Tier 1: Static Analysis (no LLM) ────────────────────────────────────────

def check_syntax(filepath):
    """Check Python syntax."""
    issues = []
    try:
        py_compile.compile(str(filepath), doraise=True)
    except py_compile.PyCompileError as e:
        issues.append({
            "severity": "error",
            "type": "syntax_error",
            "message": str(e),
            "line": getattr(e, 'lineno', None),
        })
    return issues


def check_ast_patterns(filepath):
    """AST-based pattern checks for common bugs."""
    issues = []
    try:
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return issues  # Already caught by syntax check

    # Check for bare except
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append({
                "severity": "warning",
                "type": "bare_except",
                "message": "Bare 'except:' catches all exceptions including KeyboardInterrupt",
                "line": node.lineno,
            })

    # Check for mutable default arguments
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append({
                        "severity": "warning",
                        "type": "mutable_default",
                        "message": f"Mutable default argument in function '{node.name}'",
                        "line": node.lineno,
                    })

    # Check for f-strings or format strings with potential issues
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for open() without encoding on text files
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                args = node.args
                keywords = {kw.arg: kw for kw in node.keywords}
                # If mode is not 'b' and no encoding specified
                mode = None
                if len(args) >= 2:
                    if isinstance(args[1], ast.Constant):
                        mode = args[1].value
                elif "mode" in keywords:
                    if isinstance(keywords["mode"].value, ast.Constant):
                        mode = keywords["mode"].value.value
                if mode and "b" not in str(mode):
                    if "encoding" not in keywords:
                        pass  # This is common in the codebase, don't flag

    # Check for unused imports (simple version)
    imported_names = set()
    used_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names.add((name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names.add((name, node.lineno))
        elif isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)

    for name, lineno in imported_names:
        if name not in used_names and name != "*":
            issues.append({
                "severity": "info",
                "type": "unused_import",
                "message": f"Import '{name}' appears unused",
                "line": lineno,
            })

    return issues


def check_common_patterns(filepath):
    """Regex-based checks for common issues."""
    issues = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception:
        return issues

    for i, line in enumerate(lines, 1):
        # Hardcoded localhost URLs (should be configurable)
        if "localhost" in line and "OLLAMA_URL" not in line and "#" not in line.split("localhost")[0]:
            if "http://localhost" in line and "=" in line:
                pass  # Defining the URL constant is fine

        # Division that might be integer division in Python 3
        # (not very useful, skip)

        # TODO/FIXME/HACK/XXX markers
        stripped = line.strip()
        for marker in ["TODO", "FIXME", "HACK", "XXX", "BUG"]:
            if marker in stripped and stripped.lstrip("#").strip().startswith(marker):
                issues.append({
                    "severity": "info",
                    "type": "todo_marker",
                    "message": stripped.strip("# ").strip(),
                    "line": i,
                })

        # Subprocess with shell=True (security risk)
        if "shell=True" in line:
            issues.append({
                "severity": "warning",
                "type": "shell_injection_risk",
                "message": "subprocess with shell=True — potential command injection",
                "line": i,
            })

        # Broad timeout values (> 600 seconds)
        if "timeout=" in line:
            try:
                # Extract timeout value
                after = line.split("timeout=")[1]
                val_str = ""
                for ch in after:
                    if ch.isdigit() or ch == ".":
                        val_str += ch
                    else:
                        break
                if val_str and float(val_str) > 600:
                    issues.append({
                        "severity": "info",
                        "type": "long_timeout",
                        "message": f"Timeout of {val_str}s — is this intentional?",
                        "line": i,
                    })
            except (ValueError, IndexError):
                pass

    return issues


def run_tier1(filepath):
    """Run all Tier 1 (static) checks on a file."""
    all_issues = []
    all_issues.extend(check_syntax(filepath))
    if not any(i["type"] == "syntax_error" for i in all_issues):
        all_issues.extend(check_ast_patterns(filepath))
    all_issues.extend(check_common_patterns(filepath))
    return all_issues


# ── Tier 2: LLM-Powered Review ──────────────────────────────────────────────

LLM_REVIEW_PROMPT = """\
You are a code reviewer doing a bug hunt. Review this Python file for:

1. BUGS: Logic errors, off-by-one errors, race conditions, unhandled edge cases
2. CRASHES: Missing null checks, index out of bounds, type mismatches
3. DATA LOSS: Files not closed, results not saved, checkpoints that could corrupt
4. SECURITY: Command injection, path traversal, secrets in code
5. CONSISTENCY: Does this file's patterns match what you'd expect from the imports and structure?

File: {filename}
```python
{code}
```

If you find issues, output a JSON array of objects:
{{"severity": "error|warning|info", "type": "short_type", "message": "description", "line": N}}

If the code looks clean, output an empty array: []

Focus on REAL bugs, not style preferences. Output ONLY the JSON array."""


def run_tier2(filepath):
    """LLM-powered semantic review."""
    try:
        with open(filepath) as f:
            code = f.read()
    except Exception:
        return []

    # Skip very large files (LLM context limit)
    if len(code) > 15000:
        # Send just the first and last 5000 chars with a note
        code = code[:5000] + "\n\n# ... [TRUNCATED — file is very long] ...\n\n" + code[-5000:]

    prompt = LLM_REVIEW_PROMPT.format(
        filename=filepath.name,
        code=code,
    )

    payload = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2000}
    })

    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        resp = data.get("response", "")

        # Parse JSON from response
        text = resp.strip()
        # Strip think tags
        while "<think>" in text:
            start = text.index("<think>")
            end = text.index("</think>", start) + len("</think>") if "</think>" in text[start:] else len(text)
            text = text[:start] + text[end:]
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [r for r in result if isinstance(r, dict)]
        except json.JSONDecodeError:
            # Try finding [...] in text
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                try:
                    result = json.loads(text[start:end+1])
                    if isinstance(result, list):
                        return [r for r in result if isinstance(r, dict)]
                except json.JSONDecodeError:
                    pass
        return []
    except Exception:
        return []


# ── Cross-file consistency checks ────────────────────────────────────────────

def check_cross_file_consistency(files):
    """Check for inconsistencies across files."""
    issues = []

    # Collect all OLLAMA_URL definitions
    ollama_urls = {}
    for f in files:
        try:
            with open(f) as fh:
                for i, line in enumerate(fh, 1):
                    if "OLLAMA_URL" in line and "=" in line and not line.strip().startswith("#"):
                        ollama_urls[f.name] = (line.strip(), i)
        except Exception:
            pass

    # Check they're all the same
    unique_urls = set(v[0] for v in ollama_urls.values())
    if len(unique_urls) > 1:
        issues.append({
            "severity": "warning",
            "type": "inconsistent_urls",
            "message": f"OLLAMA_URL defined differently across files: {list(ollama_urls.keys())}",
            "files": {k: v[0] for k, v in ollama_urls.items()},
        })

    # Collect all MODEL definitions
    models = {}
    for f in files:
        try:
            with open(f) as fh:
                for i, line in enumerate(fh, 1):
                    stripped = line.strip()
                    if stripped.startswith("MODEL = ") or stripped.startswith("MODEL="):
                        models.setdefault(f.name, []).append((stripped, i))
        except Exception:
            pass

    # Check WEIGHT_GRID consistency
    grids = {}
    for f in files:
        try:
            with open(f) as fh:
                for i, line in enumerate(fh, 1):
                    if "WEIGHT_GRID" in line and "=" in line and "[" in line:
                        grids[f.name] = (line.strip(), i)
        except Exception:
            pass

    unique_grids = set(v[0] for v in grids.values())
    if len(unique_grids) > 1:
        issues.append({
            "severity": "warning",
            "type": "inconsistent_weight_grid",
            "message": f"WEIGHT_GRID defined differently across files: {list(grids.keys())}",
            "files": {k: v[0] for k, v in grids.items()},
        })

    return issues


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(file_results, cross_file_issues):
    """Generate the pest control report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm_model": LLM_MODEL,
        "files_scanned": len(file_results),
        "total_issues": sum(len(r["tier1"]) + len(r["tier2"]) for r in file_results.values()),
        "errors": 0,
        "warnings": 0,
        "info": 0,
        "cross_file_issues": cross_file_issues,
        "files": {},
    }

    for filepath, result in file_results.items():
        all_issues = result["tier1"] + result["tier2"]
        for issue in all_issues:
            sev = issue.get("severity", "info")
            if sev == "error":
                report["errors"] += 1
            elif sev == "warning":
                report["warnings"] += 1
            else:
                report["info"] += 1

        if all_issues:  # Only include files with issues
            report["files"][str(filepath)] = {
                "tier1_issues": result["tier1"],
                "tier2_issues": result["tier2"],
                "total": len(all_issues),
            }

    return report


# ── Main ─────────────────────────────────────────────────────────────────────

def scan(files, use_llm=True):
    """Run pest control scan on given files."""
    file_results = {}

    for filepath in files:
        rel = filepath.relative_to(PROJECT)
        print(f"  [{rel}]", end=" ", flush=True)

        tier1 = run_tier1(filepath)
        tier1_summary = f"T1:{len(tier1)}"

        tier2 = []
        if use_llm:
            tier2 = run_tier2(filepath)
            tier2_summary = f"T2:{len(tier2)}"
        else:
            tier2_summary = "T2:skip"

        total = len(tier1) + len(tier2)
        status = "CLEAN" if total == 0 else f"{total} issues"
        errors = sum(1 for i in tier1 + tier2 if i.get("severity") == "error")
        if errors:
            status = f"{errors} ERRORS + {total - errors} other"

        print(f"{tier1_summary} {tier2_summary} → {status}")
        file_results[filepath] = {"tier1": tier1, "tier2": tier2}

    return file_results


def main():
    args = set(sys.argv[1:])

    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    use_llm = "--no-llm" not in args
    specific_file = None
    if "--file" in args:
        idx = sys.argv.index("--file")
        specific_file = PROJECT / sys.argv[idx + 1]

    if "--watch" in args:
        print(f"Pest Control: watching for changes (Ctrl+C to stop)")
        print(f"LLM model: {LLM_MODEL}")
        print()
        last_scan = time.time()

        while True:
            # Check files modified in last 2 minutes
            files = get_python_files(since_minutes=2)
            if files:
                print(f"\n[{time.strftime('%H:%M:%S')}] {len(files)} files changed:")
                file_results = scan(files, use_llm=use_llm)
                cross = check_cross_file_consistency(get_python_files())
                report = generate_report(file_results, cross)
                REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(REPORT_PATH, "w") as f:
                    json.dump(report, f, indent=2)
                if report["errors"] or report["warnings"]:
                    print(f"\n  ⚠ {report['errors']} errors, {report['warnings']} warnings")
            time.sleep(60)

    elif specific_file:
        if not specific_file.exists():
            print(f"File not found: {specific_file}")
            sys.exit(1)
        print(f"Pest Control: scanning {specific_file.name}")
        file_results = scan([specific_file], use_llm=use_llm)
        cross = []
        report = generate_report(file_results, cross)
        print(f"\nResult: {report['errors']} errors, {report['warnings']} warnings, {report['info']} info")

    else:
        if "--full" in args:
            files = get_python_files()
            print(f"Pest Control: full scan of {len(files)} Python files")
        else:
            # Default: files modified in last 24 hours
            files = get_python_files(since_minutes=24*60)
            if not files:
                files = get_python_files()
            print(f"Pest Control: scanning {len(files)} Python files")

        print(f"LLM model: {LLM_MODEL}" if use_llm else "LLM: disabled (--no-llm)")
        print()

        file_results = scan(files, use_llm=use_llm)
        cross = check_cross_file_consistency(get_python_files())
        report = generate_report(file_results, cross)

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*50}")
        print(f"PEST CONTROL REPORT")
        print(f"{'='*50}")
        print(f"Files scanned: {report['files_scanned']}")
        print(f"Errors: {report['errors']}")
        print(f"Warnings: {report['warnings']}")
        print(f"Info: {report['info']}")

        if cross:
            print(f"\nCross-file issues:")
            for issue in cross:
                print(f"  [{issue['severity']}] {issue['message']}")

        if report["files"]:
            print(f"\nPer-file:")
            for fpath, fdata in report["files"].items():
                fname = Path(fpath).name
                print(f"\n  {fname}:")
                for issue in fdata["tier1_issues"] + fdata["tier2_issues"]:
                    sev = issue.get("severity", "info").upper()
                    line = issue.get("line", "?")
                    msg = issue.get("message", "")
                    print(f"    [{sev}] L{line}: {msg}")
        else:
            print("\nAll clean!")

        print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
