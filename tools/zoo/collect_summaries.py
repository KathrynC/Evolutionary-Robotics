#!/usr/bin/env python3
"""
tools/zoo/collect_summaries.py

Repo extension:

Aggregates summary.json files produced by tools/zoo/run_zoo.py into TSV/CSV for analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


import os
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry_root", default="artifacts/telemetry", help="root dir: <variant>/<run>/summary.json")
    ap.add_argument("--out", default="artifacts/rules/zoo_scores.tsv", help="output TSV")
    args = ap.parse_args()

    root = Path(args.telemetry_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary in sorted(root.glob("*/*/summary.json")):
        d = json.loads(summary.read_text(encoding="utf-8"))
        vid = d.get("variant_id", summary.parents[1].name)
        run_id = d.get("run_id", summary.parent.name)
        dx = (d.get("delta") or {}).get("dx")
        upright = d.get("upright_fraction")
        roll = d.get("max_abs_roll")
        pitch = d.get("max_abs_pitch")
        rows.append((vid, run_id, dx, upright, roll, pitch, str(summary)))

    lines = ["variant_id\trun_id\tdx\tupright_fraction\tmax_abs_roll\tmax_abs_pitch\tsummary_path\n"]
    for vid, run_id, dx, upright, roll, pitch, sp in rows:
        def f(x):
            return "" if x is None else f"{x:.6g}"
        lines.append(f"{vid}\t{run_id}\t{f(dx)}\t{f(upright)}\t{f(roll)}\t{f(pitch)}\t{sp}\n")

    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
