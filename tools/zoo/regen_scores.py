#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

SCORE_KEYS = ("score", "fitness", "total_score", "reward")


def find_score(d: Any) -> Optional[float]:
    if isinstance(d, dict):
        for k in SCORE_KEYS:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    pass
        for k in ("metrics", "result", "summary", "eval"):
            if k in d:
                s = find_score(d[k])
                if s is not None:
                    return s
        for v in d.values():
            s = find_score(v)
            if s is not None:
                return s
    elif isinstance(d, list):
        for v in d:
            s = find_score(v)
            if s is not None:
                return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="artifacts/rules/zoo", help="dir containing variant_*.json")
    ap.add_argument("--out", default="artifacts/rules/zoo_scores.tsv", help="output TSV path")
    args = ap.parse_args()

    vdir = Path(args.variants)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(vdir.glob("variant_*.json")):
        vid = p.stem.replace("variant_", "")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        score = find_score(data)
        rows.append((vid, score, str(p)))

    lines = ["variant_id\tscore\tpath\n"]
    for vid, score, path in rows:
        sval = "" if score is None else f"{score:.6g}"
        lines.append(f"{vid}\t{sval}\t{path}\n")

    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
