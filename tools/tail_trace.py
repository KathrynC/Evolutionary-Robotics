#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: tail_trace.py path/to/trace.jsonl")
        sys.exit(2)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            print(line.rstrip())

if __name__ == "__main__":
    main()
