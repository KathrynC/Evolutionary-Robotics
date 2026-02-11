#!/usr/bin/env python3
"""tools.replay_a_mode

Replay utilities for A-mode experiments (loading traces and reproducing key views).
"""

import argparse, glob, json, os
from pathlib import Path

def latest_trace():
    paths = sorted(glob.glob("artifacts/traces/run*.jsonl"))
    if not paths:
        raise SystemExit("No traces found in artifacts/traces (run simulation.py first).")
    return paths[-1]

def fmt_cmd(mt):
    if not isinstance(mt, dict) or not mt:
        return ""
    # Prefer these two if present, otherwise show up to 2 keys
    keys = ["Torso_BackLeg", "Torso_FrontLeg"]
    out = []
    for k in keys:
        if k in mt:
            out.append(f"{k}={mt[k]:+0.3f}")
    if not out:
        for k in list(mt.keys())[:2]:
            try:
                out.append(f"{k}={float(mt[k]):+0.3f}")
            except Exception:
                out.append(f"{k}={mt[k]}")
    return "  ".join(out)

def get_neuron(ev, nid):
    nn = (ev.get("nn") or {})
    neurons = (nn.get("neurons") or {})
    nd = neurons.get(str(nid)) or neurons.get(nid)
    if isinstance(nd, dict):
        return nd.get("value")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="Trace path (default: latest)")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--watch", default="3,4", help="Comma-separated neuron IDs to watch")
    args = ap.parse_args()

    path = args.path or latest_trace()
    watch = [int(x) for x in args.watch.split(",") if x.strip()]

    print(f"trace: {path}")
    hdr = f"{'i':>6} {'phase':>10} {'target':>9} " + " ".join([f"n{w:>2}" for w in watch]) + "   cmd"
    print(hdr)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("type") != "step":
                continue
            i = int(ev.get("i", -1))
            if i < args.start:
                continue
            if args.end is not None and i > args.end:
                break
            if args.every > 1 and (i - args.start) % args.every != 0:
                continue

            phase = ev.get("phase", None)
            target = ev.get("target", None)

            vals = []
            for w in watch:
                v = get_neuron(ev, w)
                vals.append("   ?" if v is None else f"{v:+0.3f}")

            cmd = fmt_cmd(ev.get("motor_targets"))
            phs = " " * 10 if phase is None else f"{phase:10.4f}"
            tgt = " " * 9 if target is None else f"{target:9.3f}"
            print(f"{i:6d} {phs} {tgt} " + " ".join(vals) + ("   " + cmd if cmd else ""))

if __name__ == "__main__":
    main()
