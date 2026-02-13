#!/usr/bin/env python3
"""tools.rule_table

Generate tabular views of gait rules/variants and export them for analysis.
"""

import argparse, glob, json, math
from collections import defaultdict

def latest_trace():
    """Return the path to the most recent trace file in artifacts/traces/."""
    paths = sorted(glob.glob("artifacts/traces/run*.jsonl"))
    if not paths:
        raise SystemExit("No traces found in artifacts/traces (run simulation.py first).")
    return paths[-1]

def get_neuron(ev, nid):
    """Extract a neuron's value from a trace event dict, or None if absent."""
    nn = (ev.get("nn") or {})
    neurons = (nn.get("neurons") or {})
    nd = neurons.get(str(nid)) or neurons.get(nid)
    if isinstance(nd, dict):
        return nd.get("value")
    return None

def main():
    """Build a phase-bin x contact-state rule table from a trace file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="Trace path (default: latest)")
    ap.add_argument("--bins", type=int, default=16, help="Phase bins over 0..2π")
    ap.add_argument("--sensors", default="0,1,2", help="Comma-separated sensor neuron ids")
    ap.add_argument("--motors", default="Torso_BackLeg,Torso_FrontLeg", help="Comma-separated motor target keys")
    args = ap.parse_args()

    path = args.path or latest_trace()
    sensors = [int(x) for x in args.sensors.split(",") if x.strip()]
    motor_keys = [x.strip() for x in args.motors.split(",") if x.strip()]

    # key = (phase_bin, contact_bits_tuple)
    acc = defaultdict(lambda: {"n":0, "sum":{k:0.0 for k in motor_keys}})

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("type") != "step":
                continue
            phase = ev.get("phase", None)
            if phase is None:
                continue

            # map phase to [0, 2π)
            ph = float(phase) % (2.0 * math.pi)
            b = int((ph / (2.0 * math.pi)) * args.bins)
            if b == args.bins:
                b = args.bins - 1

            bits = []
            for sid in sensors:
                v = get_neuron(ev, sid)
                # Treat >0 as "contact/on"
                bits.append(1 if (v is not None and float(v) > 0.0) else 0)
            bits = tuple(bits)

            mt = ev.get("motor_targets") or {}
            key = (b, bits)
            acc[key]["n"] += 1
            for mk in motor_keys:
                if mk in mt:
                    acc[key]["sum"][mk] += float(mt[mk])

    # Print
    print(f"trace: {path}")
    print(f"phase bins: {args.bins}  sensors: {sensors}  motors: {motor_keys}")
    print()
    print(f"{'bin':>3}  {'contacts':>10}  {'n':>6}  " + "  ".join([f"{mk:>16}" for mk in motor_keys]))

    for (b, bits) in sorted(acc.keys()):
        n = acc[(b,bits)]["n"]
        means = []
        for mk in motor_keys:
            means.append(acc[(b,bits)]["sum"][mk] / max(1, n))
        contacts = "".join(str(x) for x in bits)
        row = f"{b:3d}  {contacts:>10}  {n:6d}  " + "  ".join([f"{m:+16.3f}" for m in means])
        print(row)

if __name__ == "__main__":
    main()
