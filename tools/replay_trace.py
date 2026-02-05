#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TRACE_DIR_DEFAULT = Path("artifacts/traces")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def latest_trace_file(trace_dir: Path) -> Path:
    files = sorted(trace_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No run_*.jsonl files found in {trace_dir}")
    return files[0]


def iter_events(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                eprint(f"[warn] JSON decode failed at {path}:{ln}")
                continue
            yield ev


def fmt(x: Any, width: int = 7, prec: int = 3) -> str:
    if x is None:
        return " " * (width - 1) + "?"
    try:
        xf = float(x)
    except Exception:
        return str(x)[:width].rjust(width)
    if math.isnan(xf):
        return " " * (width - 3) + "NaN"
    s = f"{xf:.{prec}f}"
    return s.rjust(width)


@dataclass
class StepView:
    i: int
    target: Optional[float]
    n_neurons: int
    n_synapses: int
    sensors_on: int
    motors_on: int
    motors_mean: Optional[float]
    motors_max: Optional[float]
    top: List[Tuple[str, Optional[float], str]]  # (id, value, type)


def summarize_step(ev: Dict[str, Any], topk: int = 3, motor_eps: float = 0.05) -> Optional[StepView]:
    if ev.get("type") != "step":
        return None

    i = ev.get("i", -1)
    target = ev.get("target", None)

    nn = ev.get("nn") or {}
    neurons: Dict[str, Dict[str, Any]] = nn.get("neurons") or {}
    synapses: List[Dict[str, Any]] = nn.get("synapses") or []

    # Counts
    n_neurons = len(neurons)
    n_synapses = len(synapses)

    # Sensor/motor activity
    sensors_on = 0
    motors: List[Tuple[str, float, str]] = []

    for nid, nd in neurons.items():
        if not isinstance(nd, dict):
            continue
        ntype = str(nd.get("type") or "").lower()
        v = nd.get("value", None)
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue

        if "sensor" in ntype and vf > 0:
            sensors_on += 1

        if "motor" in ntype:
            if abs(vf) >= motor_eps:
                motors.append((str(nid), vf, ntype))

    motors_on = len(motors)
    motors_mean = (sum(v for _, v, _ in motors) / motors_on) if motors_on else None
    motors_max = (max(abs(v) for _, v, _ in motors)) if motors_on else None

    # Top activations (by abs value) across all neurons with numeric values
    scored: List[Tuple[str, float, str]] = []
    for nid, nd in neurons.items():
        if not isinstance(nd, dict):
            continue
        v = nd.get("value", None)
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        ntype = str(nd.get("type") or "").lower()
        scored.append((str(nid), vf, ntype))
    scored.sort(key=lambda t: abs(t[1]), reverse=True)
    top = [(nid, vf, ntype) for (nid, vf, ntype) in scored[:topk]]

    return StepView(
        i=int(i),
        target=float(target) if target is not None else None,
        n_neurons=n_neurons,
        n_synapses=n_synapses,
        sensors_on=sensors_on,
        motors_on=motors_on,
        motors_mean=motors_mean,
        motors_max=motors_max,
        top=top,
    )


def cmd_replay(path: Path, every: int, start: int, end: Optional[int], topk: int, motor_eps: float, watch: List[str]):
    # Header
    print(
        " i".rjust(6),
        "target".rjust(8),
        "N".rjust(6),
        "S".rjust(6),
        "sens+".rjust(7),
        "mot+".rjust(7),
        "motμ".rjust(8),
        "mot|max|".rjust(10),
        "top".ljust(1),
        sep=" ",
    )

    for ev in iter_events(path):
        sv = summarize_step(ev, topk=topk, motor_eps=motor_eps)
        if sv is None:
            continue
        if sv.i < start:
            continue
        if end is not None and sv.i > end:
            break
        if every > 1 and (sv.i - start) % every != 0:
            continue

        top_str = ", ".join(
            f"{nid}:{(f'{v:+.2f}' if v is not None else '?')}[{t.replace('neuron','').strip() or '?'}]"
            for nid, v, t in sv.top
        )

        line = (
            str(sv.i).rjust(6),
            fmt(sv.target, width=8, prec=3),
            str(sv.n_neurons).rjust(6),
            str(sv.n_synapses).rjust(6),
            str(sv.sensors_on).rjust(7),
            str(sv.motors_on).rjust(7),
            fmt(sv.motors_mean, width=8, prec=3),
            fmt(sv.motors_max, width=10, prec=3),
            top_str,
        )

        print(*line, sep=" ")

        # Optional: watch specific neuron ids each printed step
        if watch:
            nn = (ev.get("nn") or {}).get("neurons") or {}
            watch_vals = []
            for wid in watch:
                nd = nn.get(wid) or nn.get(str(wid))
                v = nd.get("value") if isinstance(nd, dict) else None
                watch_vals.append(f"{wid}={v}")
            print(" " * 2 + "watch: " + "  ".join(watch_vals))


def cmd_summary(path: Path, motor_eps: float):
    # Streaming summary: per-neuron stats
    n_count: Dict[str, int] = {}
    n_sum: Dict[str, float] = {}
    n_abs_sum: Dict[str, float] = {}
    n_min: Dict[str, float] = {}
    n_max: Dict[str, float] = {}
    n_type: Dict[str, str] = {}

    steps = 0
    for ev in iter_events(path):
        if ev.get("type") != "step":
            continue
        steps += 1
        nn = ev.get("nn") or {}
        neurons = nn.get("neurons") or {}
        for nid, nd in neurons.items():
            if not isinstance(nd, dict):
                continue
            v = nd.get("value", None)
            if v is None:
                continue
            try:
                vf = float(v)
            except Exception:
                continue

            nid = str(nid)
            n_type[nid] = str(nd.get("type") or "").lower()
            n_count[nid] = n_count.get(nid, 0) + 1
            n_sum[nid] = n_sum.get(nid, 0.0) + vf
            n_abs_sum[nid] = n_abs_sum.get(nid, 0.0) + abs(vf)
            n_min[nid] = min(n_min.get(nid, vf), vf)
            n_max[nid] = max(n_max.get(nid, vf), vf)

    print(f"Trace: {path}")
    print(f"Steps: {steps}")
    print()

    # Show top motors by average abs value
    motors = []
    for nid, c in n_count.items():
        t = n_type.get(nid, "")
        if "motor" in t:
            mean = n_sum[nid] / c
            mean_abs = n_abs_sum[nid] / c
            motors.append((mean_abs, nid, mean, n_min[nid], n_max[nid], c))
    motors.sort(reverse=True)

    print("Top motor neurons by avg |value| (motor_eps affects replay only):")
    print(" avg|v|".rjust(9), " id".rjust(6), " mean".rjust(9), " min".rjust(9), " max".rjust(9), " n".rjust(7))
    for mean_abs, nid, mean, vmin, vmax, c in motors[:10]:
        print(fmt(mean_abs, 9, 4), str(nid).rjust(6), fmt(mean, 9, 4), fmt(vmin, 9, 4), fmt(vmax, 9, 4), str(c).rjust(7))

    # Show top sensors by “on-rate” (fraction of >0)
    sensors = []
    for nid, c in n_count.items():
        t = n_type.get(nid, "")
        if "sensor" in t:
            # crude on-rate estimate: mean shifted to [0,1] doesn’t apply; count positives directly would require another pass
            # so we approximate with avg value (often -1/ +1)
            mean = n_sum[nid] / c
            sensors.append((mean, nid, n_min[nid], n_max[nid], c))
    sensors.sort(reverse=True)

    print()
    print("Sensor neuron avg value (often -1/+1):")
    print(" mean".rjust(9), " id".rjust(6), " min".rjust(9), " max".rjust(9), " n".rjust(7))
    for mean, nid, vmin, vmax, c in sensors[:10]:
        print(fmt(mean, 9, 4), str(nid).rjust(6), fmt(vmin, 9, 4), fmt(vmax, 9, 4), str(c).rjust(7))


def main():
    ap = argparse.ArgumentParser(description="Replay or summarize Ludobots JSONL neuron traces.")
    ap.add_argument("--file", type=str, default=None, help="Path to a trace JSONL file.")
    ap.add_argument("--latest", action="store_true", help="Use latest run_*.jsonl from artifacts/traces.")
    ap.add_argument("--trace-dir", type=str, default=str(TRACE_DIR_DEFAULT), help="Trace directory (default: artifacts/traces).")

    sub = ap.add_subparsers(dest="cmd", required=False)

    rp = sub.add_parser("replay", help="Print a compact per-step view.")
    rp.add_argument("--every", type=int, default=10, help="Print every N steps (default 10). Use 1 for all.")
    rp.add_argument("--start", type=int, default=0, help="Start step index (default 0).")
    rp.add_argument("--end", type=int, default=None, help="End step index (inclusive).")
    rp.add_argument("--topk", type=int, default=3, help="Show top K neuron activations (default 3).")
    rp.add_argument("--motor-eps", type=float, default=0.05, help="Motor 'active' threshold (default 0.05).")
    rp.add_argument("--watch", type=str, default="", help="Comma-separated neuron IDs to print each line (e.g., '3,4').")

    sp = sub.add_parser("summary", help="Streaming summary stats over the trace.")
    sp.add_argument("--motor-eps", type=float, default=0.05, help="Motor 'active' threshold note (default 0.05).")

    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    if args.latest:
        path = latest_trace_file(trace_dir)
    elif args.file:
        path = Path(args.file)
    else:
        # default behavior: latest + replay
        path = latest_trace_file(trace_dir)
        args.cmd = args.cmd or "replay"

    if not path.exists():
        raise FileNotFoundError(path)

    cmd = args.cmd or "replay"
    if cmd == "replay":
        watch = [s.strip() for s in (args.watch.split(",") if args.watch else []) if s.strip()]
        cmd_replay(path, every=args.every, start=args.start, end=args.end, topk=args.topk, motor_eps=args.motor_eps, watch=watch)
    elif cmd == "summary":
        cmd_summary(path, motor_eps=args.motor_eps)
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
