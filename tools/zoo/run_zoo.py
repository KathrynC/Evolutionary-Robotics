from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

def make_summary_from_jsonl(run_dir: Path, variant_id: str, run_id: str) -> bool:
    jl = run_dir / "telemetry.jsonl"
    sj = run_dir / "summary.json"
    if sj.exists():
        return True
    if not jl.exists():
        return False
    recs = []
    for line in jl.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    if not recs:
        return False
    t0 = recs[0].get("base", {})
    tN = recs[-1].get("base", {})
    def f(d, k):
        try:
            return float(d.get(k))
        except Exception:
            return None
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    tip_steps = 0
    for r in recs:
        rpy = r.get("rpy", {})
        rr = f(rpy, "r")
        pp = f(rpy, "p")
        if rr is not None:
            max_abs_roll = max(max_abs_roll, abs(rr))
        if pp is not None:
            max_abs_pitch = max(max_abs_pitch, abs(pp))
        if rr is not None and pp is not None:
            if abs(rr) > 1.0 or abs(pp) > 1.0:
                tip_steps += 1
    upright_fraction = 1.0 - tip_steps / float(len(recs)) if recs else None
    dx = f(tN, "x") - f(t0, "x") if f(tN, "x") is not None and f(t0, "x") is not None else None
    dy = f(tN, "y") - f(t0, "y") if f(tN, "y") is not None and f(t0, "y") is not None else None
    dz = f(tN, "z") - f(t0, "z") if f(tN, "z") is not None and f(t0, "z") is not None else None
    final_rpy = recs[-1].get("rpy", {})
    summary = {
        "variant_id": variant_id,
        "run_id": run_id,
        "steps_logged": len(recs),
        "max_abs_roll": max_abs_roll,
        "max_abs_pitch": max_abs_pitch,
        "upright_fraction": upright_fraction,
        "delta": {"dx": dx, "dy": dy, "dz": dz},
        "final_rpy": final_rpy,
    }
    sj.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants_dir", default="artifacts/rules/zoo")
    ap.add_argument("--variant_glob", default="variant_*.json")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--simulation", default="python3 simulation.py")
    ap.add_argument("--telemetry_root", default="artifacts/telemetry")
    ap.add_argument("--telemetry_every", type=int, default=10)
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    vdir = Path(args.variants_dir)
    variants = sorted(vdir.glob(args.variant_glob))
    if args.limit and args.limit > 0:
        variants = variants[:args.limit]

    sim_cmd = args.simulation.split()
    telemetry_root = Path(args.telemetry_root)
    telemetry_root.mkdir(parents=True, exist_ok=True)

    for vp in variants:
        variant_id = vp.stem.replace("variant_", "")
        for r in range(args.runs):
            run_id = f"{tag}_r{r:02d}"
            run_dir = telemetry_root / variant_id / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["GAIT_VARIANT_PATH"] = str(vp)
            env["TELEMETRY"] = "1"
            env["TELEMETRY_EVERY"] = str(args.telemetry_every)
            env["TELEMETRY_OUT"] = str(telemetry_root)
            env["TELEMETRY_VARIANT_ID"] = str(variant_id)
            env["TELEMETRY_RUN_ID"] = str(run_id)
            if args.headless:
                env["HEADLESS"] = "1"
            log_path = run_dir / "stdout_stderr.log"
            with open(log_path, "w", encoding="utf-8") as f:
                p = subprocess.run(sim_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            status = {
                "variant_id": variant_id,
                "run_id": run_id,
                "variant_path": str(vp),
                "returncode": p.returncode,
                "log_path": str(log_path),
            }
            (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            make_summary_from_jsonl(run_dir, variant_id, run_id)

if __name__ == "__main__":
    main()
