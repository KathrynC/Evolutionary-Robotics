"""tools.gait_zoo

Utilities for working with the "gait zoo": enumerating/expanding gait variants,
running batches, and writing artifact outputs for later analysis.
"""

import argparse, json, os, random, subprocess, sys
from pathlib import Path

def load_rb(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def save_rb(j, p: Path):
    p.write_text(json.dumps(j, indent=2, sort_keys=True), encoding="utf-8")

def rotate_bins(rb, k):
    bins = int(rb["bins"])
    rules = rb["rules"]
    out = dict(rb)
    out_rules = {}
    for b_str, mp in rules.items():
        b = int(b_str)
        out_rules[str((b + k) % bins)] = mp
    out["rules"] = out_rules
    out["meta"] = {**rb.get("meta", {}), "rotate_bins": k}
    return out

def swap_motors(rb):
    out = json.loads(json.dumps(rb))
    for b, mp in out["rules"].items():
        for c, row in mp.items():
            back = float(row.get("Torso_BackLeg", 0.0))
            front = float(row.get("Torso_FrontLeg", 0.0))
            row["Torso_BackLeg"], row["Torso_FrontLeg"] = front, back
    out["meta"] = {**rb.get("meta", {}), "swap_motors": True}
    return out

def flip_one(rb, which):
    out = json.loads(json.dumps(rb))
    for b, mp in out["rules"].items():
        for c, row in mp.items():
            row[which] = -float(row.get(which, 0.0))
    out["meta"] = {**rb.get("meta", {}), "flip": which}
    return out

def quantize(rb, levels):
    out = json.loads(json.dumps(rb))
    vals = []
    for b, mp in out["rules"].items():
        for c, row in mp.items():
            vals.append(abs(float(row.get("Torso_BackLeg", 0.0))))
            vals.append(abs(float(row.get("Torso_FrontLeg", 0.0))))
    mx = max(vals) if vals else 1.0

    if levels == 3:
        qs = [-mx, 0.0, mx]
    elif levels == 5:
        qs = [-mx, -mx / 2.0, 0.0, mx / 2.0, mx]
    else:
        return out

    def q(v):
        v = float(v)
        return min(qs, key=lambda a: abs(a - v))

    for b, mp in out["rules"].items():
        for c, row in mp.items():
            row["Torso_BackLeg"] = q(row.get("Torso_BackLeg", 0.0))
            row["Torso_FrontLeg"] = q(row.get("Torso_FrontLeg", 0.0))

    out["meta"] = {**rb.get("meta", {}), "quantize_levels": levels}
    return out

def run_one(rule_path: Path):
    env = os.environ.copy()
    env["MODE"] = "B"
    env["RULEBOOK"] = str(rule_path)
    p = subprocess.run([sys.executable, "simulation.py"], capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rulebook", default="artifacts/rules/rulebook_bins64.json")
    ap.add_argument("--outdir", default="artifacts/rules/zoo")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    base_path = Path(args.rulebook)
    if not base_path.exists():
        print("Missing rulebook:", base_path)
        sys.exit(2)

    base = load_rb(base_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    makers = []
    bins = int(base.get("bins", 64))
    step = max(1, bins // 8)

    for k in range(0, bins, step):
        makers.append(lambda rb, k=k: rotate_bins(rb, k))

    makers += [
        swap_motors,
        lambda rb: flip_one(rb, "Torso_BackLeg"),
        lambda rb: flip_one(rb, "Torso_FrontLeg"),
        lambda rb: quantize(rb, 3),
        lambda rb: quantize(rb, 5),
    ]

    random.shuffle(makers)

    for i in range(args.n):
        mk = makers[i % len(makers)]
        rb = mk(base)
        rp = outdir / f"variant_{i:03d}.json"
        save_rb(rb, rp)
        code, out, err = run_one(rp)
        print(f"[{i:03d}] {rp.name} rc={code}")
        if out.strip():
            print(out.strip().splitlines()[-1])
        if code != 0 and err.strip():
            print(err.strip().splitlines()[-1])

if __name__ == "__main__":
    main()
