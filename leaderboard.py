import json
from pathlib import Path

BASE = Path("/Users/gardenofcomputation/pybullet_test/Evolutionary-Robotics/artifacts")

conditions = {
    "verbs": "structured_random_verbs.json",
    "theorems": "structured_random_theorems.json",
    "bible": "structured_random_bible.json",
    "places": "structured_random_places.json",
    "baseline": "structured_random_baseline.json",
}

pool = []
for cond, fname in conditions.items():
    with open(BASE / fname) as f:
        for r in json.load(f):
            r["condition"] = cond
            pool.append(r)

print(f"Total gaits: {len(pool)}\n")

# 1. Top 20 by |DX|
print("=" * 80)
print("TOP 20 BY |DX| (DISPLACEMENT)")
print("=" * 80)
ranked = sorted(pool, key=lambda r: abs(r["dx"]), reverse=True)
print(f"{'#':>3}  {'Cond':<10} {'DX':>8}  {'Seed'}")
print("-" * 80)
for i, r in enumerate(ranked[:20], 1):
    seed = str(r["seed"])[:65]
    print(f"{i:3d}  {r['condition']:<10} {r['dx']:+8.2f}  {seed}")

# 2. Top 20 by Speed
print(f"\n{'=' * 80}")
print("TOP 20 BY SPEED")
print("=" * 80)
ranked = sorted(pool, key=lambda r: r["speed"], reverse=True)
print(f"{'#':>3}  {'Cond':<10} {'Speed':>8}  {'Seed'}")
print("-" * 80)
for i, r in enumerate(ranked[:20], 1):
    seed = str(r["seed"])[:65]
    print(f"{i:3d}  {r['condition']:<10} {r['speed']:8.3f}  {seed}")

# 3. Top 20 by Efficiency
print(f"\n{'=' * 80}")
print("TOP 20 BY EFFICIENCY")
print("=" * 80)
ranked = sorted(pool, key=lambda r: r["efficiency"], reverse=True)
print(f"{'#':>3}  {'Cond':<10} {'Effic':>10}  {'Seed'}")
print("-" * 80)
for i, r in enumerate(ranked[:20], 1):
    seed = str(r["seed"])[:65]
    print(f"{i:3d}  {r['condition']:<10} {r['efficiency']:10.6f}  {seed}")

# 4. Top 20 by Phase Lock
print(f"\n{'=' * 80}")
print("TOP 20 BY PHASE LOCK")
print("=" * 80)
ranked = sorted(pool, key=lambda r: r["phase_lock"], reverse=True)
print(f"{'#':>3}  {'Cond':<10} {'PhaseLk':>8}  {'Seed'}")
print("-" * 80)
for i, r in enumerate(ranked[:20], 1):
    seed = str(r["seed"])[:65]
    print(f"{i:3d}  {r['condition']:<10} {r['phase_lock']:8.4f}  {seed}")

# 5. Bottom 20 (dead/worst)
print(f"\n{'=' * 80}")
print("BOTTOM 20 BY |DX| (DEAD / WORST GAITS)")
print("=" * 80)
ranked = sorted(pool, key=lambda r: abs(r["dx"]))
print(f"{'#':>3}  {'Cond':<10} {'DX':>8}  {'Seed'}")
print("-" * 80)
for i, r in enumerate(ranked[:20], 1):
    seed = str(r["seed"])[:65]
    print(f"{i:3d}  {r['condition']:<10} {r['dx']:+8.2f}  {seed}")

# 6. Condition breakdown
print(f"\n{'=' * 80}")
print("CONDITION BREAKDOWN")
print("=" * 80)
for cond in ["verbs", "theorems", "bible", "places", "baseline"]:
    entries = [r for r in pool if r["condition"] == cond]
    print(f"  {cond}: {len(entries)} gaits")
