#!/usr/bin/env python3
"""
proximity_perturbation_study.py

Tests the smoothing hypothesis: do proximity sensors reduce the cliffiness
of the fitness landscape at known cliff points?

Protocol:
1. Pick the cliffiest point from atlas_cliffiness.json (idx 18, cliffiness 47.61m)
2. CONDITION A (touch-only, 6D): Perturb touch weights in N random directions,
   measure DX changes, compute cliffiness and sign flip rate
3. CONDITION B (touch+proximity, 22D): Same touch weights + K different random
   proximity weight vectors. Perturb the SAME touch-weight directions, measure
   DX changes. If proximity smooths the landscape, cliffiness should drop and
   sign flip rate should decrease.

The key insight: proximity sensors provide continuous distance signals near
contact. If a touch-only cliff is caused by a contact discontinuity (foot
barely touching vs barely not touching), the proximity signal provides a
gradient through that discontinuity, potentially stabilizing motor output.
"""

import json
import time
import numpy as np
from pathlib import Path

# Must be run from the project directory
PROJECT = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(PROJECT))

import pybullet as p
import pybullet_data
import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK

# ── Configuration ────────────────────────────────────────────────────────────

N_DIRECTIONS = 30       # Probe directions per condition (more = better stats)
R_PROBE = 0.05          # Perturbation radius (matches atlas)
N_PROX_VECTORS = 10     # Number of random proximity weight sets to test
RNG_SEED = 2024         # Reproducibility

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]
PROXIMITY_SENSORS = [5, 6, 7, 8, 9, 10, 11, 12]
MOTOR_NEURONS = [3, 4]

# Top cliffiest point from atlas (idx 18)
CLIFF_WEIGHTS = {
    "w03": -0.473, "w04": -0.315,
    "w13": -0.122, "w14": -0.149,
    "w23": -0.567, "w24":  0.242,
}

# ── Simulation helpers ───────────────────────────────────────────────────────

def write_brain_touch_only(weights):
    """Write classic 6-synapse brain.nndf."""
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for s in [0, 1, 2]:
            for m in [3, 4]:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def write_brain_extended(weights):
    """Write 22-synapse brain.nndf with touch + proximity."""
    proximity_neurons = [
        (5,  "Torso",    "front"),
        (6,  "Torso",    "back"),
        (7,  "Torso",    "left"),
        (8,  "Torso",    "right"),
        (9,  "Torso",    "up"),
        (10, "Torso",    "down"),
        (11, "BackLeg",  "down"),
        (12, "FrontLeg", "down"),
    ]
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for nid, linkName, rayDir in proximity_neurons:
            f.write(f'    <neuron name = "{nid}" type = "proximity" '
                    f'linkName = "{linkName}" rayDir = "{rayDir}" />\n')
        for s in [0, 1, 2]:
            for m in [3, 4]:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        for nid, _, _ in proximity_neurons:
            for m in [3, 4]:
                w = weights[f"wp{nid}_{m}"]
                f.write(f'    <synapse sourceNeuronName = "{nid}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def simulate_dx(brain_writer, weights):
    """Run a headless sim and return net x-displacement."""
    brain_writer(weights)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(c.DT)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
    for link in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, link, lateralFriction=mu, restitution=0.0)

    nn = NEURAL_NETWORK(str(PROJECT / "brain.nndf"))
    max_force = float(getattr(c, "MAX_FORCE", 150.0))

    x0 = p.getBasePositionAndOrientation(robotId)[0][0]

    for i in range(c.SIM_STEPS):
        for neuronName in nn.neurons:
            n_obj = nn.neurons[neuronName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes,
                                                p.POSITION_CONTROL,
                                                n_obj.Get_Value(), max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes,
                                                n_obj.Get_Value(), max_force)
        p.stepSimulation()
        nn.Update()

    xf = p.getBasePositionAndOrientation(robotId)[0][0]
    p.disconnect()
    return xf - x0


def random_unit_direction(ndim, rng):
    """Random unit vector in ndim-dimensional space."""
    v = rng.standard_normal(ndim)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.ones(ndim) / np.sqrt(ndim)


def perturb_touch_weights(base_weights, direction_6d, radius):
    """Perturb only the 6 touch weights along a direction."""
    w = dict(base_weights)
    for i, wn in enumerate(WEIGHT_NAMES):
        w[wn] = base_weights[wn] + radius * direction_6d[i]
    return w


# ── Main study ───────────────────────────────────────────────────────────────

def run_study():
    rng = np.random.default_rng(RNG_SEED)

    # Pre-generate shared probe directions (6D touch subspace)
    directions = np.array([random_unit_direction(6, rng) for _ in range(N_DIRECTIONS)])

    print("=" * 72)
    print("PROXIMITY SMOOTHING PERTURBATION STUDY")
    print("=" * 72)
    print(f"Cliff point weights: {CLIFF_WEIGHTS}")
    print(f"Probe radius: {R_PROBE}")
    print(f"Directions: {N_DIRECTIONS}")
    print(f"Proximity weight vectors: {N_PROX_VECTORS}")
    total_sims = 1 + N_DIRECTIONS + N_PROX_VECTORS * (1 + N_DIRECTIONS)
    print(f"Total simulations: {total_sims}")
    print()

    t_start = time.perf_counter()

    # ── CONDITION A: Touch-only (6D) ─────────────────────────────────────
    print("─── CONDITION A: Touch-only (6D) ───")
    base_dx_touch = simulate_dx(write_brain_touch_only, CLIFF_WEIGHTS)
    print(f"  Base DX: {base_dx_touch:+.4f}m")

    delta_dxs_touch = np.empty(N_DIRECTIONS)
    for k in range(N_DIRECTIONS):
        pw = perturb_touch_weights(CLIFF_WEIGHTS, directions[k], R_PROBE)
        dx = simulate_dx(write_brain_touch_only, pw)
        delta_dxs_touch[k] = dx - base_dx_touch

    cliffiness_touch = float(np.max(np.abs(delta_dxs_touch)))
    n_pos = np.sum(delta_dxs_touch > 0)
    n_neg = np.sum(delta_dxs_touch < 0)
    sign_flip_rate_touch = float(min(n_pos, n_neg)) / N_DIRECTIONS
    std_touch = float(np.std(delta_dxs_touch))
    mean_abs_touch = float(np.mean(np.abs(delta_dxs_touch)))

    print(f"  Cliffiness (max |ΔDX|): {cliffiness_touch:.4f}m")
    print(f"  Mean |ΔDX|: {mean_abs_touch:.4f}m")
    print(f"  Std ΔDX: {std_touch:.4f}m")
    print(f"  Sign split: {int(n_pos)}+ / {int(n_neg)}-")
    print(f"  Sign flip rate: {sign_flip_rate_touch:.3f}")
    print()

    # ── CONDITION B: Touch + Proximity (22D) ─────────────────────────────
    print("─── CONDITION B: Touch + Proximity (22D) ───")
    print(f"  Testing {N_PROX_VECTORS} random proximity weight vectors...")
    print()

    prox_results = []
    for v in range(N_PROX_VECTORS):
        # Generate random proximity weights
        prox_weights = {}
        for s in PROXIMITY_SENSORS:
            for m in MOTOR_NEURONS:
                prox_weights[f"wp{s}_{m}"] = float(rng.uniform(-1, 1))

        # Full extended weight dict
        ext_weights = dict(CLIFF_WEIGHTS)
        ext_weights.update(prox_weights)

        # Base DX with proximity
        base_dx_ext = simulate_dx(write_brain_extended, ext_weights)

        # Perturb SAME touch directions
        delta_dxs_ext = np.empty(N_DIRECTIONS)
        for k in range(N_DIRECTIONS):
            pw = dict(ext_weights)
            for i, wn in enumerate(WEIGHT_NAMES):
                pw[wn] = ext_weights[wn] + R_PROBE * directions[k][i]
            dx = simulate_dx(write_brain_extended, pw)
            delta_dxs_ext[k] = dx - base_dx_ext

        cliff_ext = float(np.max(np.abs(delta_dxs_ext)))
        n_pos_ext = np.sum(delta_dxs_ext > 0)
        n_neg_ext = np.sum(delta_dxs_ext < 0)
        sfr_ext = float(min(n_pos_ext, n_neg_ext)) / N_DIRECTIONS
        std_ext = float(np.std(delta_dxs_ext))
        mean_abs_ext = float(np.mean(np.abs(delta_dxs_ext)))

        prox_results.append({
            "vector_idx": v,
            "base_dx": base_dx_ext,
            "cliffiness": cliff_ext,
            "mean_abs_delta_dx": mean_abs_ext,
            "std_delta_dx": std_ext,
            "sign_flip_rate": sfr_ext,
            "sign_split": (int(n_pos_ext), int(n_neg_ext)),
            "delta_dxs": delta_dxs_ext.tolist(),
            "proximity_weights": prox_weights,
        })

        elapsed = time.perf_counter() - t_start
        print(f"  [{v+1:2d}/{N_PROX_VECTORS}] {elapsed:5.1f}s  "
              f"baseDX={base_dx_ext:+7.2f}  cliff={cliff_ext:6.3f}  "
              f"mean|Δ|={mean_abs_ext:6.3f}  sfr={sfr_ext:.2f}")

    # ── Analysis ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    cliff_vals = [r["cliffiness"] for r in prox_results]
    sfr_vals = [r["sign_flip_rate"] for r in prox_results]
    mean_abs_vals = [r["mean_abs_delta_dx"] for r in prox_results]

    print()
    print(f"{'Metric':<25s} {'Touch-only (6D)':>15s}   {'Touch+Prox (22D)':>20s}")
    print(f"{'─'*25} {'─'*15}   {'─'*20}")
    print(f"{'Cliffiness (max|ΔDX|)':<25s} {cliffiness_touch:>15.4f}m  "
          f" {np.median(cliff_vals):>8.4f}m (median)")
    print(f"{'Mean |ΔDX|':<25s} {mean_abs_touch:>15.4f}m  "
          f" {np.median(mean_abs_vals):>8.4f}m (median)")
    print(f"{'Std ΔDX':<25s} {std_touch:>15.4f}m  "
          f" {np.median([r['std_delta_dx'] for r in prox_results]):>8.4f}m (median)")
    print(f"{'Sign flip rate':<25s} {sign_flip_rate_touch:>15.3f}   "
          f" {np.median(sfr_vals):>8.3f} (median)")

    print()
    print("Proximity condition distribution:")
    print(f"  Cliffiness: min={min(cliff_vals):.4f}  "
          f"median={np.median(cliff_vals):.4f}  max={max(cliff_vals):.4f}")
    print(f"  Sign flip:  min={min(sfr_vals):.3f}  "
          f"median={np.median(sfr_vals):.3f}  max={max(sfr_vals):.3f}")
    print(f"  Mean |ΔDX|: min={min(mean_abs_vals):.4f}  "
          f"median={np.median(mean_abs_vals):.4f}  max={max(mean_abs_vals):.4f}")

    # Smoothing verdict
    print()
    cliff_reduction = (cliffiness_touch - np.median(cliff_vals)) / cliffiness_touch * 100
    sfr_reduction = (sign_flip_rate_touch - np.median(sfr_vals)) / max(sign_flip_rate_touch, 1e-9) * 100
    mean_reduction = (mean_abs_touch - np.median(mean_abs_vals)) / mean_abs_touch * 100

    print(f"Cliffiness change:  {cliff_reduction:+.1f}%  "
          f"({'SMOOTHER' if cliff_reduction > 0 else 'ROUGHER'})")
    print(f"Sign flip change:   {sfr_reduction:+.1f}%  "
          f"({'SMOOTHER' if sfr_reduction > 0 else 'ROUGHER'})")
    print(f"Mean |ΔDX| change:  {mean_reduction:+.1f}%  "
          f"({'SMOOTHER' if mean_reduction > 0 else 'ROUGHER'})")

    total_time = time.perf_counter() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_sims} sims, "
          f"{total_sims/total_time:.1f} sims/s)")

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "cliff_point": CLIFF_WEIGHTS,
        "r_probe": R_PROBE,
        "n_directions": N_DIRECTIONS,
        "n_prox_vectors": N_PROX_VECTORS,
        "touch_only": {
            "base_dx": base_dx_touch,
            "cliffiness": cliffiness_touch,
            "mean_abs_delta_dx": mean_abs_touch,
            "std_delta_dx": std_touch,
            "sign_flip_rate": sign_flip_rate_touch,
            "sign_split": (int(n_pos), int(n_neg)),
            "delta_dxs": delta_dxs_touch.tolist(),
        },
        "touch_plus_proximity": prox_results,
        "summary": {
            "cliffiness_reduction_pct": cliff_reduction,
            "sign_flip_reduction_pct": sfr_reduction,
            "mean_abs_reduction_pct": mean_reduction,
        },
        "directions": directions.tolist(),
    }

    out_path = PROJECT / "artifacts" / "proximity_perturbation_study.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    run_study()
