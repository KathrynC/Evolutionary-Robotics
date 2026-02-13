#!/usr/bin/env python3
"""
random_search_analytics.py

Generate 5 random brains and analyze each with the full Beer-framework
analytics pipeline (outcome, contact, coordination, rotation_axis).

For each trial:
  1. Sample 6 random synaptic weights in [-1, 1]
  2. Write brain.nndf
  3. Run a headless 4000-step simulation with full-resolution telemetry
  4. Compute all 4 analytics pillars from the telemetry

Outputs:
  artifacts/telemetry/random_trial_0/ .. random_trial_4/  (telemetry JSONL)
  artifacts/random_search_analytics.json                   (combined results)
  stdout: summary table

Usage:
    python3 random_search_analytics.py
"""

import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from tools.telemetry.logger import TelemetryLogger
from compute_beer_analytics import load_telemetry, compute_all, DT, NumpyEncoder

NUM_TRIALS = 5
SENSOR_NEURONS = [0, 1, 2]
MOTOR_NEURONS = [3, 4]
TELEMETRY_ROOT = PROJECT / "artifacts" / "telemetry"
OUT_JSON = PROJECT / "artifacts" / "random_search_analytics.json"


def generate_random_weights():
    """Return a dict of 6 random weights in [-1, 1]."""
    weights = {}
    for s in SENSOR_NEURONS:
        for m in MOTOR_NEURONS:
            weights[f"w{s}{m}"] = random.uniform(-1, 1)
    return weights


def write_brain(weights):
    """Write brain.nndf with the given 6-synapse weights."""
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for s in SENSOR_NEURONS:
            for m in MOTOR_NEURONS:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def safe_get_base_pose(body_id):
    try:
        return p.getBasePositionAndOrientation(body_id)
    except Exception:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)


def run_trial(trial_name, weights, out_dir):
    """Run one headless simulation with full telemetry. Returns final position."""
    write_brain(weights)

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

    nn = NEURAL_NETWORK("brain.nndf")

    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetryLogger(
        robotId, out_dir, every=1,
        variant_id=trial_name, run_id="definitive",
        enabled=True,
    )

    max_force = float(getattr(c, "MAX_FORCE", 150.0))

    for i in range(c.SIM_STEPS):
        for neuronName in nn.neurons:
            n = nn.neurons[neuronName]
            if n.Is_Motor_Neuron():
                jn = n.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, n.Get_Value(), max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, p.POSITION_CONTROL,
                                                n.Get_Value(), max_force)
        p.stepSimulation()
        nn.Update()
        telemetry.log_step(i)

    telemetry.finalize()
    pos = safe_get_base_pose(robotId)[0]
    p.disconnect()
    return pos


def main():
    # Back up brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    results = []
    print(f"Running {NUM_TRIALS} random-search trials with full telemetry...\n")
    t_total = time.perf_counter()

    for trial in range(NUM_TRIALS):
        trial_name = f"random_trial_{trial}"
        weights = generate_random_weights()
        out_dir = TELEMETRY_ROOT / trial_name

        t0 = time.perf_counter()
        pos = run_trial(trial_name, weights, out_dir)
        elapsed = time.perf_counter() - t0

        # Compute Beer analytics from the telemetry we just wrote
        data = load_telemetry(trial_name)
        analytics = compute_all(data, DT)

        result = {
            "name": trial_name,
            "weights": weights,
            "analytics": analytics,
        }
        results.append(result)

        o = analytics["outcome"]
        coord = analytics["coordination"]
        contact = analytics["contact"]
        ra = analytics["rotation_axis"]
        print(f"  Trial {trial}  DX={o['dx']:+7.3f}  speed={o['mean_speed']:.4f}  "
              f"eff={o['distance_per_work']:.5f}  phase_lock={coord['phase_lock_score']:.3f}  "
              f"entropy={contact['contact_entropy_bits']:.2f}  "
              f"roll_dom={ra['axis_dominance'][0]:.2f}  ({elapsed:.2f}s)")

    total_elapsed = time.perf_counter() - t_total

    # Restore brain.nndf
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # Save combined JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    # Summary comparison
    print(f"\n{'='*80}")
    print(f"RANDOM SEARCH SUMMARY ({NUM_TRIALS} trials, {total_elapsed:.1f}s)")
    print(f"{'='*80}")
    print(f"{'Trial':<18} {'DX':>8} {'Speed':>8} {'Efficiency':>11} "
          f"{'PhaseLock':>10} {'Entropy':>8} {'RollDom':>8}")
    print("-" * 80)

    best_dx = None
    best_idx = -1
    for i, r in enumerate(results):
        o = r["analytics"]["outcome"]
        coord = r["analytics"]["coordination"]
        contact = r["analytics"]["contact"]
        ra = r["analytics"]["rotation_axis"]
        dx = o["dx"]
        print(f"  {r['name']:<16} {dx:+8.3f} {o['mean_speed']:8.4f} "
              f"{o['distance_per_work']:11.5f} {coord['phase_lock_score']:10.3f} "
              f"{contact['contact_entropy_bits']:8.2f} {ra['axis_dominance'][0]:8.2f}")
        if best_dx is None or abs(dx) > abs(best_dx):
            best_dx = dx
            best_idx = i

    print("-" * 80)
    best = results[best_idx]
    print(f"  Best mover: {best['name']} (DX={best_dx:+.3f})")
    print(f"  Weights: {best['weights']}")


if __name__ == "__main__":
    main()
