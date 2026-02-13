#!/usr/bin/env python3
"""
validation_experiments.py

Three validation experiments from the project roadmap:

1. Timestep Halving Test — Run the novelty champion at DT=1/480 (double resolution)
   with SIM_STEPS=8000 (same wall-clock duration). If DX changes significantly,
   the 60m result may be a simulation artifact.

2. Signal Path Tracing for w03=-1.31 — Instrument the NN to record pre-tanh
   activations and post-tanh motor values for motor neuron 3. Determines whether
   the out-of-[-1,1] weight saturates the tanh and how much extra drive it provides.

3. Random Walk at r=0.2 vs Novelty Seeker — Run 1000 random walk evaluations
   with the same step size (r=0.2) that the novelty seeker uses. Tests whether
   the novelty mechanism matters or if the step size alone is sufficient.

Outputs:
    artifacts/validation_experiments.json
    Console summary of all three experiments

Usage:
    python3 validation_experiments.py
"""

import json
import math
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

# ── Novelty Champion weights ─────────────────────────────────────────────────

NC_WEIGHTS = {
    "w03": -1.3083167156740476,
    "w04": -0.34279812804233867,
    "w13": 0.8331363773051514,
    "w14": -0.37582983217830773,
    "w23": -0.0369713954829298,
    "w24": 0.4375020967145814,
}

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]


# ── Shared simulation harness ────────────────────────────────────────────────

def write_brain_6syn(weights, path=None):
    """Write a 6-synapse brain.nndf file from a weight dict."""
    if path is None:
        path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor" jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor" jointName = "Torso_FrontLeg" />\n')
        # 3 sensors (0,1,2) x 2 motors (3,4) = 6 synapses
        for si in range(3):
            for mi in (3, 4):
                key = f"w{si}{mi}"
                val = weights[key]
                f.write(f'    <synapse sourceNeuronName = "{si}" '
                        f'targetNeuronName = "{mi}" weight = "{val}" />\n')
        f.write('</neuralNetwork>\n')


def simulate_dx(weights, dt=1/240, sim_steps=4000, record_nn=False):
    """Run a headless sim and return DX (and optionally NN trace data).

    Matches the walker_competition.py / analyze_novelty_champion.py loop exactly:
      1. Apply motor commands from current NN values (raw tanh output, no scaling/offset)
      2. Step physics
      3. Update NN (sensor reads + recompute motor neurons)

    Args:
        weights: dict of 6 synapse weights
        dt: physics timestep
        sim_steps: number of steps
        record_nn: if True, also record pre-tanh activations and post-tanh values

    Returns:
        dx (float) if record_nn is False
        (dx, nn_trace) if record_nn is True, where nn_trace is a dict of arrays
    """
    write_brain_6syn(weights)
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(dt)

    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robot_id)

    # Apply friction (same as walker_competition.py)
    mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
    for link in range(-1, p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, link, lateralFriction=mu, restitution=0.0)

    nn = NEURAL_NETWORK(str(PROJECT / "brain.nndf"))
    max_force = float(getattr(c, "MAX_FORCE", 150.0))

    start_x = p.getBasePositionAndOrientation(robot_id)[0][0]

    # NN trace arrays (only allocated if needed)
    if record_nn:
        pre_tanh_3 = np.zeros(sim_steps)
        pre_tanh_4 = np.zeros(sim_steps)
        post_tanh_3 = np.zeros(sim_steps)
        post_tanh_4 = np.zeros(sim_steps)
        sensor_vals = np.zeros((sim_steps, 3))

    for i in range(sim_steps):
        # 1. Act: apply motor commands from current NN values (raw, no scale/offset)
        for nName in nn.neurons:
            n_obj = nn.neurons[nName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robot_id, jn_bytes,
                                                n_obj.Get_Value(), max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robot_id, jn_bytes,
                                                p.POSITION_CONTROL,
                                                n_obj.Get_Value(), max_force)

        # 2. Step physics
        p.stepSimulation()

        # 3. Update NN (sensor reads + recompute motor neurons)
        # For record_nn mode, we intercept the Update to capture pre-tanh values
        if record_nn:
            # Manual NN update to capture pre-tanh activations
            for neuronName in nn.neurons:
                if nn.neurons[neuronName].Is_Sensor_Neuron():
                    nn.neurons[neuronName].Update_Sensor_Neuron()

            for si, sname in enumerate(["0", "1", "2"]):
                sensor_vals[i, si] = nn.neurons[sname].Get_Value()

            for targetName in nn.neurons:
                if not nn.neurons[targetName].Is_Sensor_Neuron():
                    total = 0.0
                    for sourceName in nn.neurons:
                        key = (sourceName, targetName)
                        if key in nn.synapses:
                            total += (nn.synapses[key].Get_Weight()
                                      * nn.neurons[sourceName].Get_Value())
                    val = math.tanh(total)
                    try:
                        nn.neurons[targetName].Set_Value(val)
                    except AttributeError:
                        nn.neurons[targetName].value = val

                    if targetName == "3":
                        pre_tanh_3[i] = total
                        post_tanh_3[i] = val
                    elif targetName == "4":
                        pre_tanh_4[i] = total
                        post_tanh_4[i] = val
        else:
            nn.Update()

    end_x = p.getBasePositionAndOrientation(robot_id)[0][0]
    p.disconnect()
    dx = end_x - start_x

    if record_nn:
        return dx, {
            "pre_tanh_3": pre_tanh_3,
            "pre_tanh_4": pre_tanh_4,
            "post_tanh_3": post_tanh_3,
            "post_tanh_4": post_tanh_4,
            "sensor_vals": sensor_vals,
        }
    return dx


def simulate_dx_only(weights):
    """Lightweight sim returning just DX (standard DT and steps)."""
    return simulate_dx(weights, dt=1/240, sim_steps=4000)


# ── Experiment 1: Timestep Halving ───────────────────────────────────────────

def experiment_1_timestep_halving():
    """Run NC at DT=1/240 and DT=1/480, compare DX."""
    print("=" * 80)
    print("EXPERIMENT 1: Timestep Halving Test")
    print("=" * 80)

    # Baseline: standard timestep
    t0 = time.time()
    dx_240 = simulate_dx(NC_WEIGHTS, dt=1/240, sim_steps=4000)
    t1 = time.time()
    print(f"  DT=1/240, steps=4000: DX = {dx_240:+.2f}m  ({t1-t0:.1f}s)")

    # Halved timestep: double the steps to keep same wall-clock sim duration
    t0 = time.time()
    dx_480 = simulate_dx(NC_WEIGHTS, dt=1/480, sim_steps=8000)
    t1 = time.time()
    print(f"  DT=1/480, steps=8000: DX = {dx_480:+.2f}m  ({t1-t0:.1f}s)")

    # Quarter timestep for extra validation
    t0 = time.time()
    dx_960 = simulate_dx(NC_WEIGHTS, dt=1/960, sim_steps=16000)
    t1 = time.time()
    print(f"  DT=1/960, steps=16000: DX = {dx_960:+.2f}m  ({t1-t0:.1f}s)")

    pct_change_480 = (dx_480 - dx_240) / abs(dx_240) * 100
    pct_change_960 = (dx_960 - dx_240) / abs(dx_240) * 100

    print(f"\n  Change 240→480: {pct_change_480:+.1f}%")
    print(f"  Change 240→960: {pct_change_960:+.1f}%")

    if abs(pct_change_480) < 5:
        verdict = "PASS — gait is timestep-stable (< 5% change at half DT)"
    elif abs(pct_change_480) < 15:
        verdict = "MARGINAL — moderate sensitivity to timestep (5-15% change)"
    else:
        verdict = "FAIL — gait is a simulation artifact (> 15% change at half DT)"

    print(f"  Verdict: {verdict}")

    return {
        "dx_dt240": float(dx_240),
        "dx_dt480": float(dx_480),
        "dx_dt960": float(dx_960),
        "pct_change_480": float(pct_change_480),
        "pct_change_960": float(pct_change_960),
        "verdict": verdict,
    }


# ── Experiment 2: Signal Path Tracing ────────────────────────────────────────

def experiment_2_signal_path():
    """Trace how w03=-1.31 flows through the NN. Record pre/post-tanh values."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Signal Path Tracing for w03=-1.31")
    print("=" * 80)

    dx, trace = simulate_dx(NC_WEIGHTS, record_nn=True)

    pre3 = trace["pre_tanh_3"]
    post3 = trace["post_tanh_3"]
    sensors = trace["sensor_vals"]

    # Sensor statistics
    s0_on = np.mean(sensors[:, 0] > 0.5)  # fraction of time torso touches ground
    s1_on = np.mean(sensors[:, 1] > 0.5)  # back leg
    s2_on = np.mean(sensors[:, 2] > 0.5)  # front leg

    print(f"\n  Sensor duty cycles: Torso={s0_on:.1%}, BackLeg={s1_on:.1%}, FrontLeg={s2_on:.1%}")

    # Pre-tanh activation statistics for motor 3
    print(f"\n  Motor 3 (BackLeg) pre-tanh activation:")
    print(f"    Range: [{pre3.min():.3f}, {pre3.max():.3f}]")
    print(f"    Mean:  {pre3.mean():.3f}")
    print(f"    Std:   {pre3.std():.3f}")

    # Post-tanh motor output
    print(f"  Motor 3 post-tanh output:")
    print(f"    Range: [{post3.min():.3f}, {post3.max():.3f}]")
    print(f"    Mean:  {post3.mean():.3f}")

    # Compare: what would happen if w03 were clamped to -1.0?
    # When s0=1, s1=0, s2=0: pre_tanh = w03*1 = -1.308 → tanh = -0.862
    # If w03=-1.0:             pre_tanh = -1.0         → tanh = -0.762
    # Difference: 0.100 in motor output
    tanh_at_actual = math.tanh(NC_WEIGHTS["w03"])
    tanh_at_clamped = math.tanh(-1.0)
    extra_drive = abs(tanh_at_actual) - abs(tanh_at_clamped)

    print(f"\n  Signal path analysis:")
    print(f"    w03 = {NC_WEIGHTS['w03']:.4f} (outside [-1, 1])")
    print(f"    tanh(w03) when s0=1 alone: {tanh_at_actual:.4f}")
    print(f"    tanh(-1.0) if clamped:     {tanh_at_clamped:.4f}")
    print(f"    Extra motor drive from out-of-range weight: {extra_drive:.4f}")
    print(f"    This translates to {extra_drive * c.NN_MOTOR_SCALE:.4f} rad extra joint angle")

    # Saturation analysis: how often is |pre_tanh| > 1.5 (deep in tanh saturation)?
    saturated_frac = np.mean(np.abs(pre3) > 1.5)
    print(f"\n  Saturation: {saturated_frac:.1%} of steps have |pre_tanh_3| > 1.5")

    # Now actually test: run with w03 clamped to -1.0
    clamped_weights = dict(NC_WEIGHTS)
    clamped_weights["w03"] = -1.0
    dx_clamped = simulate_dx(clamped_weights)
    pct_loss = (dx_clamped - dx) / abs(dx) * 100

    print(f"\n  Clamping test:")
    print(f"    DX with w03={NC_WEIGHTS['w03']:.4f}: {dx:+.2f}m")
    print(f"    DX with w03=-1.0 (clamped):     {dx_clamped:+.2f}m")
    print(f"    Change: {pct_loss:+.1f}%")

    if abs(pct_loss) < 5:
        verdict = "The out-of-range weight provides marginal benefit (< 5% DX change when clamped)"
    else:
        verdict = f"The out-of-range weight is significant ({pct_loss:+.1f}% DX change when clamped)"

    print(f"  Verdict: {verdict}")

    return {
        "dx_original": float(dx),
        "dx_clamped_w03": float(dx_clamped),
        "pct_change_clamped": float(pct_loss),
        "sensor_duty_torso": float(s0_on),
        "sensor_duty_back": float(s1_on),
        "sensor_duty_front": float(s2_on),
        "pre_tanh_3_range": [float(pre3.min()), float(pre3.max())],
        "pre_tanh_3_mean": float(pre3.mean()),
        "post_tanh_3_range": [float(post3.min()), float(post3.max())],
        "tanh_at_actual_w03": float(tanh_at_actual),
        "tanh_at_clamped_w03": float(tanh_at_clamped),
        "extra_drive_from_oor": float(extra_drive),
        "saturation_frac": float(saturated_frac),
        "verdict": verdict,
    }


# ── Experiment 3: Random Walk vs Novelty ─────────────────────────────────────

def experiment_3_random_walk_vs_novelty():
    """Run 1000 random walks with r=0.2 step size, compare to novelty seeker."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Random Walk at r=0.2 vs Novelty Seeker")
    print("=" * 80)

    rng = np.random.default_rng(42)
    N_EVALS = 1000
    STEP_SIZE = 0.2

    # Start from a random initial point (same as novelty seeker's starting condition)
    best_dx = 0.0
    best_weights = None
    all_dx = []

    # Strategy: random walk — each step perturbs the current best by r=0.2
    # in a random direction on the 6D unit sphere
    current_weights = {k: rng.uniform(-1, 1) for k in WEIGHT_NAMES}

    t0 = time.time()
    for trial in range(N_EVALS):
        # Random direction on 6D unit sphere
        direction = rng.standard_normal(6)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = np.zeros(6)
            direction[0] = 1.0
        else:
            direction /= norm

        # Perturb current position
        trial_weights = {}
        for j, k in enumerate(WEIGHT_NAMES):
            trial_weights[k] = current_weights[k] + STEP_SIZE * direction[j]

        dx = simulate_dx_only(trial_weights)
        all_dx.append(abs(dx))

        if abs(dx) > abs(best_dx):
            best_dx = dx
            best_weights = dict(trial_weights)
            current_weights = dict(trial_weights)  # walk toward the best

        if (trial + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{trial+1:4d}/{N_EVALS}] best |DX|={abs(best_dx):.2f}m  "
                  f"mean |DX|={np.mean(all_dx):.2f}m  ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    # Also run pure random sampling (no walk, just random points) for comparison
    print(f"\n  Running {N_EVALS} purely random samples for comparison...")
    pure_random_dx = []
    t1 = time.time()
    for trial in range(N_EVALS):
        rand_weights = {k: rng.uniform(-1, 1) for k in WEIGHT_NAMES}
        dx = simulate_dx_only(rand_weights)
        pure_random_dx.append(abs(dx))
        if (trial + 1) % 100 == 0:
            elapsed2 = time.time() - t1
            print(f"  [{trial+1:4d}/{N_EVALS}] best random |DX|={max(pure_random_dx):.2f}m  "
                  f"mean={np.mean(pure_random_dx):.2f}m  ({elapsed2:.1f}s)")

    novelty_dx = 60.19  # from walker_competition.py results

    print(f"\n  Results ({N_EVALS} evals each):")
    print(f"    Novelty Seeker:     best |DX| = {novelty_dx:.2f}m")
    print(f"    Random Walk r=0.2:  best |DX| = {abs(best_dx):.2f}m  "
          f"(mean {np.mean(all_dx):.2f}m)")
    print(f"    Pure Random:        best |DX| = {max(pure_random_dx):.2f}m  "
          f"(mean {np.mean(pure_random_dx):.2f}m)")

    walk_vs_novelty = abs(best_dx) / novelty_dx * 100
    random_vs_novelty = max(pure_random_dx) / novelty_dx * 100

    print(f"\n    Random walk achieves {walk_vs_novelty:.1f}% of novelty seeker")
    print(f"    Pure random achieves {random_vs_novelty:.1f}% of novelty seeker")

    if walk_vs_novelty > 90:
        verdict = "Step size alone is sufficient — novelty mechanism adds little"
    elif walk_vs_novelty > 60:
        verdict = "Step size helps, but novelty mechanism provides additional benefit"
    else:
        verdict = "Novelty mechanism is the key driver, not step size"

    print(f"    Verdict: {verdict}")

    return {
        "novelty_seeker_dx": novelty_dx,
        "random_walk_best_dx": float(best_dx),
        "random_walk_best_abs_dx": float(abs(best_dx)),
        "random_walk_mean_abs_dx": float(np.mean(all_dx)),
        "pure_random_best_abs_dx": float(max(pure_random_dx)),
        "pure_random_mean_abs_dx": float(np.mean(pure_random_dx)),
        "walk_vs_novelty_pct": float(walk_vs_novelty),
        "random_vs_novelty_pct": float(random_vs_novelty),
        "n_evals": N_EVALS,
        "step_size": STEP_SIZE,
        "verdict": verdict,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run all three validation experiments and save results."""
    t_start = time.time()

    results = {}
    results["exp1_timestep_halving"] = experiment_1_timestep_halving()
    results["exp2_signal_path"] = experiment_2_signal_path()
    results["exp3_random_walk"] = experiment_3_random_walk_vs_novelty()

    total_time = time.time() - t_start

    print("\n" + "=" * 80)
    print(f"All experiments complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 80)

    # Save results
    out_path = PROJECT / "artifacts" / "validation_experiments.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
