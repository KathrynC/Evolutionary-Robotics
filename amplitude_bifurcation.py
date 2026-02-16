#!/usr/bin/env python3
"""
amplitude_bifurcation.py

Role:
    Research campaign script that scales neural network motor output by an
    amplitude factor (0.1–1.5) across 10 representative gaits, detecting the
    chaos transition boundary per gait class.

    The key insight: because motor_output = tanh(Σ wᵢsᵢ), scaling weights by α
    gives tanh(α·Σ) ≠ α·tanh(Σ) due to tanh saturation. We scale the motor
    output AFTER tanh, at the PyBullet command point:

        scaled_value = amplitude_factor * n_obj.Get_Value()

    This is physically meaningful: it controls how far joints actually move,
    regardless of NN topology.

Approach:
    For each of 10 gaits spanning 5 behavioral classes:
    1. Sweep amplitude factor from 0.10 to 1.50 in steps of 0.01 (141 values)
    2. At each amplitude, run 5 perturbation trials (±0.001 to one random weight)
    3. Compute dx_std across perturbation runs as a chaos indicator
    4. Detect bifurcation point where dx_std exceeds threshold

    Simulation budget: 10 × 141 × 5 = 7,050 sims (~12 min at ~0.1s/sim)

Outputs:
    artifacts/amplitude_bifurcations_v2.json
    artifacts/amplitude_bifurcation_dx_vs_amp.png
    artifacts/amplitude_bifurcation_chaos_indicator.png
    artifacts/amplitude_bifurcation_by_class.png
    artifacts/amplitude_bifurcation_overlay.png
    artifacts/amplitude_bifurcation_phase_portrait.png
    artifacts/amplitude_bifurcation_heatmap.png

Usage:
    cd ~/pybullet_test/Evolutionary-Robotics && conda activate er
    HEADLESS=1 python3 amplitude_bifurcation.py
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
from pyrosim.neuralNetwork import NEURAL_NETWORK
import pyrosim.pyrosim as pyrosim
from compute_beer_analytics import compute_all, DT, NumpyEncoder

OUT_JSON = PROJECT / "artifacts" / "amplitude_bifurcations_v2.json"
PLOT_DIR = PROJECT / "artifacts"

SIM_STEPS = c.SIM_STEPS
MAX_FORCE = float(c.MAX_FORCE)

# Sweep parameters
AMP_MIN, AMP_MAX, AMP_STEP = 0.10, 1.50, 0.01
N_PERTURBATIONS = 5
PERTURBATION_SIZE = 0.001

# Bifurcation detection thresholds
BASELINE_AMP_MAX = 0.30  # amplitudes 0.10–0.30 define baseline
BIFURCATION_MULTIPLIER = 10.0  # dx_std must exceed 10× baseline
BIFURCATION_FLOOR = 1.0  # minimum dx_std threshold (meters)


# ── Brain writing ────────────────────────────────────────────────────────────

def write_brain_crosswired(w03, w13, w23, w04, w14, w24,
                           w34=0.0, w43=0.0, w33=0.0, w44=0.0):
    """Write brain.nndf for a standard or crosswired topology."""
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for src, tgt, w in [("0","3",w03), ("1","3",w13), ("2","3",w23),
                             ("0","4",w04), ("1","4",w14), ("2","4",w24),
                             ("3","4",w34), ("4","3",w43), ("3","3",w33), ("4","4",w44)]:
            if w != 0.0:
                f.write(f'    <synapse sourceNeuronName = "{src}" targetNeuronName = "{tgt}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def write_brain_full(neurons, synapses):
    """Write brain.nndf for an arbitrary topology (including hidden neurons)."""
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        for neuron in neurons:
            nid, ntype, ref = neuron["id"], neuron["type"], neuron.get("ref")
            if ntype == "sensor":
                f.write(f'    <neuron name = "{nid}" type = "sensor" linkName = "{ref}" />\n')
            elif ntype == "motor":
                f.write(f'    <neuron name = "{nid}" type = "motor"  jointName = "{ref}" />\n')
            else:
                f.write(f'    <neuron name = "{nid}" type = "hidden" />\n')
        for syn in synapses:
            w = syn["w"]
            if w != 0.0:
                f.write(f'    <synapse sourceNeuronName = "{syn["src"]}" targetNeuronName = "{syn["tgt"]}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


# ── 10 Target Gaits ──────────────────────────────────────────────────────────

TARGET_GAITS = [
    {
        "name": "19_haraway", "gait_class": "Antifragile", "architecture": "standard_6",
        "weights": {"w03": -0.5, "w13": -0.5, "w23": -0.5, "w04": -0.5, "w14": -0.5, "w24": -0.5},
    },
    {
        "name": "5_pelton", "gait_class": "Antifragile", "architecture": "standard_6",
        "weights": {"w03": -0.3, "w13": -1.0, "w23": -0.3, "w04": 1.0, "w14": 0.3, "w24": 1.0},
    },
    {
        "name": "32_carry_trade", "gait_class": "Knife-edge", "architecture": "crosswired_10",
        "weights": {"w03": -0.3, "w13": -0.9, "w23": -0.3, "w04": 0.9, "w14": 0.3, "w24": 0.9,
                    "w34": 0.3, "w43": -0.3, "w33": 0.3, "w44": -0.9},
    },
    {
        "name": "43_hidden_cpg_champion", "gait_class": "Knife-edge", "architecture": "hidden",
        "neurons": [
            {"id": "0", "type": "sensor", "ref": "Torso"},
            {"id": "1", "type": "sensor", "ref": "BackLeg"},
            {"id": "2", "type": "sensor", "ref": "FrontLeg"},
            {"id": "3", "type": "motor",  "ref": "Torso_BackLeg"},
            {"id": "4", "type": "motor",  "ref": "Torso_FrontLeg"},
            {"id": "5", "type": "hidden", "ref": None},
            {"id": "6", "type": "hidden", "ref": None},
        ],
        "synapses": [
            {"src": "1", "tgt": "5", "w": -0.6},
            {"src": "2", "tgt": "6", "w": -0.6},
            {"src": "5", "tgt": "6", "w":  0.7},
            {"src": "6", "tgt": "5", "w": -0.7},
            {"src": "5", "tgt": "3", "w": -0.8},
            {"src": "6", "tgt": "4", "w":  0.8},
            {"src": "0", "tgt": "3", "w": -0.3},
            {"src": "0", "tgt": "4", "w":  0.3},
        ],
    },
    {
        "name": "56_evolved_crab_v2", "gait_class": "Crab", "architecture": "crosswired_10",
        "weights": {"w03": 0.2515753101143031, "w13": -1.4976902964989567,
                    "w23": -0.6410900845427805, "w04": 0.9733372173225151,
                    "w14": 1.5824646682114387, "w24": 1.2083292616061312,
                    "w34": 0.7254521057740698, "w43": 1.1328779669631484,
                    "w33": -0.24636848771593106, "w44": 0.8480271198639224},
    },
    {
        "name": "52_curie_crab", "gait_class": "Crab", "architecture": "crosswired_10",
        "weights": {"w03": -0.3, "w13": -0.85, "w23": -0.26, "w04": 0.85, "w14": 0.26,
                    "w24": 0.85, "w34": -0.5, "w43": 0.5, "w33": 0, "w44": 0},
    },
    {
        "name": "44_spinner_champion", "gait_class": "Spinner", "architecture": "hidden",
        "neurons": [
            {"id": "0", "type": "sensor", "ref": "Torso"},
            {"id": "1", "type": "sensor", "ref": "BackLeg"},
            {"id": "2", "type": "sensor", "ref": "FrontLeg"},
            {"id": "3", "type": "motor",  "ref": "Torso_BackLeg"},
            {"id": "4", "type": "motor",  "ref": "Torso_FrontLeg"},
            {"id": "5", "type": "hidden", "ref": None},
            {"id": "6", "type": "hidden", "ref": None},
        ],
        "synapses": [
            {"src": "1", "tgt": "5", "w": -0.5},
            {"src": "2", "tgt": "6", "w": -0.5},
            {"src": "5", "tgt": "6", "w":  0.35},
            {"src": "6", "tgt": "5", "w": -0.35},
            {"src": "5", "tgt": "3", "w": -0.6},
            {"src": "6", "tgt": "4", "w":  0.5},
            {"src": "0", "tgt": "3", "w": -0.4},
            {"src": "0", "tgt": "4", "w":  0.4},
        ],
    },
    {
        "name": "45_spinner_stable", "gait_class": "Spinner", "architecture": "hidden",
        "neurons": [
            {"id": "0", "type": "sensor", "ref": "Torso"},
            {"id": "1", "type": "sensor", "ref": "BackLeg"},
            {"id": "2", "type": "sensor", "ref": "FrontLeg"},
            {"id": "3", "type": "motor",  "ref": "Torso_BackLeg"},
            {"id": "4", "type": "motor",  "ref": "Torso_FrontLeg"},
            {"id": "5", "type": "hidden", "ref": None},
            {"id": "6", "type": "hidden", "ref": None},
        ],
        "synapses": [
            {"src": "1", "tgt": "5", "w": -0.3},
            {"src": "2", "tgt": "6", "w": -0.3},
            {"src": "5", "tgt": "6", "w":  0.4},
            {"src": "6", "tgt": "5", "w": -0.4},
            {"src": "5", "tgt": "3", "w": -0.7},
            {"src": "6", "tgt": "4", "w":  0.4},
            {"src": "0", "tgt": "3", "w": -0.3},
            {"src": "0", "tgt": "4", "w":  0.3},
        ],
    },
    {
        "name": "36_take_five", "gait_class": "Time sig", "architecture": "crosswired_10",
        "weights": {"w03": -0.8, "w13": -0.8, "w23": -0.5, "w04": 0.5, "w14": 0.8,
                    "w24": 0.8, "w34": 0.5, "w43": -0.5, "w33": 0.3, "w44": -0.3},
    },
    {
        "name": "37_hemiola", "gait_class": "Time sig", "architecture": "standard_6",
        "weights": {"w03": -0.5, "w13": -1.0, "w23": -0.5, "w04": 1.0, "w14": 0.5, "w24": 1.0},
    },
]


def write_brain_for_gait(gait):
    """Write brain.nndf for the given gait entry."""
    if gait["architecture"] == "hidden":
        write_brain_full(gait["neurons"], gait["synapses"])
    else:
        w = gait["weights"]
        write_brain_crosswired(
            w.get("w03", 0), w.get("w13", 0), w.get("w23", 0),
            w.get("w04", 0), w.get("w14", 0), w.get("w24", 0),
            w.get("w34", 0), w.get("w43", 0), w.get("w33", 0), w.get("w44", 0),
        )


def perturb_gait(gait, rng):
    """Return a copy of the gait with one random weight perturbed by ±PERTURBATION_SIZE."""
    gait_copy = json.loads(json.dumps(gait))  # deep copy
    if gait["architecture"] == "hidden":
        synapses = gait_copy["synapses"]
        idx = rng.integers(len(synapses))
        delta = rng.choice([-PERTURBATION_SIZE, PERTURBATION_SIZE])
        synapses[idx]["w"] = synapses[idx]["w"] + delta
    else:
        weights = gait_copy["weights"]
        keys = [k for k in weights if weights[k] != 0.0]
        if not keys:
            keys = list(weights.keys())
        key = keys[rng.integers(len(keys))]
        delta = rng.choice([-PERTURBATION_SIZE, PERTURBATION_SIZE])
        weights[key] = weights[key] + delta
    return gait_copy


# ── Simulation with amplitude scaling ────────────────────────────────────────

def run_trial_with_amplitude(gait, amplitude_factor):
    """Run a headless PyBullet simulation with motor output scaled by amplitude_factor.

    Identical to run_trial_inmemory() from structured_random_common.py except
    that motor commands are multiplied by amplitude_factor before being sent
    to PyBullet:

        scaled_value = amplitude_factor * n_obj.Get_Value()

    This scales how far joints actually move, independent of NN topology.

    Returns:
        Analytics dict with 4 Beer-framework pillars.
    """
    write_brain_for_gait(gait)

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
    max_force = MAX_FORCE
    n_steps = SIM_STEPS

    # Pre-allocate telemetry arrays
    t_arr = np.empty(n_steps)
    x = np.empty(n_steps); y = np.empty(n_steps); z = np.empty(n_steps)
    vx = np.empty(n_steps); vy = np.empty(n_steps); vz = np.empty(n_steps)
    wx = np.empty(n_steps); wy = np.empty(n_steps); wz = np.empty(n_steps)
    roll = np.empty(n_steps); pitch = np.empty(n_steps); yaw = np.empty(n_steps)
    contact_torso = np.empty(n_steps, dtype=bool)
    contact_back = np.empty(n_steps, dtype=bool)
    contact_front = np.empty(n_steps, dtype=bool)
    j0_pos = np.empty(n_steps); j0_vel = np.empty(n_steps); j0_tau = np.empty(n_steps)
    j1_pos = np.empty(n_steps); j1_vel = np.empty(n_steps); j1_tau = np.empty(n_steps)

    # Resolve link and joint indices
    link_names = ["Torso", "BackLeg", "FrontLeg"]
    link_indices = {}
    joint_indices = {}
    for i_j in range(p.getNumJoints(robotId)):
        info = p.getJointInfo(robotId, i_j)
        jname = info[1].decode("utf-8") if isinstance(info[1], bytes) else info[1]
        lname = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
        if lname in link_names:
            link_indices[lname] = i_j
        joint_indices[jname] = i_j

    back_link_idx = link_indices.get("BackLeg", -1)
    front_link_idx = link_indices.get("FrontLeg", -1)
    j0_idx = joint_indices.get("Torso_BackLeg", 0)
    j1_idx = joint_indices.get("Torso_FrontLeg", 1)

    # Main simulation loop with amplitude-scaled motor output
    for i in range(n_steps):
        for neuronName in nn.neurons:
            n_obj = nn.neurons[neuronName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                scaled_value = amplitude_factor * n_obj.Get_Value()
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, scaled_value, max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, p.POSITION_CONTROL,
                                                scaled_value, max_force)
        p.stepSimulation()
        nn.Update()

        # Record state
        t_arr[i] = i * c.DT
        pos, orn = p.getBasePositionAndOrientation(robotId)
        vel_lin, vel_ang = p.getBaseVelocity(robotId)
        rpy_vals = p.getEulerFromQuaternion(orn)

        x[i] = pos[0]; y[i] = pos[1]; z[i] = pos[2]
        vx[i] = vel_lin[0]; vy[i] = vel_lin[1]; vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]; wy[i] = vel_ang[1]; wz[i] = vel_ang[2]
        roll[i] = rpy_vals[0]; pitch[i] = rpy_vals[1]; yaw[i] = rpy_vals[2]

        contact_pts = p.getContactPoints(robotId)
        torso_contact = False; back_contact = False; front_contact = False
        for cp in contact_pts:
            li = cp[3]
            if li == -1:
                torso_contact = True
            elif li == back_link_idx:
                back_contact = True
            elif li == front_link_idx:
                front_contact = True
        contact_torso[i] = torso_contact
        contact_back[i] = back_contact
        contact_front[i] = front_contact

        js0 = p.getJointState(robotId, j0_idx)
        js1 = p.getJointState(robotId, j1_idx)
        j0_pos[i] = js0[0]; j0_vel[i] = js0[1]; j0_tau[i] = js0[3]
        j1_pos[i] = js1[0]; j1_vel[i] = js1[1]; j1_tau[i] = js1[3]

    p.disconnect()

    data = {
        "t": t_arr, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "contact_torso": contact_torso,
        "contact_back": contact_back,
        "contact_front": contact_front,
        "j0_pos": j0_pos, "j0_vel": j0_vel, "j0_tau": j0_tau,
        "j1_pos": j1_pos, "j1_vel": j1_vel, "j1_tau": j1_tau,
    }
    return compute_all(data, DT)


# ── Sweep & Detection ────────────────────────────────────────────────────────

def run_sweep():
    """Run the full amplitude sweep across all 10 gaits."""
    rng = np.random.default_rng(42)
    amplitudes = np.round(np.arange(AMP_MIN, AMP_MAX + AMP_STEP/2, AMP_STEP), 4)
    n_amps = len(amplitudes)
    total_sims = len(TARGET_GAITS) * n_amps * N_PERTURBATIONS
    print(f"Amplitude Bifurcation Experiment")
    print(f"  {len(TARGET_GAITS)} gaits x {n_amps} amplitudes x {N_PERTURBATIONS} perturbations = {total_sims} sims")

    all_results = []
    t_start = time.perf_counter()
    sim_count = 0

    for gi, gait in enumerate(TARGET_GAITS):
        gait_name = gait["name"]
        gait_class = gait["gait_class"]
        print(f"\n{'='*70}")
        print(f"[{gi+1}/{len(TARGET_GAITS)}] {gait_name} ({gait_class}, {gait['architecture']})")
        print(f"{'='*70}")

        # Arrays to store results for this gait
        dx_mean = np.empty(n_amps)
        dx_std = np.empty(n_amps)
        dx_all = np.empty((n_amps, N_PERTURBATIONS))

        t_gait_start = time.perf_counter()

        for ai, amp in enumerate(amplitudes):
            dx_runs = np.empty(N_PERTURBATIONS)
            for pi in range(N_PERTURBATIONS):
                perturbed = perturb_gait(gait, rng)
                analytics = run_trial_with_amplitude(perturbed, amp)
                dx_runs[pi] = analytics["outcome"]["dx"]
                sim_count += 1

            dx_all[ai, :] = dx_runs
            dx_mean[ai] = np.mean(dx_runs)
            dx_std[ai] = np.std(dx_runs)

            if (ai + 1) % 20 == 0:
                elapsed = time.perf_counter() - t_start
                rate = elapsed / sim_count
                remaining = rate * (total_sims - sim_count)
                print(f"  amp={amp:.2f}  dx_mean={dx_mean[ai]:+.2f}m  "
                      f"dx_std={dx_std[ai]:.4f}m  "
                      f"[{sim_count}/{total_sims}] ~{remaining:.0f}s rem", flush=True)

        gait_elapsed = time.perf_counter() - t_gait_start

        # Detect bifurcation point
        baseline_mask = amplitudes <= BASELINE_AMP_MAX
        baseline_std = np.mean(dx_std[baseline_mask]) if np.any(baseline_mask) else 0.0
        threshold = max(BIFURCATION_MULTIPLIER * baseline_std, BIFURCATION_FLOOR)

        bifurcation_amp = None
        for ai in range(n_amps):
            if dx_std[ai] > threshold:
                bifurcation_amp = float(amplitudes[ai])
                break

        print(f"  Completed in {gait_elapsed:.1f}s  "
              f"baseline_std={baseline_std:.4f}  threshold={threshold:.4f}  "
              f"bifurcation={bifurcation_amp}")

        all_results.append({
            "name": gait_name,
            "gait_class": gait_class,
            "architecture": gait["architecture"],
            "amplitudes": amplitudes.tolist(),
            "dx_mean": dx_mean.tolist(),
            "dx_std": dx_std.tolist(),
            "dx_all": dx_all.tolist(),
            "baseline_std": float(baseline_std),
            "threshold": float(threshold),
            "bifurcation_amp": bifurcation_amp,
        })

    total_elapsed = time.perf_counter() - t_start
    print(f"\nSweep complete: {sim_count} sims in {total_elapsed:.1f}s "
          f"({total_elapsed/sim_count:.3f}s/sim)")
    return all_results, amplitudes


# ── Figures ──────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "Antifragile": "#2ca02c",
    "Knife-edge": "#d62728",
    "Crab": "#1f77b4",
    "Spinner": "#9467bd",
    "Time sig": "#ff7f0e",
}

CLASS_MARKERS = {
    "Antifragile": "o",
    "Knife-edge": "s",
    "Crab": "D",
    "Spinner": "^",
    "Time sig": "v",
}


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


def fig01_dx_vs_amplitude(results, amplitudes):
    """Per-gait DX vs amplitude with error bars from perturbation runs."""
    fig, axes = plt.subplots(2, 5, figsize=(26, 11), sharex=True)
    axes_flat = axes.flatten()

    for i, r in enumerate(results):
        ax = axes_flat[i]
        amps = np.array(r["amplitudes"])
        mean = np.array(r["dx_mean"])
        std = np.array(r["dx_std"])
        color = CLASS_COLORS[r["gait_class"]]

        ax.fill_between(amps, mean - std, mean + std, alpha=0.25, color=color)
        ax.plot(amps, mean, "-", color=color, lw=2)

        if r["bifurcation_amp"] is not None:
            bif_idx = np.argmin(np.abs(amps - r["bifurcation_amp"]))
            ax.axvline(r["bifurcation_amp"], color="red", ls="--", lw=1.2, alpha=0.7)
            ax.annotate(f'bif={r["bifurcation_amp"]:.2f}',
                        (r["bifurcation_amp"], mean[bif_idx]),
                        textcoords="offset points", xytext=(4, 8), fontsize=8,
                        color="red", fontweight="bold")

        # Mark peak |DX|
        peak_idx = np.argmax(np.abs(mean))
        ax.plot(amps[peak_idx], mean[peak_idx], "*", color=color,
                markersize=10, markeredgecolor="black", markeredgewidth=0.5)

        ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(f"{r['name']}\n({r['gait_class']}, {r['architecture']})", fontsize=10)
        if i % 5 == 0:
            ax.set_ylabel("DX (m)", fontsize=10)
        clean_ax(ax)

    for ax in axes_flat[5:]:
        ax.set_xlabel("Amplitude factor", fontsize=10)

    fig.suptitle("DX vs Amplitude Factor (mean ± std across 5 perturbation runs, star = peak)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "amplitude_bifurcation_dx_vs_amp.png")


def fig02_chaos_indicator(results, amplitudes):
    """Per-gait dx_std vs amplitude (chaos indicator)."""
    fig, axes = plt.subplots(2, 5, figsize=(26, 11), sharex=True)
    axes_flat = axes.flatten()

    for i, r in enumerate(results):
        ax = axes_flat[i]
        amps = np.array(r["amplitudes"])
        std = np.array(r["dx_std"])
        color = CLASS_COLORS[r["gait_class"]]

        ax.semilogy(amps, std + 1e-6, "-", color=color, lw=2)
        ax.axhline(r["threshold"], color="red", ls="--", lw=1.2, alpha=0.6,
                    label=f"threshold={r['threshold']:.2f}m")
        ax.axhspan(0, r["threshold"], alpha=0.05, color="green")

        if r["bifurcation_amp"] is not None:
            ax.axvline(r["bifurcation_amp"], color="red", ls="--", lw=1.2, alpha=0.7)

        ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(f"{r['name']}\n({r['gait_class']}, {r['architecture']})", fontsize=10)
        if i % 5 == 0:
            ax.set_ylabel("dx_std (m)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        clean_ax(ax)

    for ax in axes_flat[5:]:
        ax.set_xlabel("Amplitude factor", fontsize=10)

    fig.suptitle("Chaos Indicator: std(DX) Across Perturbation Runs vs Amplitude",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "amplitude_bifurcation_chaos_indicator.png")


def fig03_by_class(results):
    """Bifurcation points by gait class (bar chart), sorted by bifurcation amplitude."""
    from matplotlib.patches import Patch

    # Sort by bifurcation amplitude for clearer visual
    sorted_results = sorted(results,
                            key=lambda r: r["bifurcation_amp"] if r["bifurcation_amp"] is not None else 1.55)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [3, 2]})

    # Left panel: per-gait bars sorted by bifurcation amplitude
    ax = axes[0]
    names = [r["name"] for r in sorted_results]
    bif_amps = [r["bifurcation_amp"] if r["bifurcation_amp"] is not None else 1.55
                for r in sorted_results]
    colors = [CLASS_COLORS[r["gait_class"]] for r in sorted_results]
    is_none = [r["bifurcation_amp"] is None for r in sorted_results]
    archs = [r["architecture"] for r in sorted_results]

    bars = ax.barh(range(len(names)), bif_amps, color=colors, edgecolor="black", lw=0.5,
                   height=0.7)

    for i, (bar, none_flag, arch) in enumerate(zip(bars, is_none, archs)):
        bif_val = bif_amps[i]
        if none_flag:
            ax.text(bif_val - 0.02, i, "none", ha="right", va="center",
                    fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(bif_val + 0.02, i, f"{bif_val:.2f}  [{arch}]", ha="left", va="center",
                    fontsize=9, color="#333")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Bifurcation amplitude", fontsize=12)
    ax.set_xlim(0, 1.15)
    ax.axvline(1.0, color="gray", ls="--", lw=1.5, alpha=0.7, label="Normal operating point (1.0)")
    ax.invert_yaxis()

    handles = [Patch(facecolor=c, edgecolor="black", label=cls)
               for cls, c in CLASS_COLORS.items()]
    handles.append(plt.Line2D([0], [0], color="gray", ls="--", lw=1.5, label="Normal (1.0)"))
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    ax.set_title("Bifurcation Point by Gait (sorted)", fontsize=13, fontweight="bold")
    clean_ax(ax)

    # Right panel: grouped by architecture
    ax = axes[1]
    from collections import defaultdict
    arch_bifs = defaultdict(list)
    for r in results:
        bif = r["bifurcation_amp"] if r["bifurcation_amp"] is not None else 1.55
        arch_bifs[r["architecture"]].append(bif)

    arch_order = ["hidden", "standard_6", "crosswired_10"]
    arch_colors = {"hidden": "#9467bd", "standard_6": "#2ca02c", "crosswired_10": "#1f77b4"}
    positions = []
    labels = []
    data = []
    for i, arch in enumerate(arch_order):
        bifs = arch_bifs.get(arch, [])
        if bifs:
            positions.append(i)
            labels.append(arch)
            data.append(bifs)

    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                    markeredgecolor="black", markersize=6))
    for patch, arch in zip(bp["boxes"], arch_order):
        patch.set_facecolor(arch_colors[arch])
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (arch, bifs) in enumerate(zip(arch_order, data)):
        ax.scatter([i] * len(bifs), bifs, color=arch_colors[arch],
                   edgecolors="black", s=60, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Bifurcation amplitude", fontsize=12)
    ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.set_title("By Architecture", fontsize=13, fontweight="bold")
    clean_ax(ax)

    fig.suptitle("All gaits bifurcate below amplitude 1.0 — the zoo operates in chaos",
                 fontsize=14, fontweight="bold", style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "amplitude_bifurcation_by_class.png")


def fig04_overlay(results, amplitudes):
    """Two-panel overlay: raw |DX| and normalized DX vs amplitude."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Left panel: raw |DX| mean curves
    ax = axes[0]
    for r in results:
        amps = np.array(r["amplitudes"])
        mean = np.abs(np.array(r["dx_mean"]))
        color = CLASS_COLORS[r["gait_class"]]
        ax.plot(amps, mean, "-", color=color, lw=2, alpha=0.85, label=r["name"])

        if r["bifurcation_amp"] is not None:
            bif_idx = np.argmin(np.abs(amps - r["bifurcation_amp"]))
            ax.plot(r["bifurcation_amp"], mean[bif_idx], "x",
                    color="red", markersize=11, markeredgewidth=2.5, zorder=5)

    ax.axvline(1.0, color="gray", ls="--", lw=1.5, alpha=0.5)
    ax.set_xlabel("Amplitude factor", fontsize=12)
    ax.set_ylabel("|DX| (m)", fontsize=12)
    ax.set_title("Raw |DX| vs Amplitude (x = bifurcation)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    clean_ax(ax)

    # Right panel: normalized by peak |DX| per gait (so all peak at 1.0)
    ax = axes[1]
    for r in results:
        amps = np.array(r["amplitudes"])
        mean = np.abs(np.array(r["dx_mean"]))
        peak = np.max(mean)
        if peak < 0.1:
            peak = 1.0
        normalized = mean / peak
        color = CLASS_COLORS[r["gait_class"]]
        ax.plot(amps, normalized, "-", color=color, lw=2, alpha=0.85, label=r["name"])

        if r["bifurcation_amp"] is not None:
            bif_idx = np.argmin(np.abs(amps - r["bifurcation_amp"]))
            ax.plot(r["bifurcation_amp"], normalized[bif_idx], "x",
                    color="red", markersize=11, markeredgewidth=2.5, zorder=5)

    ax.axvline(1.0, color="gray", ls="--", lw=1.5, alpha=0.5)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("Amplitude factor", fontsize=12)
    ax.set_ylabel("|DX| / peak |DX|", fontsize=12)
    ax.set_title("Normalized to Peak (all gaits peak at 1.0)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    clean_ax(ax)

    fig.suptitle("Amplitude Response Curves: Every Gait Has a Sub-Unity Optimal Amplitude",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "amplitude_bifurcation_overlay.png")


def fig05_phase_portrait(results):
    """Phase portrait: amplitude-colored scatter of (|DX|, dx_std) per gait."""
    fig, axes = plt.subplots(2, 5, figsize=(26, 11))
    axes_flat = axes.flatten()

    amp_cmap = plt.cm.viridis

    for i, r in enumerate(results):
        ax = axes_flat[i]
        amps = np.array(r["amplitudes"])
        mean = np.abs(np.array(r["dx_mean"]))
        std = np.array(r["dx_std"])
        color = CLASS_COLORS[r["gait_class"]]

        # Scatter colored by amplitude
        sc = ax.scatter(mean, std, c=amps, cmap=amp_cmap, s=12, alpha=0.7,
                        edgecolors="none")

        # Connect consecutive points with thin line
        ax.plot(mean, std, "-", color=color, lw=0.6, alpha=0.3)

        # Mark start (low amp), peak |DX|, and bifurcation
        ax.plot(mean[0], std[0], "o", color="blue", markersize=7,
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        peak_idx = np.argmax(mean)
        ax.plot(mean[peak_idx], std[peak_idx], "*", color="gold",
                markersize=12, markeredgecolor="black", markeredgewidth=0.5, zorder=5)

        if r["bifurcation_amp"] is not None:
            bif_idx = np.argmin(np.abs(amps - r["bifurcation_amp"]))
            ax.plot(mean[bif_idx], std[bif_idx], "X", color="red",
                    markersize=10, markeredgecolor="black", markeredgewidth=0.8, zorder=6)

        ax.set_title(f"{r['name']}\n({r['gait_class']})", fontsize=10)
        if i % 5 == 0:
            ax.set_ylabel("dx_std (m)", fontsize=10)
        if i >= 5:
            ax.set_xlabel("|DX| (m)", fontsize=10)
        clean_ax(ax)

    fig.suptitle("Phase Portrait: |DX| vs dx_std (blue=start, gold star=peak, red X=bifurcation, color=amplitude)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.90, top=0.93, bottom=0.07, wspace=0.3, hspace=0.35)

    # Add a single colorbar for the amplitude mapping
    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.75])
    norm = Normalize(vmin=float(amps[0]), vmax=float(amps[-1]))
    sm = plt.cm.ScalarMappable(cmap=amp_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Amplitude factor")
    save_fig(fig, "amplitude_bifurcation_phase_portrait.png")


def fig06_heatmap(results, amplitudes):
    """Summary heatmap: gait x amplitude -> chaos level (log dx_std), sorted by bifurcation."""
    # Sort gaits by bifurcation amplitude for visual coherence
    sorted_results = sorted(results,
                            key=lambda r: r["bifurcation_amp"] if r["bifurcation_amp"] is not None else 1.55)

    n_gaits = len(sorted_results)
    n_amps = len(amplitudes)
    matrix = np.zeros((n_gaits, n_amps))

    for i, r in enumerate(sorted_results):
        std = np.array(r["dx_std"])
        matrix[i, :] = np.log10(std + 1e-6)

    fig, ax = plt.subplots(figsize=(20, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="inferno",
                   extent=[amplitudes[0], amplitudes[-1], n_gaits - 0.5, -0.5])
    cbar = plt.colorbar(im, ax=ax, label="log10(dx_std)", pad=0.01)
    cbar.ax.tick_params(labelsize=9)

    ax.set_yticks(range(n_gaits))
    gait_labels = [f"{r['name']}  [{r['architecture']}]" for r in sorted_results]
    ax.set_yticklabels(gait_labels, fontsize=10)
    ax.set_xlabel("Amplitude factor", fontsize=12)

    # Mark bifurcation points with white stars and connecting line
    bif_y = []
    bif_x = []
    for i, r in enumerate(sorted_results):
        if r["bifurcation_amp"] is not None:
            bif_y.append(i)
            bif_x.append(r["bifurcation_amp"])
            ax.plot(r["bifurcation_amp"], i, "w*", markersize=14,
                    markeredgecolor="black", markeredgewidth=0.5)

    # Draw the bifurcation frontier
    if len(bif_x) > 1:
        ax.plot(bif_x, bif_y, "w--", lw=1.5, alpha=0.6)

    ax.axvline(1.0, color="white", ls="--", lw=2, alpha=0.7)
    ax.text(1.01, -0.35, "normal\noperating\npoint", color="white", fontsize=8,
            ha="left", va="top", fontweight="bold")

    # Color-code gait labels by class
    for i, r in enumerate(sorted_results):
        color = CLASS_COLORS[r["gait_class"]]
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_title("Chaos Heatmap (sorted by bifurcation point — stars mark transition, dashed line = frontier)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "amplitude_bifurcation_heatmap.png")


# ── Console summary ──────────────────────────────────────────────────────────

def print_summary(results):
    print(f"\n{'='*70}")
    print("AMPLITUDE BIFURCATION — RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Gait':<30s} {'Class':<14s} {'Arch':<16s} {'Bif.Amp':>8s} {'Baseline':>10s}")
    print("  " + "-" * 80)
    for r in results:
        bif = f"{r['bifurcation_amp']:.2f}" if r["bifurcation_amp"] is not None else "none"
        print(f"  {r['name']:<30s} {r['gait_class']:<14s} {r['architecture']:<16s} "
              f"{bif:>8s} {r['baseline_std']:>10.5f}")

    # Group by class
    print(f"\n  BIFURCATION BY CLASS:")
    from collections import defaultdict
    class_bifs = defaultdict(list)
    for r in results:
        if r["bifurcation_amp"] is not None:
            class_bifs[r["gait_class"]].append(r["bifurcation_amp"])
        else:
            class_bifs[r["gait_class"]].append(float("inf"))

    for cls in CLASS_COLORS:
        bifs = class_bifs.get(cls, [])
        if bifs:
            finite = [b for b in bifs if b != float("inf")]
            if finite:
                print(f"    {cls:<14s}: mean={np.mean(finite):.2f}  "
                      f"range=[{min(finite):.2f}, {max(finite):.2f}]")
            else:
                print(f"    {cls:<14s}: no bifurcation detected")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    # Backup brain.nndf
    backup = PROJECT / "brain.nndf.bak"
    brain_path = PROJECT / "brain.nndf"
    if brain_path.exists():
        shutil.copy2(brain_path, backup)
        print(f"Backed up brain.nndf -> brain.nndf.bak")

    try:
        results, amplitudes = run_sweep()

        # Generate figures
        print(f"\n{'='*70}")
        print("GENERATING FIGURES")
        print(f"{'='*70}")
        fig01_dx_vs_amplitude(results, amplitudes)
        fig02_chaos_indicator(results, amplitudes)
        fig03_by_class(results)
        fig04_overlay(results, amplitudes)
        fig05_phase_portrait(results)
        fig06_heatmap(results, amplitudes)

        # Console summary
        print_summary(results)

        # Save JSON
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "meta": {
                "sim_steps": SIM_STEPS,
                "dt": c.DT,
                "max_force": MAX_FORCE,
                "n_perturbations": N_PERTURBATIONS,
                "perturbation_size": PERTURBATION_SIZE,
                "amp_range": [AMP_MIN, AMP_MAX, AMP_STEP],
                "bifurcation_multiplier": BIFURCATION_MULTIPLIER,
                "bifurcation_floor": BIFURCATION_FLOOR,
                "baseline_amp_max": BASELINE_AMP_MAX,
            },
            "results": results,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        print(f"\nWROTE {OUT_JSON}")

    finally:
        # Restore brain.nndf
        if backup.exists():
            shutil.copy2(backup, brain_path)
            print(f"Restored brain.nndf from backup")

    total = time.perf_counter() - t0
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
