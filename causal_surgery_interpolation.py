#!/usr/bin/env python3
"""
causal_surgery_interpolation.py

Role:
    Two-part analysis of the Synapse Gait Zoo champions: static ablation
    (causal surgery) and linear interpolation through 6D weight space.
    Characterizes synapse importance and landscape smoothness.

Part 1 -- Causal Surgery (Ablation Study):
    For each of 3 champions (Novelty Champion, CPG Champion, Trial 3),
    systematically ablate each synapse (zero / half / negate) and measure
    the effect on locomotion metrics.
    63 simulations total (3 champions x 6-8 synapses x 3 modes + baselines).

Part 2 -- Gait Interpolation:
    Linearly interpolate between pairs of 6-synapse gaits and track how
    metrics change across the landscape. Reveals cliffs, smooth gradients,
    and intermediate super-gaits.
    153 simulations total (51 points x 3 pairs).

Notes:
    - Unlike causal_surgery.py (which modifies weights mid-simulation), this
      script performs static ablation: each simulation starts with the modified
      weights and runs to completion.
    - Supports both standard 6-synapse and CPG 7-neuron/8-synapse topologies.
    - brain.nndf is backed up before the run and restored afterward.
    - Full telemetry is collected and Beer-framework analytics are computed
      via compute_all() for every simulation.
    - Phase portrait and XY trajectory data are stored only at quartile
      points (t=0, 0.25, 0.5, 0.75, 1.0) to limit memory usage.

Outputs:
    artifacts/causal_surgery_interpolation.json
    artifacts/plots/surg_fig01_heatmap.png
    artifacts/plots/surg_fig02_dx_waterfall.png
    artifacts/plots/surg_fig03_critical_path.png
    artifacts/plots/surg_fig04_sign_flip.png
    artifacts/plots/surg_fig05_interp_dx.png
    artifacts/plots/surg_fig06_interp_multi.png
    artifacts/plots/surg_fig07_interp_phase.png
    artifacts/plots/surg_fig08_interp_xy.png

Usage:
    python3 causal_surgery_interpolation.py
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
from matplotlib.gridspec import GridSpec

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from compute_beer_analytics import compute_all, DT, NumpyEncoder, _fft_peak

PLOT_DIR = PROJECT / "artifacts" / "plots"
OUT_JSON = PROJECT / "artifacts" / "causal_surgery_interpolation.json"

# ── Champion definitions ─────────────────────────────────────────────────────

NOVELTY_CHAMPION = {
    "name": "Novelty Champion",
    "architecture": "standard",
    "weights": {
        "w03": -1.3083167156740476,
        "w04": -0.34279812804233867,
        "w13": 0.8331363773051514,
        "w14": -0.37582983217830773,
        "w23": -0.0369713954829298,
        "w24": 0.4375020967145814,
    },
}

TRIAL3 = {
    "name": "Trial 3",
    "architecture": "standard",
    "weights": {
        "w03": -0.5971393487736976,
        "w04": -0.4236677331634211,
        "w13": 0.11222931078528431,
        "w14": -0.004679977731207874,
        "w23": 0.2970146930268889,
        "w24": 0.21399448704946855,
    },
}

# CPG Champion: hidden-layer half-center oscillator (7 neurons, 8 synapses)
CPG_CHAMPION = {
    "name": "CPG Champion",
    "architecture": "cpg",
    "synapses": [
        {"src": "1", "tgt": "5", "w": -0.6, "label": "BackLeg→Hidden5"},
        {"src": "2", "tgt": "6", "w": -0.6, "label": "FrontLeg→Hidden6"},
        {"src": "5", "tgt": "6", "w":  0.7, "label": "Hidden5→Hidden6"},
        {"src": "6", "tgt": "5", "w": -0.7, "label": "Hidden6→Hidden5"},
        {"src": "5", "tgt": "3", "w": -0.8, "label": "Hidden5→BackMotor"},
        {"src": "6", "tgt": "4", "w":  0.8, "label": "Hidden6→FrontMotor"},
        {"src": "0", "tgt": "3", "w": -0.3, "label": "Torso→BackMotor"},
        {"src": "0", "tgt": "4", "w":  0.3, "label": "Torso→FrontMotor"},
    ],
}


# ── Shared simulation infrastructure ─────────────────────────────────────────

def write_brain_standard(weights):
    """Write brain.nndf for the standard 6-synapse topology.

    Defines 3 sensor neurons (Torso, BackLeg, FrontLeg), 2 motor neurons
    (Torso_BackLeg, Torso_FrontLeg), and 6 fully-connected synapses.

    Args:
        weights: dict with keys "w03","w04","w13","w14","w23","w24"
                 mapping synapse names to float values.

    Side effects:
        Overwrites brain.nndf in the project directory.
    """
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


def write_brain_cpg(synapses):
    """Write brain.nndf for the CPG hidden-layer topology (7 neurons, 8 synapses).

    Defines 3 sensor neurons, 2 motor neurons, and 2 hidden neurons (5, 6)
    forming a half-center oscillator. Synapses are specified as a list of
    dicts rather than a weight-name dict.

    Args:
        synapses: list of dicts, each with keys "src", "tgt", "w" (and
                  optionally "label"). Example:
                  {"src": "5", "tgt": "6", "w": 0.7, "label": "Hidden5->Hidden6"}

    Side effects:
        Overwrites brain.nndf in the project directory.
    """
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        f.write('    <neuron name = "5" type = "hidden" />\n')
        f.write('    <neuron name = "6" type = "hidden" />\n')
        for syn in synapses:
            f.write(f'    <synapse sourceNeuronName = "{syn["src"]}" '
                    f'targetNeuronName = "{syn["tgt"]}" weight = "{syn["w"]}" />\n')
        f.write('</neuralNetwork>\n')


def simulate_full(architecture="standard", weights=None, synapses=None):
    """Run a full headless simulation and return per-step telemetry data.

    Supports both the standard 6-synapse topology and the CPG hidden-layer
    topology. Collects position, velocity, angular velocity, orientation,
    ground contact, and joint state at every timestep.

    Args:
        architecture: "standard" for 6-synapse or "cpg" for hidden-layer.
        weights: dict of synapse weights (required when architecture="standard").
        synapses: list of synapse dicts (required when architecture="cpg").

    Returns:
        dict of numpy arrays with keys: t, x, y, z, vx, vy, vz,
        wx, wy, wz, roll, pitch, yaw, contact_torso, contact_back,
        contact_front, j0_pos, j0_vel, j0_tau, j1_pos, j1_vel, j1_tau.

    Side effects:
        - Overwrites brain.nndf via write_brain_standard() or write_brain_cpg().
        - Creates and destroys a PyBullet physics connection.
    """
    if architecture == "standard":
        write_brain_standard(weights)
    else:
        write_brain_cpg(synapses)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(c.DT)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    # Apply uniform friction to every link (including base at index -1)
    # to match the friction model used during evolution
    mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
    for link in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, link, lateralFriction=mu, restitution=0.0)

    nn = NEURAL_NETWORK("brain.nndf")
    max_force = float(getattr(c, "MAX_FORCE", 150.0))
    n_steps = c.SIM_STEPS

    t_arr = np.empty(n_steps)
    x = np.empty(n_steps); y = np.empty(n_steps); z = np.empty(n_steps)
    vx = np.empty(n_steps); vy = np.empty(n_steps); vz = np.empty(n_steps)
    wx = np.empty(n_steps); wy = np.empty(n_steps); wz = np.empty(n_steps)
    roll_arr = np.empty(n_steps); pitch_arr = np.empty(n_steps); yaw_arr = np.empty(n_steps)
    ct = np.empty(n_steps, dtype=bool)
    cb = np.empty(n_steps, dtype=bool)
    cf = np.empty(n_steps, dtype=bool)
    j0p = np.empty(n_steps); j0v = np.empty(n_steps); j0t = np.empty(n_steps)
    j1p = np.empty(n_steps); j1v = np.empty(n_steps); j1t = np.empty(n_steps)

    link_indices = {}
    joint_indices = {}
    for i_j in range(p.getNumJoints(robotId)):
        info = p.getJointInfo(robotId, i_j)
        jname = info[1].decode("utf-8") if isinstance(info[1], bytes) else info[1]
        lname = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
        link_indices[lname] = i_j
        joint_indices[jname] = i_j

    back_li = link_indices.get("BackLeg", -1)
    front_li = link_indices.get("FrontLeg", -1)
    j0_idx = joint_indices.get("Torso_BackLeg", 0)
    j1_idx = joint_indices.get("Torso_FrontLeg", 1)

    for i in range(n_steps):
        for nName in nn.neurons:
            n_obj = nn.neurons[nName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, n_obj.Get_Value(), max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, p.POSITION_CONTROL,
                                                n_obj.Get_Value(), max_force)
        p.stepSimulation()
        nn.Update()

        t_arr[i] = i * c.DT
        pos, orn = p.getBasePositionAndOrientation(robotId)
        vel_lin, vel_ang = p.getBaseVelocity(robotId)
        rpy_vals = p.getEulerFromQuaternion(orn)

        x[i] = pos[0]; y[i] = pos[1]; z[i] = pos[2]
        vx[i] = vel_lin[0]; vy[i] = vel_lin[1]; vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]; wy[i] = vel_ang[1]; wz[i] = vel_ang[2]
        roll_arr[i] = rpy_vals[0]; pitch_arr[i] = rpy_vals[1]; yaw_arr[i] = rpy_vals[2]

        # Determine ground contact per link: cp[3] is the link index on bodyA
        contact_pts = p.getContactPoints(robotId)
        tc = bc = fc = False
        for cp in contact_pts:
            li = cp[3]
            if li == -1: tc = True
            elif li == back_li: bc = True
            elif li == front_li: fc = True
        ct[i] = tc; cb[i] = bc; cf[i] = fc

        js0 = p.getJointState(robotId, j0_idx)
        js1 = p.getJointState(robotId, j1_idx)
        j0p[i] = js0[0]; j0v[i] = js0[1]; j0t[i] = js0[3]
        j1p[i] = js1[0]; j1v[i] = js1[1]; j1t[i] = js1[3]

    p.disconnect()

    return {
        "t": t_arr, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll_arr, "pitch": pitch_arr, "yaw": yaw_arr,
        "contact_torso": ct, "contact_back": cb, "contact_front": cf,
        "j0_pos": j0p, "j0_vel": j0v, "j0_tau": j0t,
        "j1_pos": j1p, "j1_vel": j1v, "j1_tau": j1t,
    }


def compute_metrics(data):
    """Run Beer-framework analytics and extract a flat dict of key metrics.

    Computes the full analytics pipeline via compute_all(), then extracts
    13 scalar metrics plus FFT-based joint frequencies.

    Args:
        data: dict of numpy arrays from simulate_full().

    Returns:
        dict with keys: dx, dy, net_distance, mean_speed, speed_cv,
        work_proxy, distance_per_work, path_straightness,
        heading_consistency, phase_lock, contact_entropy, j0_freq, j1_freq.
    """
    a = compute_all(data, DT)
    x, y = data["x"], data["y"]
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    net_distance = np.sqrt(dx**2 + dy**2)

    j0_freq, _ = _fft_peak(data["j0_pos"], DT)
    j1_freq, _ = _fft_peak(data["j1_pos"], DT)

    return {
        "dx": dx,
        "dy": dy,
        "net_distance": float(net_distance),
        "mean_speed": a["outcome"]["mean_speed"],
        "speed_cv": a["outcome"]["speed_cv"],
        "work_proxy": a["outcome"]["work_proxy"],
        "distance_per_work": a["outcome"]["distance_per_work"],
        "path_straightness": a["outcome"]["path_straightness"],
        "heading_consistency": a["outcome"]["heading_consistency"],
        "phase_lock": a["coordination"]["phase_lock_score"],
        "contact_entropy": a["contact"]["contact_entropy_bits"],
        "j0_freq": j0_freq,
        "j1_freq": j1_freq,
    }


# ── Part 1: Causal Surgery ───────────────────────────────────────────────────

ABLATION_MODES = ["zero", "half", "negate"]
SYNAPSE_KEYS_6 = ["w03", "w04", "w13", "w14", "w23", "w24"]

SYNAPSE_LABELS_6 = {
    "w03": "Torso→Back",
    "w04": "Torso→Front",
    "w13": "BackLeg→Back",
    "w14": "BackLeg→Front",
    "w23": "FrontLeg→Back",
    "w24": "FrontLeg→Front",
}


def ablate_standard(weights, synapse_key, mode):
    """Return a new weight dict with one synapse ablated.

    Args:
        weights: dict mapping synapse names to float values.
        synapse_key: which synapse to ablate (e.g., "w03").
        mode: ablation type -- "zero" (set to 0), "half" (multiply by 0.5),
              or "negate" (flip sign).

    Returns:
        New weight dict with the specified synapse modified. The original
        dict is not mutated.
    """
    w = dict(weights)
    orig = w[synapse_key]
    if mode == "zero":
        w[synapse_key] = 0.0
    elif mode == "half":
        w[synapse_key] = orig * 0.5
    elif mode == "negate":
        w[synapse_key] = -orig
    return w


def ablate_cpg(synapses, syn_idx, mode):
    """Return a new synapse list with one synapse ablated.

    Args:
        synapses: list of synapse dicts (each with "src", "tgt", "w").
        syn_idx: index of the synapse to ablate.
        mode: ablation type -- "zero", "half", or "negate".

    Returns:
        New list of synapse dicts (deep-copied) with the specified synapse
        modified. The original list is not mutated.
    """
    new_syns = [dict(s) for s in synapses]
    orig = new_syns[syn_idx]["w"]
    if mode == "zero":
        new_syns[syn_idx]["w"] = 0.0
    elif mode == "half":
        new_syns[syn_idx]["w"] = orig * 0.5
    elif mode == "negate":
        new_syns[syn_idx]["w"] = -orig
    return new_syns


def run_surgery(champion):
    """Run baseline + all ablation combinations for a single champion.

    For each synapse in the champion's topology, runs 3 ablation modes
    (zero, half, negate) and computes absolute and percentage deltas
    vs the unablated baseline for every metric.

    Args:
        champion: dict with keys "name", "architecture", and either
                  "weights" (standard) or "synapses" (CPG).

    Returns:
        dict with keys:
            name: champion name string.
            architecture: "standard" or "cpg".
            baseline: metrics dict from the unablated simulation.
            ablations: list of dicts, each containing synapse_key,
                synapse_label, synapse_idx, original_weight,
                ablated_weight, mode, metrics, deltas, pct_deltas.
    """
    name = champion["name"]
    arch = champion["architecture"]
    print(f"\n  Surgery: {name} ({arch})")

    # Baseline
    if arch == "standard":
        data = simulate_full(architecture="standard", weights=champion["weights"])
    else:
        data = simulate_full(architecture="cpg", synapses=champion["synapses"])
    baseline = compute_metrics(data)
    print(f"    Baseline DX: {baseline['dx']:.2f}")

    ablations = []

    if arch == "standard":
        synapse_keys = SYNAPSE_KEYS_6
        synapse_labels = [SYNAPSE_LABELS_6[k] for k in synapse_keys]
        for si, skey in enumerate(synapse_keys):
            for mode in ABLATION_MODES:
                w_abl = ablate_standard(champion["weights"], skey, mode)
                data = simulate_full(architecture="standard", weights=w_abl)
                metrics = compute_metrics(data)

                # Compute absolute and percentage deltas vs the un-ablated baseline
                deltas = {}
                pct_deltas = {}
                for mk in baseline:
                    if isinstance(baseline[mk], (int, float)):
                        d = metrics[mk] - baseline[mk]
                        deltas[mk] = d
                        # Guard against near-zero baselines to avoid division blow-up
                        if abs(baseline[mk]) > 1e-12:
                            pct_deltas[mk] = d / abs(baseline[mk]) * 100.0
                        else:
                            pct_deltas[mk] = 0.0

                ablations.append({
                    "synapse_key": skey,
                    "synapse_label": synapse_labels[si],
                    "synapse_idx": si,
                    "original_weight": champion["weights"][skey],
                    "ablated_weight": w_abl[skey],
                    "mode": mode,
                    "metrics": metrics,
                    "deltas": deltas,
                    "pct_deltas": pct_deltas,
                })
    else:
        # CPG architecture
        synapses = champion["synapses"]
        for si, syn in enumerate(synapses):
            for mode in ABLATION_MODES:
                syn_abl = ablate_cpg(synapses, si, mode)
                data = simulate_full(architecture="cpg", synapses=syn_abl)
                metrics = compute_metrics(data)

                deltas = {}
                pct_deltas = {}
                for mk in baseline:
                    if isinstance(baseline[mk], (int, float)):
                        d = metrics[mk] - baseline[mk]
                        deltas[mk] = d
                        if abs(baseline[mk]) > 1e-12:
                            pct_deltas[mk] = d / abs(baseline[mk]) * 100.0
                        else:
                            pct_deltas[mk] = 0.0

                ablations.append({
                    "synapse_key": f"{syn['src']}→{syn['tgt']}",
                    "synapse_label": syn["label"],
                    "synapse_idx": si,
                    "original_weight": syn["w"],
                    "ablated_weight": syn_abl[si]["w"],
                    "mode": mode,
                    "metrics": metrics,
                    "deltas": deltas,
                    "pct_deltas": pct_deltas,
                })

    n_abl = len(ablations)
    print(f"    Completed {n_abl} ablations")

    return {
        "name": name,
        "architecture": arch,
        "baseline": baseline,
        "ablations": ablations,
    }


# ── Part 2: Gait Interpolation ───────────────────────────────────────────────

def interpolate_weights(w_a, w_b, t):
    """Linear interpolation between two weight dicts: w(t) = (1-t)*w_A + t*w_B.

    Args:
        w_a: weight dict for gait A (t=0 endpoint).
        w_b: weight dict for gait B (t=1 endpoint).
        t: interpolation parameter in [0, 1].

    Returns:
        New weight dict with linearly interpolated values.
    """
    return {k: (1.0 - t) * w_a[k] + t * w_b[k] for k in w_a}


def run_interpolation(pair_name, w_a, w_b, n_steps=51):
    """Run interpolation between two 6-synapse gaits along a linear transect.

    Simulates n_steps evenly spaced points from gait A (t=0) to gait B
    (t=1) in 6D weight space. Full XY trajectory and phase portrait data
    are stored only at quartile points to limit memory.

    Args:
        pair_name: descriptive label for the pair (e.g., "Nov<->T3").
        w_a: weight dict for gait A (t=0 endpoint).
        w_b: weight dict for gait B (t=1 endpoint).
        n_steps: number of interpolation points. Defaults to 51.

    Returns:
        list of result dicts, one per interpolation point, each with:
            t: interpolation parameter (float 0..1).
            weights: interpolated weight dict.
            metrics: full metrics dict from compute_metrics().
            xy_trajectory: dict with "x","y" lists, or None if not a quartile.
            phase_data: dict with "j0_pos","j1_pos" lists, or None.
    """
    print(f"\n  Interpolation: {pair_name} ({n_steps} points)")
    t_values = np.linspace(0.0, 1.0, n_steps)
    results = []

    for i, t in enumerate(t_values):
        # Walk along the linear transect in 6D weight space from gait A to gait B
        w_interp = interpolate_weights(w_a, w_b, t)
        data = simulate_full(architecture="standard", weights=w_interp)
        metrics = compute_metrics(data)

        # Only store full trajectory/phase data at quartile points (t=0, 0.25, 0.5, 0.75, 1.0)
        # to keep memory usage reasonable while capturing phase portrait evolution
        results.append({
            "t": float(t),
            "weights": w_interp,
            "metrics": metrics,
            "xy_trajectory": {
                "x": data["x"].tolist(),
                "y": data["y"].tolist(),
            } if abs(t - round(t * 4) / 4) < 0.01 else None,
            "phase_data": {
                "j0_pos": data["j0_pos"].tolist(),
                "j1_pos": data["j1_pos"].tolist(),
            } if abs(t - round(t * 4) / 4) < 0.01 else None,
        })

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{n_steps}] t={t:.2f}  DX={metrics['dx']:.2f}")

    return results


# ── Plotting helpers ──────────────────────────────────────────────────────────

def clean_ax(ax):
    """Remove top and right spines for a cleaner plot appearance.

    Args:
        ax: matplotlib Axes instance.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    """Save figure to PLOT_DIR as PNG at 100 dpi and close it.

    Args:
        fig: matplotlib Figure instance.
        name: filename (e.g., "surg_fig01_heatmap.png").

    Side effects:
        Creates PLOT_DIR if it does not exist.
        Writes the PNG file and closes the figure.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


CHAMPION_COLORS = {
    "Novelty Champion": "#C44E52",
    "CPG Champion": "#4C72B0",
    "Trial 3": "#55A868",
}

PAIR_COLORS = {
    "Nov↔T3": "#8B5CF6",
    "Nov↔Dead": "#EC4899",
    "T3↔Dead": "#F59E0B",
}


# ── Surgery figures ───────────────────────────────────────────────────────────

def plot_surgery_heatmap(surgery_results):
    """Fig 1: 3-panel heatmap of metric change (%) per synapse per ablation type.

    One panel per champion. Rows are grouped by synapse with 3 sub-rows
    per ablation mode (zero, half, negate). Columns are 6 key metrics.
    Color scale is diverging RdBu_r, capped at +/-100%.

    Args:
        surgery_results: list of 3 result dicts from run_surgery().

    Side effects:
        Writes surg_fig01_heatmap.png to PLOT_DIR.
    """
    heatmap_metrics = ["dx", "mean_speed", "work_proxy", "phase_lock",
                       "heading_consistency", "contact_entropy"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))

    for col, sr in enumerate(surgery_results):
        ax = axes[col]
        name = sr["name"]
        ablations = sr["ablations"]

        # Get unique synapse labels (in order)
        seen = []
        syn_labels = []
        for abl in ablations:
            key = abl["synapse_key"]
            if key not in seen:
                seen.append(key)
                syn_labels.append(abl["synapse_label"])

        n_syn = len(seen)
        n_modes = len(ABLATION_MODES)
        n_metrics = len(heatmap_metrics)

        # Build matrix: rows = synapses × modes (grouped by synapse), cols = metrics
        # Each synapse gets 3 consecutive rows (zero, half, negate)
        n_rows = n_syn * n_modes
        matrix = np.zeros((n_rows, n_metrics))
        row_labels = []

        for abl in ablations:
            si = abl["synapse_idx"]
            mi = ABLATION_MODES.index(abl["mode"])
            # Interleave modes within each synapse group
            row_idx = si * n_modes + mi
            for ci, mk in enumerate(heatmap_metrics):
                matrix[row_idx, ci] = abl["pct_deltas"].get(mk, 0.0)
            row_labels.append(f"{abl['synapse_label']}|{abl['mode']}")

        # Cap colour scale at ±100% so extreme outliers don't wash out detail
        vmax = min(100, np.max(np.abs(matrix)) * 1.1) if np.max(np.abs(matrix)) > 0 else 100
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(heatmap_metrics, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_title(f"{name}", fontsize=10, color=CHAMPION_COLORS.get(name, "black"))

        # Add grid lines between synapses
        for si in range(1, n_syn):
            ax.axhline(si * n_modes - 0.5, color="black", lw=0.5, alpha=0.3)

    fig.colorbar(im, ax=axes, label="Change from baseline (%)", shrink=0.8)
    fig.suptitle("Causal Surgery: Metric Change (%) by Synapse and Ablation Type", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    save_fig(fig, "surg_fig01_heatmap.png")


def plot_surgery_waterfall(surgery_results):
    """Fig 2: Waterfall chart showing baseline DX vs ablated DX (zero mode only).

    One panel per champion. Bars show the ablated DX for each synapse,
    colored green (improvement) or red (degradation). A dashed baseline
    line shows the unablated DX. Delta labels are placed on each bar.

    Args:
        surgery_results: list of 3 result dicts from run_surgery().

    Side effects:
        Writes surg_fig02_dx_waterfall.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for col, sr in enumerate(surgery_results):
        ax = axes[col]
        name = sr["name"]
        baseline_dx = sr["baseline"]["dx"]
        color = CHAMPION_COLORS.get(name, "#333333")

        # Get zero-mode ablations only
        zero_abls = [a for a in sr["ablations"] if a["mode"] == "zero"]
        syn_labels = [a["synapse_label"] for a in zero_abls]
        ablated_dxs = [a["metrics"]["dx"] for a in zero_abls]
        n = len(zero_abls)

        x_pos = np.arange(n)
        bar_heights = [adx - baseline_dx for adx in ablated_dxs]
        bar_colors = ["#C44E52" if h < 0 else "#55A868" for h in bar_heights]

        ax.axhline(baseline_dx, color=color, ls="--", lw=1.5, alpha=0.7,
                    label=f"Baseline: {baseline_dx:.1f}m")
        bars = ax.bar(x_pos, ablated_dxs, color=bar_colors, alpha=0.8, width=0.6)

        # Add delta labels on bars
        for xi, (adx, bh) in enumerate(zip(ablated_dxs, bar_heights)):
            ax.text(xi, adx + np.sign(bh) * 0.5, f"{bh:+.1f}",
                    ha="center", va="bottom" if bh >= 0 else "top", fontsize=7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(syn_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("DX (m)")
        ax.set_title(f"{name}", fontsize=10, color=color)
        ax.legend(fontsize=8)
        clean_ax(ax)

    fig.suptitle("Causal Surgery: DX After Zeroing Each Synapse", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "surg_fig02_dx_waterfall.png")


def plot_surgery_critical_path(surgery_results):
    """Fig 3: Horizontal bar chart ranking synapses by |delta-DX| when zeroed.

    One panel per champion. Synapses are sorted from most to least
    impactful. This identifies the critical path: the synapses whose
    removal most disrupts locomotion.

    Args:
        surgery_results: list of 3 result dicts from run_surgery().

    Side effects:
        Writes surg_fig03_critical_path.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for col, sr in enumerate(surgery_results):
        ax = axes[col]
        name = sr["name"]
        color = CHAMPION_COLORS.get(name, "#333333")

        zero_abls = [a for a in sr["ablations"] if a["mode"] == "zero"]
        labels = [a["synapse_label"] for a in zero_abls]
        abs_delta_dx = [abs(a["deltas"]["dx"]) for a in zero_abls]

        # Sort by |ΔDX| descending
        sorted_idx = np.argsort(abs_delta_dx)[::-1]
        sorted_labels = [labels[i] for i in sorted_idx]
        sorted_deltas = [abs_delta_dx[i] for i in sorted_idx]

        x_pos = np.arange(len(sorted_labels))
        ax.barh(x_pos, sorted_deltas, color=color, alpha=0.8, height=0.6)

        for xi, d in enumerate(sorted_deltas):
            ax.text(d + 0.3, xi, f"{d:.1f}m", va="center", fontsize=8)

        ax.set_yticks(x_pos)
        ax.set_yticklabels(sorted_labels, fontsize=9)
        ax.set_xlabel("|ΔDX| (m)")
        ax.set_title(f"{name}", fontsize=10, color=color)
        ax.invert_yaxis()
        clean_ax(ax)

    fig.suptitle("Critical Path: Synapse Impact on DX (Zero Ablation)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "surg_fig03_critical_path.png")


def plot_surgery_sign_flip(surgery_results):
    """Fig 4: Scatter of baseline DX vs sign-flipped (negated) DX per synapse.

    All 3 champions are overlaid. Points on the y=x diagonal indicate
    sign-flips that had no effect; points far below indicate catastrophic
    flips that destroyed locomotion.

    Args:
        surgery_results: list of 3 result dicts from run_surgery().

    Side effects:
        Writes surg_fig04_sign_flip.png to PLOT_DIR.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for sr in surgery_results:
        name = sr["name"]
        baseline_dx = sr["baseline"]["dx"]
        color = CHAMPION_COLORS.get(name, "#333333")

        negate_abls = [a for a in sr["ablations"] if a["mode"] == "negate"]
        for a in negate_abls:
            flipped_dx = a["metrics"]["dx"]
            ax.scatter(baseline_dx, flipped_dx, c=color, s=80, alpha=0.7,
                       edgecolors="white", linewidths=0.5, zorder=3)
            ax.annotate(a["synapse_label"], (baseline_dx, flipped_dx),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(5, 3), textcoords="offset points")

    # Diagonal = sign flip had no effect; points below = flip hurt locomotion
    lims = ax.get_xlim()
    ax.plot([-100, 100], [-100, 100], "k--", lw=0.5, alpha=0.3, label="No change")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)

    # Legend for champions
    for name, color in CHAMPION_COLORS.items():
        ax.scatter([], [], c=color, s=60, label=name)
    ax.legend(fontsize=8, loc="upper left")

    ax.set_xlabel("Baseline DX (m)")
    ax.set_ylabel("Sign-Flipped DX (m)")
    ax.set_title("Sign Flip Surgery: Which Flips Are Catastrophic vs Constructive?", fontsize=12)
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "surg_fig04_sign_flip.png")


# ── Interpolation figures ─────────────────────────────────────────────────────

def plot_interp_dx(interp_results):
    """Fig 5: DX vs interpolation parameter t for all 3 gait pairs.

    Shows how displacement changes along the linear transect between
    two gaits. Horizontal dotted lines mark the pure endpoint DX values.
    Cliffs (sudden jumps) and smooth gradients are visually apparent.

    Args:
        interp_results: dict keyed by pair name (e.g., "Nov<->T3"), each
            value a list of result dicts from run_interpolation().

    Side effects:
        Writes surg_fig05_interp_dx.png to PLOT_DIR.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for pair_name, results in interp_results.items():
        ts = [r["t"] for r in results]
        dxs = [r["metrics"]["dx"] for r in results]
        color = PAIR_COLORS.get(pair_name, "#333333")
        ax.plot(ts, dxs, color=color, lw=2, label=pair_name, alpha=0.9)

        # Horizontal dashed lines mark DX of the two pure endpoints (t=0, t=1)
        ax.axhline(dxs[0], color=color, ls=":", lw=0.8, alpha=0.4)
        ax.axhline(dxs[-1], color=color, ls=":", lw=0.8, alpha=0.4)

    ax.set_xlabel("Interpolation t (0 = gait A, 1 = gait B)")
    ax.set_ylabel("DX (m)")
    ax.set_title("Gait Interpolation: DX Across the Landscape", fontsize=13)
    ax.legend(fontsize=9)
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "surg_fig05_interp_dx.png")


def plot_interp_multi(interp_results):
    """Fig 6: 2x3 grid of 6 metrics vs interpolation parameter t.

    Shows DX, mean speed, work proxy, phase lock, heading consistency,
    and contact entropy landscapes for all 3 gait pairs overlaid.
    Reveals which metrics change smoothly vs abruptly.

    Args:
        interp_results: dict keyed by pair name, each value a list of
            result dicts from run_interpolation().

    Side effects:
        Writes surg_fig06_interp_multi.png to PLOT_DIR.
    """
    metric_keys = ["dx", "mean_speed", "work_proxy",
                   "phase_lock", "heading_consistency", "contact_entropy"]
    metric_labels = ["DX (m)", "Mean Speed (m/s)", "Work Proxy",
                     "Phase Lock", "Heading Consistency", "Contact Entropy (bits)"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ai, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[ai // 3][ai % 3]

        for pair_name, results in interp_results.items():
            ts = [r["t"] for r in results]
            vals = [r["metrics"][mkey] for r in results]
            color = PAIR_COLORS.get(pair_name, "#333333")
            ax.plot(ts, vals, color=color, lw=1.5, label=pair_name, alpha=0.9)

        ax.set_xlabel("t")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontsize=10)
        if ai == 0:
            ax.legend(fontsize=7)
        clean_ax(ax)

    fig.suptitle("Gait Interpolation: Multi-Metric Landscape", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "surg_fig06_interp_multi.png")


def plot_interp_phase(primary_results):
    """Fig 7: Phase portraits (BackLeg vs FrontLeg joint angle) at 5 quartile points.

    Shows how the joint-angle limit cycle morphs as the interpolation
    parameter moves from gait A to gait B. Points are colored by time
    (viridis) to indicate phase portrait direction.

    Args:
        primary_results: list of result dicts from run_interpolation()
            for the primary pair (Nov<->T3). Only entries with non-None
            phase_data are plotted.

    Side effects:
        Writes surg_fig07_interp_phase.png to PLOT_DIR.
    """
    t_targets = [0.0, 0.25, 0.50, 0.75, 1.0]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4))

    for ai, t_target in enumerate(t_targets):
        ax = axes[ai]
        # Find closest result with phase data
        best = None
        best_dist = float("inf")
        for r in primary_results:
            if r["phase_data"] is not None:
                d = abs(r["t"] - t_target)
                if d < best_dist:
                    best_dist = d
                    best = r

        if best is not None:
            j0 = np.array(best["phase_data"]["j0_pos"])
            j1 = np.array(best["phase_data"]["j1_pos"])
            n = len(j0)
            # Subsample to 500 evenly-spaced points for the scatter overlay;
            # colour by time (viridis) to show phase portrait evolution direction
            scatter_idx = np.linspace(0, n - 1, 500, dtype=int)
            ax.plot(j0, j1, color="#8B5CF6", lw=0.3, alpha=0.3)
            ax.scatter(j0[scatter_idx], j1[scatter_idx], c=scatter_idx,
                       cmap="viridis", s=5, alpha=0.6)
            dx_val = best["metrics"]["dx"]
            pl_val = best["metrics"]["phase_lock"]
            ax.set_title(f"t={best['t']:.2f}\nDX={dx_val:.1f} PL={pl_val:.2f}", fontsize=9)
        else:
            ax.set_title(f"t={t_target:.2f}\n(no data)", fontsize=9)

        ax.set_xlabel("BackLeg (rad)")
        if ai == 0:
            ax.set_ylabel("FrontLeg (rad)")
        clean_ax(ax)

    fig.suptitle("Phase Portrait Morphing: Novelty Champion → Trial 3", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "surg_fig07_interp_phase.png")


def plot_interp_xy(primary_results):
    """Fig 8: XY trajectory gallery at 5 quartile interpolation points.

    Shows the robot's ground-plane path (X vs Y) at t=0, 0.25, 0.5,
    0.75, 1.0. Start is marked with a green triangle, end with a red
    star. Equal aspect ratio preserves trajectory shape.

    Args:
        primary_results: list of result dicts from run_interpolation()
            for the primary pair (Nov<->T3). Only entries with non-None
            xy_trajectory are plotted.

    Side effects:
        Writes surg_fig08_interp_xy.png to PLOT_DIR.
    """
    t_targets = [0.0, 0.25, 0.50, 0.75, 1.0]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4))

    for ai, t_target in enumerate(t_targets):
        ax = axes[ai]
        best = None
        best_dist = float("inf")
        for r in primary_results:
            if r["xy_trajectory"] is not None:
                d = abs(r["t"] - t_target)
                if d < best_dist:
                    best_dist = d
                    best = r

        if best is not None:
            xx = np.array(best["xy_trajectory"]["x"])
            yy = np.array(best["xy_trajectory"]["y"])
            ax.plot(xx, yy, color="#8B5CF6", lw=0.8, alpha=0.8)
            ax.scatter([xx[0]], [yy[0]], c="green", s=40, marker="^", zorder=5)
            ax.scatter([xx[-1]], [yy[-1]], c="red", s=40, marker="*", zorder=5)
            dx_val = best["metrics"]["dx"]
            ax.set_title(f"t={best['t']:.2f}\nDX={dx_val:.1f}m", fontsize=9)
        else:
            ax.set_title(f"t={t_target:.2f}\n(no data)", fontsize=9)

        ax.set_xlabel("X (m)")
        if ai == 0:
            ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        clean_ax(ax)

    fig.suptitle("XY Trajectories: Novelty Champion → Trial 3", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "surg_fig08_interp_xy.png")


# ── Console tables ────────────────────────────────────────────────────────────

def print_surgery_table(sr):
    """Print a formatted surgery results table for one champion to stdout.

    Displays baseline metrics followed by one row per ablation, showing
    DX, delta-DX, percentage change, speed, phase lock, heading
    consistency, and contact entropy.

    Args:
        sr: surgery result dict from run_surgery().

    Side effects:
        Prints to stdout.
    """
    name = sr["name"]
    baseline = sr["baseline"]

    print(f"\n{'='*80}")
    print(f"  SURGERY: {name}")
    print(f"  Baseline DX={baseline['dx']:.2f}  Speed={baseline['mean_speed']:.3f}  "
          f"Work={baseline['work_proxy']:.0f}  PhaseLock={baseline['phase_lock']:.3f}")
    print(f"{'='*80}")

    print(f"\n  {'Synapse':<20} {'Mode':<8} {'DX':>8} {'ΔDX':>8} {'ΔDX%':>7} "
          f"{'Speed':>7} {'PL':>6} {'HdgC':>6} {'CE':>6}")
    print("  " + "-" * 85)

    for a in sr["ablations"]:
        m = a["metrics"]
        d = a["deltas"]
        pct = a["pct_deltas"]
        print(f"  {a['synapse_label']:<20} {a['mode']:<8} {m['dx']:>8.2f} "
              f"{d['dx']:>+8.2f} {pct['dx']:>+6.1f}% "
              f"{m['mean_speed']:>7.3f} {m['phase_lock']:>6.3f} "
              f"{m['heading_consistency']:>6.3f} {m['contact_entropy']:>6.3f}")


def print_interpolation_summary(pair_name, results):
    """Print interpolation summary statistics for one gait pair to stdout.

    Reports endpoint DX values, max/min DX with their t values, DX range,
    detected cliffs (>20% jump between adjacent points), and a sampled
    table of metrics at every 5th interpolation point.

    Args:
        pair_name: descriptive label for the pair.
        results: list of result dicts from run_interpolation().

    Side effects:
        Prints to stdout.
    """
    print(f"\n{'='*70}")
    print(f"  INTERPOLATION: {pair_name}")
    print(f"{'='*70}")

    dxs = [r["metrics"]["dx"] for r in results]
    speeds = [r["metrics"]["mean_speed"] for r in results]

    max_dx_idx = np.argmax(dxs)
    min_dx_idx = np.argmin(dxs)

    print(f"  Endpoints:  t=0 DX={dxs[0]:.2f}m    t=1 DX={dxs[-1]:.2f}m")
    print(f"  Max DX:     t={results[max_dx_idx]['t']:.2f} DX={dxs[max_dx_idx]:.2f}m")
    print(f"  Min DX:     t={results[min_dx_idx]['t']:.2f} DX={dxs[min_dx_idx]:.2f}m")
    print(f"  DX range:   {max(dxs) - min(dxs):.2f}m")

    # Detect cliffs: adjacent interpolation points where DX jumps by >20%
    # relative to the larger magnitude (clamped to 1.0 to avoid false positives near zero)
    cliffs = []
    for i in range(1, len(dxs)):
        delta = abs(dxs[i] - dxs[i - 1])
        span = max(abs(dxs[i]), abs(dxs[i - 1]), 1.0)
        if delta / span > 0.20:
            cliffs.append((results[i - 1]["t"], results[i]["t"], delta))

    if cliffs:
        print(f"  Cliffs (>20% jump):")
        for t0, t1, delta in cliffs[:5]:
            print(f"    t={t0:.2f}→{t1:.2f}: ΔDX={delta:.2f}m")
    else:
        print(f"  No major cliffs detected — smooth landscape")

    # Print sample points
    print(f"\n  {'t':>5} {'DX':>8} {'Speed':>7} {'Work':>8} {'PL':>6} {'HdgC':>6} {'CE':>6}")
    print("  " + "-" * 50)
    for r in results[::5]:
        m = r["metrics"]
        print(f"  {r['t']:>5.2f} {m['dx']:>8.2f} {m['mean_speed']:>7.3f} "
              f"{m['work_proxy']:>8.0f} {m['phase_lock']:>6.3f} "
              f"{m['heading_consistency']:>6.3f} {m['contact_entropy']:>6.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run the full causal-surgery + gait-interpolation pipeline.

    Pipeline:
        1. Back up brain.nndf (restored at end).
        2. Load a "dead" gait (high path_length but near-zero DX) from
           dark_matter.json as an interpolation endpoint.
        3. Part 1: Run surgery (ablation) on 3 champions (~63 sims).
        4. Part 2: Run interpolation on 3 gait pairs (~153 sims).
        5. Save structured results to artifacts/causal_surgery_interpolation.json.
        6. Generate 8 figures (4 surgery + 4 interpolation) to artifacts/plots/.

    Side effects:
        - Overwrites and restores brain.nndf.
        - Writes JSON and PNG artifacts to artifacts/.
        - Prints progress and summary tables to stdout.
    """
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    t_start = time.perf_counter()

    # Load a "dead" gait (near-zero net displacement) for interpolation.
    # Pick the Canceller with the longest path_length -- it moves a lot but
    # goes nowhere, making it an interesting interpolation endpoint.
    with open(PROJECT / "artifacts" / "dark_matter.json") as f:
        dm = json.load(f)
    cancellers = [g for g in dm["gaits"] if g["type"] == "Canceller"]
    cancellers.sort(key=lambda g: g["descriptors"]["path_length"], reverse=True)
    dead_gait = cancellers[0]
    dead_weights = dead_gait["weights"]
    print(f"Dead gait: trial {dead_gait['trial_idx']}, "
          f"path_length={dead_gait['descriptors']['path_length']:.1f}, "
          f"DX={dead_gait['descriptors']['dx']:.2f}")

    # ── Part 1: Causal Surgery ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 1: CAUSAL SURGERY (ABLATION STUDY)")
    print("=" * 60)

    surgery_results = []
    for champion in [NOVELTY_CHAMPION, CPG_CHAMPION, TRIAL3]:
        sr = run_surgery(champion)
        surgery_results.append(sr)
        print_surgery_table(sr)

    t_surgery = time.perf_counter()
    print(f"\nSurgery complete: {t_surgery - t_start:.1f}s")

    # ── Part 2: Gait Interpolation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 2: GAIT INTERPOLATION")
    print("=" * 60)

    interp_results = {}

    # Pair 1: Novelty Champion ↔ Trial 3
    interp_results["Nov↔T3"] = run_interpolation(
        "Nov↔T3",
        NOVELTY_CHAMPION["weights"],
        TRIAL3["weights"],
        n_steps=51,
    )

    # Pair 2: Novelty Champion ↔ Dead gait
    interp_results["Nov↔Dead"] = run_interpolation(
        "Nov↔Dead",
        NOVELTY_CHAMPION["weights"],
        dead_weights,
        n_steps=51,
    )

    # Pair 3: Trial 3 ↔ Dead gait
    interp_results["T3↔Dead"] = run_interpolation(
        "T3↔Dead",
        TRIAL3["weights"],
        dead_weights,
        n_steps=51,
    )

    for pair_name, results in interp_results.items():
        print_interpolation_summary(pair_name, results)

    t_interp = time.perf_counter()
    print(f"\nInterpolation complete: {t_interp - t_surgery:.1f}s")

    # ── Restore brain.nndf ───────────────────────────────────────────────────
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # ── Save JSON ────────────────────────────────────────────────────────────
    print("\nSaving JSON...")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Strip xy_trajectory and phase_data arrays from JSON output to keep
    # the artifact file small; only scalar metrics and weights are persisted.
    interp_json = {}
    for pair_name, results in interp_results.items():
        interp_json[pair_name] = [{
            "t": r["t"],
            "weights": r["weights"],
            "metrics": r["metrics"],
        } for r in results]

    output = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_surgery_sims": sum(1 + len(sr["ablations"]) for sr in surgery_results),
            "n_interp_sims": sum(len(r) for r in interp_results.values()),
            "total_time_s": t_interp - t_start,
            "dead_gait_trial": dead_gait["trial_idx"],
        },
        "surgery": surgery_results,
        "interpolation": interp_json,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"WROTE {OUT_JSON}")

    # ── Generate figures ─────────────────────────────────────────────────────
    print("\nGenerating figures...")

    plot_surgery_heatmap(surgery_results)
    plot_surgery_waterfall(surgery_results)
    plot_surgery_critical_path(surgery_results)
    plot_surgery_sign_flip(surgery_results)

    plot_interp_dx(interp_results)
    plot_interp_multi(interp_results)

    # For phase portrait + XY gallery, use the primary pair (Nov↔T3)
    plot_interp_phase(interp_results["Nov↔T3"])
    plot_interp_xy(interp_results["Nov↔T3"])

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_time:.1f}s")
    print(f"  Surgery sims:       {sum(1 + len(sr['ablations']) for sr in surgery_results)}")
    print(f"  Interpolation sims: {sum(len(r) for r in interp_results.values())}")
    print(f"  Total sims:         {sum(1 + len(sr['ablations']) for sr in surgery_results) + sum(len(r) for r in interp_results.values())}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
