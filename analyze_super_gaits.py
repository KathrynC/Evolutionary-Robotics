#!/usr/bin/env python3
"""
analyze_super_gaits.py

Role:
    Deep comparison of super-gaits discovered by causal surgery + interpolation
    analysis, compared against the Novelty Champion baseline and the cliff-collapse
    point. Includes local sensitivity probing (+/- 1% perturbation per synapse).

4 gaits compared:
    1. Novelty Champion (NC)          -- DX~60.2m baseline
    2. Interpolation Super (t=0.52)   -- DX~68.2m, 35% less work
    3. w23-Half Variant               -- DX~68.4m, same work as NC
    4. Cliff Collapse (t=0.54)        -- catastrophic collapse to DX~-1.1m

Pipeline:
    Part 1: Simulate all 4 gaits, compute metrics via Beer-framework analytics.
    Part 2: Print full metrics table + weight structure analysis.
    Part 3: Sensitivity probes -- perturb each synapse by +/- 1% for 3 viable
            gaits (6 synapses x 2 directions x 3 gaits = 36 additional sims).
    Part 4: Save JSON artifact and generate 6 figures.

Outputs:
    artifacts/super_gaits_analysis.json
    artifacts/plots/super_fig01_trajectory.png   -- XY path, X vs time, Z bounce
    artifacts/plots/super_fig02_joints.png       -- Joint positions and velocities (2x4)
    artifacts/plots/super_fig03_phase.png        -- Phase portraits (j0 vs j1, 4 panels)
    artifacts/plots/super_fig04_energy.png       -- Instantaneous power + cumulative work
    artifacts/plots/super_fig05_contacts.png     -- Contact rasters (4 stacked panels)
    artifacts/plots/super_fig06_sensitivity.png  -- Bar chart of |dDX/dw| per synapse

Notes:
    - Total simulation count: 4 (gaits) + 36 (sensitivity) = 40 sims.
    - brain.nndf is backed up and restored around the simulation block.
    - The t=0.52 to t=0.54 cliff is the sharpest known discontinuity in the
      interpolation landscape (69m drop in DX over a 0.02 change in t).

Usage:
    python3 analyze_super_gaits.py
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

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from compute_beer_analytics import compute_all, DT, NumpyEncoder, _fft_peak

PLOT_DIR = PROJECT / "artifacts" / "plots"
OUT_JSON = PROJECT / "artifacts" / "super_gaits_analysis.json"

SYNAPSE_KEYS = ["w03", "w04", "w13", "w14", "w23", "w24"]
SYNAPSE_LABELS = {
    "w03": "Torso→Back",
    "w04": "Torso→Front",
    "w13": "BackLeg→Back",
    "w14": "BackLeg→Front",
    "w23": "FrontLeg→Back",
    "w24": "FrontLeg→Front",
}

# ── Gait definitions ─────────────────────────────────────────────────────────

_NC = {
    "w03": -1.3083167156740476,
    "w04": -0.34279812804233867,
    "w13": 0.8331363773051514,
    "w14": -0.37582983217830773,
    "w23": -0.0369713954829298,
    "w24": 0.4375020967145814,
}

_T3 = {
    "w03": -0.5971393487736976,
    "w04": -0.4236677331634211,
    "w13": 0.11222931078528431,
    "w14": -0.004679977731207874,
    "w23": 0.2970146930268889,
    "w24": 0.21399448704946855,
}

def _interp(w_a, w_b, t):
    """Linearly interpolate two weight dicts at parameter t in [0, 1].

    Args:
        w_a: Starting weight dict (returned when t=0).
        w_b: Ending weight dict (returned when t=1).
        t: Interpolation parameter in [0, 1].

    Returns:
        New dict with the same keys, values linearly blended.
    """
    # t=0 gives w_a, t=1 gives w_b
    return {k: (1.0 - t) * w_a[k] + t * w_b[k] for k in w_a}

GAITS = {
    "Novelty Champion": dict(_NC),
    "Interp Super (t=0.52)": _interp(_NC, _T3, 0.52),
    "w23-Half Variant": {
        **_NC,
        "w23": _NC["w23"] / 2.0,  # halved
    },
    "Cliff Collapse (t=0.54)": _interp(_NC, _T3, 0.54),
}

GAIT_ORDER = ["Novelty Champion", "Interp Super (t=0.52)",
              "w23-Half Variant", "Cliff Collapse (t=0.54)"]

GAIT_COLORS = {
    "Novelty Champion":        "#C44E52",
    "Interp Super (t=0.52)":   "#8B5CF6",
    "w23-Half Variant":        "#4C72B0",
    "Cliff Collapse (t=0.54)": "#999999",
}

GAIT_SHORT = {
    "Novelty Champion":        "NC",
    "Interp Super (t=0.52)":   "InterpS",
    "w23-Half Variant":        "w23Half",
    "Cliff Collapse (t=0.54)": "Cliff",
}

# ── Simulation infrastructure (reused from causal_surgery_interpolation.py) ──

def write_brain_standard(weights):
    """Write a brain.nndf file with the given 6-synapse weight dict.

    Args:
        weights: Dict with keys "w03","w13","w23","w04","w14","w24".

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


def simulate_full(weights):
    """Run a full headless PyBullet simulation and return time-series data dict.

    Args:
        weights: Dict of 6-synapse weights (w03..w24).

    Returns:
        Dict with numpy arrays for position (x, y, z), velocity (vx, vy, vz),
        angular velocity (wx, wy, wz), orientation (roll, pitch, yaw),
        ground contacts (contact_torso, contact_back, contact_front), and
        joint states (j0_pos, j0_vel, j0_tau, j1_pos, j1_vel, j1_tau)
        at each of the c.SIM_STEPS timesteps.

    Side effects:
        Overwrites brain.nndf via write_brain_standard.
    """
    write_brain_standard(weights)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(c.DT)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    # Set uniform friction on all links (including base at index -1)
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
    """Compute a flat dict of locomotion metrics from simulation time-series data.

    Wraps compute_all() from the Beer analytics pipeline and extracts commonly
    compared scalar metrics into a single flat dictionary.

    Args:
        data: Telemetry dict from simulate_full().

    Returns:
        Dict with displacement, speed, work, coordination, contact, and
        frequency metrics for a single gait run.
    """
    a = compute_all(data, DT)
    x, y = data["x"], data["y"]
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    # Euclidean displacement ignoring vertical axis
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
        "yaw_net_rad": a["outcome"]["yaw_net_rad"],
        "phase_lock": a["coordination"]["phase_lock_score"],
        "delta_phi_rad": a["coordination"]["delta_phi_rad"],
        "contact_entropy": a["contact"]["contact_entropy_bits"],
        "duty_torso": a["contact"]["duty_torso"],
        "duty_back": a["contact"]["duty_back"],
        "duty_front": a["contact"]["duty_front"],
        "j0_freq": j0_freq,
        "j1_freq": j1_freq,
        "axis_dominance": a["rotation_axis"]["axis_dominance"],
    }


def work_by_joint(data):
    """Compute mechanical work for each joint as integral of |torque * angular velocity|.

    Returns (work_j0, work_j1, power_j0_array, power_j1_array).
    """
    # Instantaneous absolute power: |tau * omega| at each timestep
    pj0 = np.abs(data["j0_tau"] * data["j0_vel"])
    pj1 = np.abs(data["j1_tau"] * data["j1_vel"])
    # Integrate power over time (rectangular rule) to get total work
    return float(np.sum(pj0) * DT), float(np.sum(pj1) * DT), pj0, pj1


# ── Plotting helpers ──────────────────────────────────────────────────────────

def clean_ax(ax):
    """Remove top and right spines for a cleaner plot appearance."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    """Save a matplotlib figure to PLOT_DIR and close it to free memory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


# ── Console tables ────────────────────────────────────────────────────────────

def print_metrics_table(all_metrics):
    """Print a formatted console table comparing all locomotion metrics across 4 gaits.

    Args:
        all_metrics: Dict mapping gait name to its metrics dict (from compute_metrics).
    """
    print(f"\n{'='*100}")
    print("FULL METRICS COMPARISON: 4 GAITS")
    print(f"{'='*100}")

    metric_rows = [
        ("DX (m)", "dx"),
        ("DY (m)", "dy"),
        ("Net distance (m)", "net_distance"),
        ("Mean speed (m/s)", "mean_speed"),
        ("Speed CV", "speed_cv"),
        ("Work proxy", "work_proxy"),
        ("Efficiency (dist/work)", "distance_per_work"),
        ("Path straightness", "path_straightness"),
        ("Heading consistency", "heading_consistency"),
        ("Yaw net (rad)", "yaw_net_rad"),
        ("Phase lock", "phase_lock"),
        ("Delta phi (rad)", "delta_phi_rad"),
        ("Contact entropy (bits)", "contact_entropy"),
        ("Duty torso", "duty_torso"),
        ("Duty back leg", "duty_back"),
        ("Duty front leg", "duty_front"),
        ("Joint 0 freq (Hz)", "j0_freq"),
        ("Joint 1 freq (Hz)", "j1_freq"),
    ]

    header = f"  {'Metric':<26}"
    for name in GAIT_ORDER:
        header += f" {GAIT_SHORT[name]:>10}"
    print(header)
    print("  " + "-" * (26 + 11 * len(GAIT_ORDER)))

    for label, key in metric_rows:
        row = f"  {label:<26}"
        for name in GAIT_ORDER:
            v = all_metrics[name][key]
            if isinstance(v, list):
                row += f" {v[0]:10.4f}"
            elif abs(v) >= 100:
                row += f" {v:10.0f}"
            else:
                row += f" {v:10.4f}"
        print(row)


def print_weight_analysis(all_metrics):
    """Print per-sensor weight decomposition and inter-gait weight deltas.

    Shows sensor-to-motor weight pairs (sum, diff), total drive to each motor,
    total weight magnitude, and the critical delta between t=0.52 and t=0.54
    (the cliff boundary).

    Args:
        all_metrics: Dict mapping gait name to metrics (unused but kept for API consistency).
    """
    print(f"\n{'='*100}")
    print("WEIGHT STRUCTURE ANALYSIS")
    print(f"{'='*100}")

    for name in GAIT_ORDER:
        w = GAITS[name]
        print(f"\n  {name}:")
        for s, sl in [(0, "Torso"), (1, "BackLeg"), (2, "FrontLeg")]:
            w3 = w[f"w{s}3"]
            w4 = w[f"w{s}4"]
            # sum = net excitation from this sensor; diff = motor selectivity
            print(f"    Sensor {s} ({sl:>8s}):  w{s}3={w3:+.4f}  w{s}4={w4:+.4f}"
                  f"  sum={w3+w4:+.4f}  diff={w3-w4:+.4f}")
        # Total afferent drive to each motor across all sensors
        tm3 = w["w03"] + w["w13"] + w["w23"]
        tm4 = w["w04"] + w["w14"] + w["w24"]
        print(f"    Total to m3 (BackMotor):  {tm3:+.4f}")
        print(f"    Total to m4 (FrontMotor): {tm4:+.4f}")
        mag = sum(abs(w[k]) for k in SYNAPSE_KEYS)
        print(f"    Total |w| magnitude:      {mag:.4f}")

    # Delta between t=0.52 and t=0.54
    print(f"\n  DELTA: Interp Super (t=0.52) vs Cliff Collapse (t=0.54):")
    w52 = GAITS["Interp Super (t=0.52)"]
    w54 = GAITS["Cliff Collapse (t=0.54)"]
    for k in SYNAPSE_KEYS:
        d = w54[k] - w52[k]
        print(f"    {SYNAPSE_LABELS[k]:>16s} ({k}): {w52[k]:+.4f} -> {w54[k]:+.4f}  delta={d:+.4f}")
    total_delta = sum(abs(w54[k] - w52[k]) for k in SYNAPSE_KEYS)
    print(f"    Total |delta|: {total_delta:.4f}")


def print_sensitivity_table(sensitivity, all_metrics):
    """Print local sensitivity (|dDX/dw|) and fragility ratios for the 3 viable gaits.

    Args:
        sensitivity: Dict mapping gait name to per-synapse sensitivity results
            (each with "abs_gradient" key).
        all_metrics: Dict mapping gait name to metrics (unused but kept for API consistency).
    """
    print(f"\n{'='*100}")
    print("LOCAL SENSITIVITY: |dDX/dw| per synapse (1% perturbation)")
    print(f"{'='*100}")

    header = f"  {'Synapse':<18}"
    for name in GAIT_ORDER[:3]:  # NC, InterpS, w23Half
        header += f" {GAIT_SHORT[name]:>10}"
    print(header)
    print("  " + "-" * (18 + 11 * 3))

    for k in SYNAPSE_KEYS:
        row = f"  {SYNAPSE_LABELS[k]:<18}"
        for name in GAIT_ORDER[:3]:
            val = sensitivity[name][k]["abs_gradient"]
            row += f" {val:10.2f}"
        print(row)

    # Total sensitivity
    row = f"  {'TOTAL':<18}"
    for name in GAIT_ORDER[:3]:
        total = sum(sensitivity[name][k]["abs_gradient"] for k in SYNAPSE_KEYS)
        row += f" {total:10.2f}"
    print(row)

    # Fragility ratio: how much more/less sensitive each gait is vs NC baseline.
    # Ratio > 1 means the gait is more fragile (small weight changes cause larger DX shifts).
    nc_total = sum(sensitivity["Novelty Champion"][k]["abs_gradient"] for k in SYNAPSE_KEYS)
    print(f"\n  Fragility ratio (total sensitivity / NC total):")
    for name in GAIT_ORDER[:3]:
        total = sum(sensitivity[name][k]["abs_gradient"] for k in SYNAPSE_KEYS)
        ratio = total / nc_total if nc_total > 1e-12 else 0
        print(f"    {name}: {ratio:.2f}x")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_fig01_trajectory(all_data):
    """Generate Fig 1: 3-panel trajectory comparison (XY path, X vs time, Z bounce).

    Args:
        all_data: Dict mapping gait name to telemetry data dict.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # XY trajectory
    ax = axes[0]
    for name in GAIT_ORDER:
        d = all_data[name]
        ax.plot(d["x"], d["y"], color=GAIT_COLORS[name], lw=0.6, alpha=0.8,
                label=GAIT_SHORT[name])
        ax.scatter([d["x"][-1]], [d["y"][-1]], c=GAIT_COLORS[name],
                   s=60, zorder=5, marker="*")
    ax.scatter([0], [0], c="black", s=50, zorder=5, marker="o")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    clean_ax(ax)

    # X vs time
    ax = axes[1]
    for name in GAIT_ORDER:
        d = all_data[name]
        ax.plot(d["t"], d["x"], color=GAIT_COLORS[name], lw=0.8,
                label=GAIT_SHORT[name])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("X position (m)")
    ax.set_title("X vs Time")
    ax.legend(fontsize=8)
    clean_ax(ax)

    # Z bounce
    ax = axes[2]
    for name in GAIT_ORDER:
        d = all_data[name]
        ax.plot(d["t"], d["z"], color=GAIT_COLORS[name], lw=0.4, alpha=0.7,
                label=GAIT_SHORT[name])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Z height (m)")
    ax.set_title("Vertical Bounce")
    ax.legend(fontsize=8)
    clean_ax(ax)

    fig.suptitle("Super-Gaits vs Novelty Champion: Trajectory Comparison", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "super_fig01_trajectory.png")


def plot_fig02_joints(all_data):
    """Generate Fig 2: 2x4 grid of joint positions (top row) and velocities (bottom row).

    Args:
        all_data: Dict mapping gait name to telemetry data dict.
    """
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))

    for col, name in enumerate(GAIT_ORDER):
        d = all_data[name]
        t = d["t"]
        color = GAIT_COLORS[name]

        ax = axes[0][col]
        ax.plot(t, d["j0_pos"], color="#DD8452", lw=0.5, label="BackLeg (j0)")
        ax.plot(t, d["j1_pos"], color="#8172B2", lw=0.5, label="FrontLeg (j1)")
        ax.set_title(f"{GAIT_SHORT[name]}", fontsize=10, color=color)
        ax.set_ylabel("Angle (rad)" if col == 0 else "")
        if col == 0:
            ax.legend(fontsize=6)
        clean_ax(ax)

        ax = axes[1][col]
        ax.plot(t, d["j0_vel"], color="#DD8452", lw=0.3, alpha=0.7, label="j0 vel")
        ax.plot(t, d["j1_vel"], color="#8172B2", lw=0.3, alpha=0.7, label="j1 vel")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (rad/s)" if col == 0 else "")
        if col == 0:
            ax.legend(fontsize=6)
        clean_ax(ax)

    fig.suptitle("Joint Dynamics: Positions (top) and Velocities (bottom)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "super_fig02_joints.png")


def plot_fig03_phase(all_data, all_metrics):
    """Generate Fig 3: 4-panel phase portraits (j0 vs j1, colored by time).

    Args:
        all_data: Dict mapping gait name to telemetry data dict.
        all_metrics: Dict mapping gait name to computed metrics (for DX/PL labels).
    """
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))

    for col, name in enumerate(GAIT_ORDER):
        ax = axes[col]
        d = all_data[name]
        color = GAIT_COLORS[name]
        n = len(d["j0_pos"])
        # Downsample to 500 evenly-spaced points for scatter overlay (avoids overplotting)
        scatter_idx = np.linspace(0, n - 1, 500, dtype=int)

        ax.plot(d["j0_pos"], d["j1_pos"], color=color, lw=0.3, alpha=0.3)
        sc = ax.scatter(d["j0_pos"][scatter_idx], d["j1_pos"][scatter_idx],
                        c=scatter_idx, cmap="viridis", s=5, alpha=0.7, zorder=3)
        ax.scatter([d["j0_pos"][0]], [d["j1_pos"][0]], c="green", s=40,
                   marker="^", zorder=5)

        m = all_metrics[name]
        ax.set_title(f"{GAIT_SHORT[name]}\nDX={m['dx']:.1f} PL={m['phase_lock']:.3f}",
                     fontsize=9, color=color)
        ax.set_xlabel("BackLeg (rad)")
        if col == 0:
            ax.set_ylabel("FrontLeg (rad)")
        clean_ax(ax)

    fig.suptitle("Phase Portraits (j0 vs j1, colored by time)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "super_fig03_phase.png")


def plot_fig04_energy(all_data):
    """Generate Fig 4: 2x4 grid of instantaneous power (top) and cumulative work (bottom).

    Args:
        all_data: Dict mapping gait name to telemetry data dict.
    """
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))

    for col, name in enumerate(GAIT_ORDER):
        d = all_data[name]
        t = d["t"]
        color = GAIT_COLORS[name]
        wj0, wj1, pj0, pj1 = work_by_joint(d)

        ax = axes[0][col]
        ax.plot(t, pj0, color="#DD8452", lw=0.3, alpha=0.6, label="j0 (BackLeg)")
        ax.plot(t, pj1, color="#8172B2", lw=0.3, alpha=0.6, label="j1 (FrontLeg)")
        ax.set_title(f"{GAIT_SHORT[name]}", fontsize=10, color=color)
        ax.set_ylabel("|Power|" if col == 0 else "")
        if col == 0:
            ax.legend(fontsize=6)
        clean_ax(ax)

        ax = axes[1][col]
        # Running integral of instantaneous power -> cumulative work over time
        cum_j0 = np.cumsum(pj0) * DT
        cum_j1 = np.cumsum(pj1) * DT
        ax.plot(t, cum_j0, color="#DD8452", lw=1.5, label=f"j0 ({wj0:.0f})")
        ax.plot(t, cum_j1, color="#8172B2", lw=1.5, label=f"j1 ({wj1:.0f})")
        ax.plot(t, cum_j0 + cum_j1, color=color, lw=2, ls="--",
                label=f"Total ({wj0+wj1:.0f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative work" if col == 0 else "")
        ax.legend(fontsize=6)
        clean_ax(ax)

    fig.suptitle("Energy: Instantaneous Power (top) and Cumulative Work (bottom)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "super_fig04_energy.png")


def plot_fig05_contacts(all_data):
    """Generate Fig 5: 4-panel stacked contact rasters (torso/back/front over time).

    Args:
        all_data: Dict mapping gait name to telemetry data dict.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    for ax, name in zip(axes, GAIT_ORDER):
        d = all_data[name]
        t_plot = d["t"]
        color = GAIT_COLORS[name]

        ct_v = d["contact_torso"].astype(float)
        cb_v = d["contact_back"].astype(float)
        cf_v = d["contact_front"].astype(float)

        # Stack contact rasters vertically: torso at y=0, back at y=1, front at y=2
        ax.fill_between(t_plot, 0, ct_v * 0.9, alpha=0.5, color="#999999", label="Torso")
        ax.fill_between(t_plot, 1, 1 + cb_v * 0.9, alpha=0.5, color="#DD8452", label="BackLeg")
        ax.fill_between(t_plot, 2, 2 + cf_v * 0.9, alpha=0.5, color="#8172B2", label="FrontLeg")
        ax.set_yticks([0.45, 1.45, 2.45])
        ax.set_yticklabels(["Torso", "Back", "Front"])
        ax.set_title(f"{name}", fontsize=10, color=color)
        if ax == axes[0]:
            ax.legend(fontsize=7, loc="upper right")
        clean_ax(ax)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Contact Patterns: Foot Rasters", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "super_fig05_contacts.png")


def plot_fig06_sensitivity(sensitivity):
    """Generate Fig 6: grouped bar chart of local |dDX/dw| for each synapse.

    Shows 3 viable gaits (NC, InterpS, w23Half) side-by-side for each of the
    6 synapses.

    Args:
        sensitivity: Dict mapping gait name to per-synapse sensitivity results.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    names_with_sens = GAIT_ORDER[:3]  # NC, InterpS, w23Half
    n_gaits = len(names_with_sens)
    n_syn = len(SYNAPSE_KEYS)
    bar_width = 0.25
    x_pos = np.arange(n_syn)

    for gi, name in enumerate(names_with_sens):
        vals = [sensitivity[name][k]["abs_gradient"] for k in SYNAPSE_KEYS]
        # Center the grouped bars around each x tick position
        offset = (gi - (n_gaits - 1) / 2) * bar_width
        ax.bar(x_pos + offset, vals, bar_width, color=GAIT_COLORS[name],
               alpha=0.85, label=GAIT_SHORT[name])

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SYNAPSE_LABELS[k] for k in SYNAPSE_KEYS], rotation=30, ha="right")
    ax.set_ylabel("|dDX/dw| (m per unit weight)")
    ax.set_title("Local Sensitivity: How Fragile Is Each Synapse?", fontsize=13)
    ax.legend(fontsize=9)
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "super_fig06_sensitivity.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run the full analysis pipeline: simulate 4 gaits, compute sensitivity,
    print tables, save JSON artifacts, and generate all 6 comparison figures.
    """
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    t_start = time.perf_counter()
    sim_count = 0

    # ── Part 1: Simulate 4 gaits ─────────────────────────────────────────────
    print("=" * 60)
    print("PART 1: SIMULATING 4 GAITS")
    print("=" * 60)

    all_data = {}
    all_metrics = {}

    for name in GAIT_ORDER:
        print(f"  Simulating {name}...")
        data = simulate_full(GAITS[name])
        metrics = compute_metrics(data)
        all_data[name] = data
        all_metrics[name] = metrics
        sim_count += 1
        print(f"    DX={metrics['dx']:.2f}m  Work={metrics['work_proxy']:.0f}"
              f"  Eff={metrics['distance_per_work']:.4f}")

    t_gaits = time.perf_counter()
    print(f"\n4 gaits simulated in {t_gaits - t_start:.1f}s")

    # ── Part 2: Console tables ───────────────────────────────────────────────
    print_metrics_table(all_metrics)
    print_weight_analysis(all_metrics)

    # ── Part 3: Sensitivity probes ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PART 2: LOCAL SENSITIVITY PROBES (±1%)")
    print(f"{'='*60}")

    sensitivity = {}
    perturbation = 0.01  # 1%

    for name in GAIT_ORDER[:3]:  # NC, InterpS, w23Half
        print(f"  Probing {name}...")
        base_dx = all_metrics[name]["dx"]
        w_base = GAITS[name]
        sens = {}

        for k in SYNAPSE_KEYS:
            w_val = w_base[k]
            # Scale perturbation relative to weight magnitude; use a small
            # fixed epsilon for near-zero weights to avoid division issues
            eps = abs(w_val) * perturbation if abs(w_val) > 1e-6 else perturbation * 0.01

            # +perturbation
            w_plus = dict(w_base)
            w_plus[k] = w_val + eps
            d_plus = simulate_full(w_plus)
            m_plus = compute_metrics(d_plus)
            sim_count += 1

            # -perturbation
            w_minus = dict(w_base)
            w_minus[k] = w_val - eps
            d_minus = simulate_full(w_minus)
            m_minus = compute_metrics(d_minus)
            sim_count += 1

            dx_plus = m_plus["dx"]
            dx_minus = m_minus["dx"]

            # Central-difference approximation: dDX/dw ≈ (f(w+eps) - f(w-eps)) / (2*eps)
            gradient = (dx_plus - dx_minus) / (2 * eps) if eps > 1e-12 else 0.0

            sens[k] = {
                "weight": w_val,
                "eps": eps,
                "dx_plus": dx_plus,
                "dx_minus": dx_minus,
                "gradient": gradient,
                "abs_gradient": abs(gradient),
                "delta_dx_plus": dx_plus - base_dx,
                "delta_dx_minus": dx_minus - base_dx,
            }

        sensitivity[name] = sens

    t_sens = time.perf_counter()
    print(f"\nSensitivity probes: {t_sens - t_gaits:.1f}s ({sim_count - 4} sims)")

    print_sensitivity_table(sensitivity, all_metrics)

    # ── Restore brain.nndf ───────────────────────────────────────────────────
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # ── Save JSON ────────────────────────────────────────────────────────────
    print("\nSaving JSON...")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_gait_sims": 4,
            "n_sensitivity_sims": sim_count - 4,
            "total_sims": sim_count,
            "total_time_s": time.perf_counter() - t_start,
        },
        "gaits": {},
        "sensitivity": {},
        "weight_analysis": {},
    }

    for name in GAIT_ORDER:
        w = GAITS[name]
        wj0, wj1, _, _ = work_by_joint(all_data[name])
        output["gaits"][name] = {
            "weights": w,
            "metrics": all_metrics[name],
            "work_j0": wj0,
            "work_j1": wj1,
        }
        # Aggregate input drive per motor neuron and total network magnitude
        tm3 = w["w03"] + w["w13"] + w["w23"]
        tm4 = w["w04"] + w["w14"] + w["w24"]
        mag = sum(abs(w[k]) for k in SYNAPSE_KEYS)
        output["weight_analysis"][name] = {
            "total_to_m3": tm3,
            "total_to_m4": tm4,
            "total_magnitude": mag,
            "per_sensor": {
                f"sensor_{s}": {
                    "to_m3": w[f"w{s}3"],
                    "to_m4": w[f"w{s}4"],
                    "sum": w[f"w{s}3"] + w[f"w{s}4"],
                    "diff": w[f"w{s}3"] - w[f"w{s}4"],
                } for s in [0, 1, 2]
            },
        }

    for name in GAIT_ORDER[:3]:
        output["sensitivity"][name] = sensitivity[name]

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"WROTE {OUT_JSON}")

    # ── Generate figures ─────────────────────────────────────────────────────
    print("\nGenerating figures...")

    plot_fig01_trajectory(all_data)
    plot_fig02_joints(all_data)
    plot_fig03_phase(all_data, all_metrics)
    plot_fig04_energy(all_data)
    plot_fig05_contacts(all_data)
    plot_fig06_sensitivity(sensitivity)

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_time:.1f}s")
    print(f"  Gait sims:        4")
    print(f"  Sensitivity sims: {sim_count - 4}")
    print(f"  Total sims:       {sim_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
