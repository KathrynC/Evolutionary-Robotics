#!/usr/bin/env python3
"""
analyze_trial3.py

Role:
    Deep analysis of random search Trial 3 ("The Accidental Masterpiece")
    vs the CPG Champion (gait 43). Side-by-side comparison of trajectories,
    joint dynamics, contact patterns, energy expenditure, phase relationships,
    angular velocity, and FFT spectra.

Pipeline:
    1. Simulate Trial 3 with full in-memory telemetry (motor neuron outputs included).
    2. Load CPG Champion from saved telemetry (different topology).
    3. Compute Beer-framework analytics for both.
    4. Print an 18-metric comparison table with ratios.
    5. Analyze Trial 3's weight structure and energy breakdown.
    6. Generate 7 publication-quality figures.

Outputs:
    artifacts/plots/trial3_fig01_trajectory.png  -- XY path, X vs time, Z bounce
    artifacts/plots/trial3_fig02_joints.png      -- Joint positions, motor outputs, velocities
    artifacts/plots/trial3_fig03_contacts.png    -- Contact rasters (2 gaits stacked)
    artifacts/plots/trial3_fig04_energy.png      -- Power and cumulative work (2x2)
    artifacts/plots/trial3_fig05_phase.png       -- Phase portrait + Hilbert phase difference
    artifacts/plots/trial3_fig06_rotation.png    -- Angular velocity components (2x3)
    artifacts/plots/trial3_fig07_fft.png         -- FFT spectra of joint angles

Notes:
    - brain.nndf is backed up before simulation and restored afterward.
    - CPG Champion motor neuron outputs are not available in saved telemetry,
      so motor-specific plots are only shown for Trial 3.
    - The Hilbert-transform phase difference is computed via numpy FFT
      (no scipy dependency).
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
from compute_beer_analytics import (compute_all, compute_outcome, compute_contact,
                                     compute_coordination, compute_rotation_axis,
                                     load_telemetry, DT, _fft_peak, _hilbert_analytic)

PLOT_DIR = PROJECT / "artifacts" / "plots"
TELEMETRY_DIR = PROJECT / "artifacts" / "telemetry"

TRIAL3_WEIGHTS = {
    "w03": -0.5971393487736976,
    "w04": -0.4236677331634211,
    "w13": 0.11222931078528431,
    "w14": -0.004679977731207874,
    "w23": 0.2970146930268889,
    "w24": 0.21399448704946855,
}


def write_brain_6syn(weights):
    """Write a 6-synapse brain.nndf file (3 sensors -> 2 motors) from a weight dict.

    Args:
        weights: Dict with keys "w03","w13","w23","w04","w14","w24" mapping to
            float synapse weights.

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
        # Write all 6 sensor-to-motor synapses (3 sensors x 2 motors)
        for s in [0, 1, 2]:
            for m in [3, 4]:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def run_with_telemetry(name, write_brain_fn, out_dir):
    """Run a headless simulation with full in-memory telemetry collection.

    Captures position, velocity, orientation, ground contacts, joint states,
    and motor neuron outputs at every timestep.

    Args:
        name: Identifier string for this run (used for logging/output naming).
        write_brain_fn: Callable that writes brain.nndf (called before simulation).
        out_dir: Output directory path (reserved for future telemetry file export).

    Returns:
        Dict of numpy arrays keyed by signal name (t, x, y, z, vx, vy, vz,
        wx, wy, wz, roll, pitch, yaw, contact_torso, contact_back, contact_front,
        j0_pos, j0_vel, j0_tau, j1_pos, j1_vel, j1_tau, m3_out, m4_out).

    Side effects:
        Calls write_brain_fn() which overwrites brain.nndf.
    """
    write_brain_fn()

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(c.DT)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    # Apply uniform friction to all links (including base link at index -1)
    mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
    for link in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, link, lateralFriction=mu, restitution=0.0)

    nn = NEURAL_NETWORK("brain.nndf")
    max_force = float(getattr(c, "MAX_FORCE", 150.0))
    n_steps = c.SIM_STEPS

    # Pre-allocate
    t_arr = np.empty(n_steps)
    x = np.empty(n_steps); y = np.empty(n_steps); z = np.empty(n_steps)
    vx = np.empty(n_steps); vy = np.empty(n_steps); vz = np.empty(n_steps)
    wx = np.empty(n_steps); wy = np.empty(n_steps); wz = np.empty(n_steps)
    roll = np.empty(n_steps); pitch = np.empty(n_steps); yaw = np.empty(n_steps)
    ct = np.empty(n_steps, dtype=bool)
    cb = np.empty(n_steps, dtype=bool)
    cf = np.empty(n_steps, dtype=bool)
    j0p = np.empty(n_steps); j0v = np.empty(n_steps); j0t = np.empty(n_steps)
    j1p = np.empty(n_steps); j1v = np.empty(n_steps); j1t = np.empty(n_steps)

    # Motor neuron outputs
    m3_out = np.empty(n_steps)
    m4_out = np.empty(n_steps)

    # Build lookup dicts mapping link/joint names to PyBullet indices
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
        # Capture motor neuron values before acting
        for nName in nn.neurons:
            n_obj = nn.neurons[nName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_str = jn.decode("utf-8") if isinstance(jn, bytes) else jn
                val = n_obj.Get_Value()
                if "BackLeg" in jn_str:
                    m3_out[i] = val
                else:
                    m4_out[i] = val
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, val, max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, p.POSITION_CONTROL,
                                                val, max_force)

        p.stepSimulation()
        nn.Update()

        t_arr[i] = i * c.DT
        pos, orn = p.getBasePositionAndOrientation(robotId)
        vel_lin, vel_ang = p.getBaseVelocity(robotId)
        rpy_vals = p.getEulerFromQuaternion(orn)

        x[i] = pos[0]; y[i] = pos[1]; z[i] = pos[2]
        vx[i] = vel_lin[0]; vy[i] = vel_lin[1]; vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]; wy[i] = vel_ang[1]; wz[i] = vel_ang[2]
        roll[i] = rpy_vals[0]; pitch[i] = rpy_vals[1]; yaw[i] = rpy_vals[2]

        # Classify ground contacts by which link is touching (cp[3] = linkIndexA)
        contact_pts = p.getContactPoints(robotId)
        tc = bc = fc = False
        for cp in contact_pts:
            li = cp[3]
            if li == -1: tc = True      # base link (Torso)
            elif li == back_li: bc = True
            elif li == front_li: fc = True
        ct[i] = tc; cb[i] = bc; cf[i] = fc

        js0 = p.getJointState(robotId, j0_idx)
        js1 = p.getJointState(robotId, j1_idx)
        j0p[i] = js0[0]; j0v[i] = js0[1]; j0t[i] = js0[3]
        j1p[i] = js1[0]; j1v[i] = js1[1]; j1t[i] = js1[3]

    p.disconnect()

    data = {
        "t": t_arr, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "contact_torso": ct, "contact_back": cb, "contact_front": cf,
        "j0_pos": j0p, "j0_vel": j0v, "j0_tau": j0t,
        "j1_pos": j1p, "j1_vel": j1v, "j1_tau": j1t,
        "m3_out": m3_out, "m4_out": m4_out,
    }
    return data


def clean_ax(ax):
    """Remove top and right spines from an axis for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_fig(fig, name):
    """Save a matplotlib figure to PLOT_DIR and close it to free memory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


def main():
    """Run Trial 3 vs CPG Champion comparison: simulate, compute analytics, and plot.

    Side effects:
        - Backs up and restores brain.nndf.
        - Runs 1 headless simulation (Trial 3).
        - Loads CPG Champion from saved telemetry.
        - Writes 7 PNG figures to artifacts/plots/.
        - Prints detailed comparison table, weight analysis, and energy breakdown
          to stdout.
    """
    # Backup brain.nndf since write_brain_6syn will overwrite it
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    # ── Run Trial 3 ──────────────────────────────────────────────────────────
    print("Running Trial 3 (Accidental Masterpiece)...")
    t3_data = run_with_telemetry(
        "trial3_masterpiece",
        lambda: write_brain_6syn(TRIAL3_WEIGHTS),
        TELEMETRY_DIR / "trial3_masterpiece",
    )
    t3_analytics = compute_all(t3_data, DT)

    # ── Load CPG Champion telemetry ──────────────────────────────────────────
    print("Loading CPG Champion (gait 43) telemetry...")
    cpg_data = load_telemetry("43_hidden_cpg_champion")
    cpg_analytics = compute_all(cpg_data, DT)
    # CPG doesn't have motor neuron output in telemetry, so we skip those plots for it

    # Restore brain.nndf
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    t3 = t3_data
    cpg = cpg_data
    t_sec = t3["t"]
    t_sec_cpg = cpg["t"]

    # ── Print comparison ─────────────────────────────────────────────────────
    t3a = t3_analytics
    cpga = cpg_analytics

    print("\n" + "=" * 80)
    print("TRIAL 3 vs CPG CHAMPION — DETAILED COMPARISON")
    print("=" * 80)

    # Build side-by-side metric table: (label, trial3_value, cpg_value)
    metrics = [
        ("DX (m)", t3a["outcome"]["dx"], cpga["outcome"]["dx"]),
        ("DY (m)", t3a["outcome"]["dy"], cpga["outcome"]["dy"]),
        ("Net distance (m)", np.sqrt(t3a["outcome"]["dx"]**2 + t3a["outcome"]["dy"]**2),
                             np.sqrt(cpga["outcome"]["dx"]**2 + cpga["outcome"]["dy"]**2)),
        ("Mean speed (m/s)", t3a["outcome"]["mean_speed"], cpga["outcome"]["mean_speed"]),
        ("Speed CV", t3a["outcome"]["speed_cv"], cpga["outcome"]["speed_cv"]),
        ("Work proxy", t3a["outcome"]["work_proxy"], cpga["outcome"]["work_proxy"]),
        ("Efficiency (dist/work)", t3a["outcome"]["distance_per_work"],
                                    cpga["outcome"]["distance_per_work"]),
        ("Yaw (rad)", t3a["outcome"]["yaw_net_rad"], cpga["outcome"]["yaw_net_rad"]),
        ("Phase lock", t3a["coordination"]["phase_lock_score"],
                       cpga["coordination"]["phase_lock_score"]),
        ("Delta phi (rad)", t3a["coordination"]["delta_phi_rad"],
                            cpga["coordination"]["delta_phi_rad"]),
        ("Joint 0 freq (Hz)", t3a["coordination"]["joint_0"]["dominant_freq_hz"],
                               cpga["coordination"]["joint_0"]["dominant_freq_hz"]),
        ("Joint 1 freq (Hz)", t3a["coordination"]["joint_1"]["dominant_freq_hz"],
                               cpga["coordination"]["joint_1"]["dominant_freq_hz"]),
        ("Contact entropy (bits)", t3a["contact"]["contact_entropy_bits"],
                                    cpga["contact"]["contact_entropy_bits"]),
        ("Duty torso", t3a["contact"]["duty_torso"], cpga["contact"]["duty_torso"]),
        ("Duty back leg", t3a["contact"]["duty_back"], cpga["contact"]["duty_back"]),
        ("Duty front leg", t3a["contact"]["duty_front"], cpga["contact"]["duty_front"]),
        ("Roll dominance", t3a["rotation_axis"]["axis_dominance"][0],
                            cpga["rotation_axis"]["axis_dominance"][0]),
        ("Axis switching (Hz)", t3a["rotation_axis"]["axis_switching_rate_hz"],
                                 cpga["rotation_axis"]["axis_switching_rate_hz"]),
    ]

    print(f"\n  {'Metric':<28} {'Trial 3':>12} {'CPG Champ':>12} {'Ratio':>10}")
    print("  " + "-" * 65)
    for label, v3, vc in metrics:
        if vc != 0:
            ratio = v3 / vc
            print(f"  {label:<28} {v3:12.4f} {vc:12.4f} {ratio:10.2f}x")
        else:
            print(f"  {label:<28} {v3:12.4f} {vc:12.4f}       —")

    # ── Weight analysis ──────────────────────────────────────────────────────
    print("\n  WEIGHT STRUCTURE:")
    print(f"    Sensor 0 (Torso):    w03={TRIAL3_WEIGHTS['w03']:+.4f}  w04={TRIAL3_WEIGHTS['w04']:+.4f}  "
          f"sum={TRIAL3_WEIGHTS['w03']+TRIAL3_WEIGHTS['w04']:+.4f}  "
          f"diff={TRIAL3_WEIGHTS['w03']-TRIAL3_WEIGHTS['w04']:+.4f}")
    print(f"    Sensor 1 (BackLeg):  w13={TRIAL3_WEIGHTS['w13']:+.4f}  w14={TRIAL3_WEIGHTS['w14']:+.4f}  "
          f"sum={TRIAL3_WEIGHTS['w13']+TRIAL3_WEIGHTS['w14']:+.4f}  "
          f"diff={TRIAL3_WEIGHTS['w13']-TRIAL3_WEIGHTS['w14']:+.4f}")
    print(f"    Sensor 2 (FrontLeg): w23={TRIAL3_WEIGHTS['w23']:+.4f}  w24={TRIAL3_WEIGHTS['w24']:+.4f}  "
          f"sum={TRIAL3_WEIGHTS['w23']+TRIAL3_WEIGHTS['w24']:+.4f}  "
          f"diff={TRIAL3_WEIGHTS['w23']-TRIAL3_WEIGHTS['w24']:+.4f}")

    # Sum of all incoming weights to each motor neuron (net excitatory/inhibitory drive)
    total_to_m3 = TRIAL3_WEIGHTS['w03'] + TRIAL3_WEIGHTS['w13'] + TRIAL3_WEIGHTS['w23']
    total_to_m4 = TRIAL3_WEIGHTS['w04'] + TRIAL3_WEIGHTS['w14'] + TRIAL3_WEIGHTS['w24']
    print(f"\n    Total drive to BackLeg motor (m3): {total_to_m3:+.4f}")
    print(f"    Total drive to FrontLeg motor (m4): {total_to_m4:+.4f}")
    print(f"    Ratio m3/m4: {total_to_m3/total_to_m4:.2f}")

    # ── Energy breakdown ─────────────────────────────────────────────────────
    # Instantaneous power = |torque * angular_velocity|; work = integral of power over time
    t3_power_j0 = np.abs(t3["j0_tau"] * t3["j0_vel"])
    t3_power_j1 = np.abs(t3["j1_tau"] * t3["j1_vel"])
    t3_work_j0 = float(np.sum(t3_power_j0) * DT)
    t3_work_j1 = float(np.sum(t3_power_j1) * DT)

    cpg_power_j0 = np.abs(cpg["j0_tau"] * cpg["j0_vel"])
    cpg_power_j1 = np.abs(cpg["j1_tau"] * cpg["j1_vel"])
    cpg_work_j0 = float(np.sum(cpg_power_j0) * DT)
    cpg_work_j1 = float(np.sum(cpg_power_j1) * DT)

    print(f"\n  ENERGY BREAKDOWN:")
    print(f"    {'Joint':<20} {'Trial 3':>10} {'CPG Champ':>10} {'Ratio':>8}")
    print(f"    {'BackLeg (j0)':<20} {t3_work_j0:10.1f} {cpg_work_j0:10.1f} {t3_work_j0/cpg_work_j0:8.2f}x")
    print(f"    {'FrontLeg (j1)':<20} {t3_work_j1:10.1f} {cpg_work_j1:10.1f} {t3_work_j1/cpg_work_j1:8.2f}x")
    print(f"    {'Total':<20} {t3_work_j0+t3_work_j1:10.1f} {cpg_work_j0+cpg_work_j1:10.1f} "
          f"{(t3_work_j0+t3_work_j1)/(cpg_work_j0+cpg_work_j1):8.2f}x")

    # ── FIGURES ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    C3 = "#C44E52"  # Trial 3 color
    CCPG = "#4C72B0"  # CPG color

    # Fig 1: XY trajectory
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(t3["x"], t3["y"], color=C3, lw=0.5, alpha=0.8, label="Trial 3")
    ax.plot(cpg["x"], cpg["y"], color=CCPG, lw=0.5, alpha=0.8, label="CPG Champion")
    ax.scatter([t3["x"][0]], [t3["y"][0]], c="black", s=40, zorder=5, marker="o")
    ax.scatter([t3["x"][-1]], [t3["y"][-1]], c=C3, s=60, zorder=5, marker="*")
    ax.scatter([cpg["x"][-1]], [cpg["y"][-1]], c=CCPG, s=60, zorder=5, marker="*")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.legend(fontsize=9); ax.set_aspect("equal")
    clean_ax(ax)

    ax = axes[1]
    ax.plot(t_sec, t3["x"], color=C3, lw=0.8, label="Trial 3")
    ax.plot(t_sec_cpg, cpg["x"], color=CCPG, lw=0.8, label="CPG Champion")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("X position (m)")
    ax.set_title("X vs Time")
    ax.legend(fontsize=9)
    clean_ax(ax)

    ax = axes[2]
    ax.plot(t_sec, t3["z"], color=C3, lw=0.5, alpha=0.8, label="Trial 3")
    ax.plot(t_sec_cpg, cpg["z"], color=CCPG, lw=0.5, alpha=0.8, label="CPG Champion")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Z height (m)")
    ax.set_title("Vertical Bounce")
    ax.legend(fontsize=9)
    clean_ax(ax)

    fig.suptitle("Trial 3 vs CPG Champion: Trajectories", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "trial3_fig01_trajectory.png")

    # Fig 2: Joint positions and motor neuron outputs
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Trial 3 joints
    ax = axes[0][0]
    ax.plot(t_sec, t3["j0_pos"], color="#55A868", lw=0.5, label="BackLeg (j0)")
    ax.plot(t_sec, t3["j1_pos"], color="#DD8452", lw=0.5, label="FrontLeg (j1)")
    ax.set_ylabel("Joint angle (rad)")
    ax.set_title("Trial 3 — Joint Positions")
    ax.legend(fontsize=8); clean_ax(ax)

    # CPG joints
    ax = axes[0][1]
    ax.plot(t_sec_cpg, cpg["j0_pos"], color="#55A868", lw=0.5, label="BackLeg (j0)")
    ax.plot(t_sec_cpg, cpg["j1_pos"], color="#DD8452", lw=0.5, label="FrontLeg (j1)")
    ax.set_ylabel("Joint angle (rad)")
    ax.set_title("CPG Champion — Joint Positions")
    ax.legend(fontsize=8); clean_ax(ax)

    # Trial 3 motor neuron outputs
    ax = axes[1][0]
    ax.plot(t_sec, t3["m3_out"], color="#55A868", lw=0.5, label="Motor 3 (BackLeg)")
    ax.plot(t_sec, t3["m4_out"], color="#DD8452", lw=0.5, label="Motor 4 (FrontLeg)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Motor neuron output")
    ax.set_title("Trial 3 — Motor Neuron Commands")
    ax.legend(fontsize=8); clean_ax(ax)

    # Trial 3 joint velocities
    ax = axes[1][1]
    ax.plot(t_sec, t3["j0_vel"], color="#55A868", lw=0.3, alpha=0.7, label="j0 vel")
    ax.plot(t_sec, t3["j1_vel"], color="#DD8452", lw=0.3, alpha=0.7, label="j1 vel")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Joint velocity (rad/s)")
    ax.set_title("Trial 3 — Joint Velocities")
    ax.legend(fontsize=8); clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "trial3_fig02_joints.png")

    # Fig 3: Contact patterns
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for ax, data, label, color in [(axes[0], t3, "Trial 3", C3),
                                     (axes[1], cpg, "CPG Champion", CCPG)]:
        t_plot = data["t"]
        ct_val = data["contact_torso"].astype(float)
        cb_val = data["contact_back"].astype(float)
        cf_val = data["contact_front"].astype(float)
        # Stacked visualization: each link gets its own y-band (0-0.9, 1-1.9, 2-2.9)
        # so contact events appear as filled bars in separate horizontal lanes
        ax.fill_between(t_plot, 0, ct_val * 0.9, alpha=0.5, color="#999999", label="Torso")
        ax.fill_between(t_plot, 1, 1 + cb_val * 0.9, alpha=0.5, color="#55A868", label="BackLeg")
        ax.fill_between(t_plot, 2, 2 + cf_val * 0.9, alpha=0.5, color="#DD8452", label="FrontLeg")
        ax.set_yticks([0.45, 1.45, 2.45])
        ax.set_yticklabels(["Torso", "BackLeg", "FrontLeg"])
        ax.set_title(f"{label} — Contact Pattern")
        ax.legend(fontsize=8, loc="upper right")
        clean_ax(ax)

    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, "trial3_fig03_contacts.png")

    # Fig 4: Energy — instantaneous power and cumulative work
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Instantaneous power
    ax = axes[0][0]
    ax.plot(t_sec, t3_power_j0, color="#55A868", lw=0.3, alpha=0.7, label="j0 (BackLeg)")
    ax.plot(t_sec, t3_power_j1, color="#DD8452", lw=0.3, alpha=0.7, label="j1 (FrontLeg)")
    ax.set_ylabel("|Power| (N*rad/s)")
    ax.set_title("Trial 3 — Instantaneous Power")
    ax.legend(fontsize=8); clean_ax(ax)

    ax = axes[0][1]
    ax.plot(t_sec_cpg, cpg_power_j0, color="#55A868", lw=0.3, alpha=0.7, label="j0 (BackLeg)")
    ax.plot(t_sec_cpg, cpg_power_j1, color="#DD8452", lw=0.3, alpha=0.7, label="j1 (FrontLeg)")
    ax.set_ylabel("|Power| (N*rad/s)")
    ax.set_title("CPG Champion — Instantaneous Power")
    ax.legend(fontsize=8); clean_ax(ax)

    # Cumulative work
    ax = axes[1][0]
    t3_cum_j0 = np.cumsum(t3_power_j0) * DT
    t3_cum_j1 = np.cumsum(t3_power_j1) * DT
    ax.plot(t_sec, t3_cum_j0, color="#55A868", lw=1.5, label=f"j0 ({t3_work_j0:.0f})")
    ax.plot(t_sec, t3_cum_j1, color="#DD8452", lw=1.5, label=f"j1 ({t3_work_j1:.0f})")
    ax.plot(t_sec, t3_cum_j0 + t3_cum_j1, color=C3, lw=2, ls="--",
            label=f"Total ({t3_work_j0+t3_work_j1:.0f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cumulative work")
    ax.set_title("Trial 3 — Cumulative Work")
    ax.legend(fontsize=8); clean_ax(ax)

    ax = axes[1][1]
    cpg_cum_j0 = np.cumsum(cpg_power_j0) * DT
    cpg_cum_j1 = np.cumsum(cpg_power_j1) * DT
    ax.plot(t_sec_cpg, cpg_cum_j0, color="#55A868", lw=1.5, label=f"j0 ({cpg_work_j0:.0f})")
    ax.plot(t_sec_cpg, cpg_cum_j1, color="#DD8452", lw=1.5, label=f"j1 ({cpg_work_j1:.0f})")
    ax.plot(t_sec_cpg, cpg_cum_j0 + cpg_cum_j1, color=CCPG, lw=2, ls="--",
            label=f"Total ({cpg_work_j0+cpg_work_j1:.0f})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cumulative work")
    ax.set_title("CPG Champion — Cumulative Work")
    ax.legend(fontsize=8); clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "trial3_fig04_energy.png")

    # Fig 5: Phase relationship
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Phase portrait: j0 vs j1
    ax = axes[0]
    ax.plot(t3["j0_pos"], t3["j1_pos"], color=C3, lw=0.3, alpha=0.6, label="Trial 3")
    ax.plot(cpg["j0_pos"], cpg["j1_pos"], color=CCPG, lw=0.3, alpha=0.6, label="CPG Champion")
    ax.set_xlabel("BackLeg angle (rad)")
    ax.set_ylabel("FrontLeg angle (rad)")
    ax.set_title("Phase Portrait (j0 vs j1)")
    ax.legend(fontsize=9)
    clean_ax(ax)

    # Instantaneous phase difference via Hilbert analytic signal
    for ax, data, label, color in [(axes[1], t3, "Trial 3", C3),
                                     (axes[2], cpg, "CPG Champion", CCPG)]:
        # Compute analytic signals (complex-valued) from mean-centered joint angles
        a0 = _hilbert_analytic(data["j0_pos"] - np.mean(data["j0_pos"]))
        a1 = _hilbert_analytic(data["j1_pos"] - np.mean(data["j1_pos"]))
        # Wrap phase difference to [-pi, pi] using complex exponential trick
        dphi = np.angle(np.exp(1j * (np.angle(a1) - np.angle(a0))))
        ax.plot(data["t"], dphi, color=color, lw=0.3, alpha=0.7)
        ax.axhline(np.mean(dphi), color="black", lw=1, ls="--",
                   label=f"mean={np.mean(dphi):.2f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Phase difference (rad)")
        ax.set_title(f"{label} — Instantaneous Phase Diff")
        ax.set_ylim(-np.pi, np.pi)
        ax.legend(fontsize=9)
        clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "trial3_fig05_phase.png")

    # Fig 6: Angular velocity components
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    components = [("wx", "Roll"), ("wy", "Pitch"), ("wz", "Yaw")]
    for col, (key, label) in enumerate(components):
        ax = axes[0][col]
        ax.plot(t_sec, t3[key], color=C3, lw=0.3, alpha=0.7)
        ax.set_title(f"Trial 3 — {label} velocity")
        ax.set_ylabel(f"w_{label[0].lower()} (rad/s)")
        clean_ax(ax)

        ax = axes[1][col]
        ax.plot(t_sec_cpg, cpg[key], color=CCPG, lw=0.3, alpha=0.7)
        ax.set_title(f"CPG Champion — {label} velocity")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"w_{label[0].lower()} (rad/s)")
        clean_ax(ax)

    fig.suptitle("Angular Velocity Components", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "trial3_fig06_rotation.png")

    # Fig 7: FFT spectra comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, label, color in [(axes[0], t3, "Trial 3", C3),
                                     (axes[1], cpg, "CPG Champion", CCPG)]:
        for jkey, jlabel, jcolor in [("j0_pos", "BackLeg", "#55A868"),
                                       ("j1_pos", "FrontLeg", "#DD8452")]:
            # Mean-center signal before FFT to remove DC component
            sig = data[jkey] - np.mean(data[jkey])
            n = len(sig)
            freqs = np.fft.rfftfreq(n, d=DT)
            # Normalize to single-sided amplitude spectrum; zero DC bin explicitly
            spectrum = np.abs(np.fft.rfft(sig)) * (2.0 / n)
            spectrum[0] = 0
            # Plot only first 100 frequency bins (low-frequency behavior)
            ax.plot(freqs[:100], spectrum[:100], color=jcolor, lw=1.2, label=jlabel)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{label} — Joint Angle FFT")
        ax.legend(fontsize=9)
        clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "trial3_fig07_fft.png")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
