#!/usr/bin/env python3
"""
analyze_novelty_champion.py

Role:
    Deep analysis of the Novelty Seeker's champion gait (DX = +60.2m) --
    the record-holder that surpassed the CPG Champion (50.1m) using a standard
    6-synapse topology. Produces a 3-way comparison against the CPG Champion
    and Trial 3 ("The Accidental Masterpiece").

Pipeline:
    1. Simulate Novelty Champion and Trial 3 (write temporary brain.nndf for each).
    2. Load CPG Champion from saved telemetry (different topology, not re-simulated).
    3. Compute Beer-framework analytics for all three.
    4. Print a 19-metric comparison table with NC/CPG ratio.
    5. Analyze the Novelty Champion's weight structure (why w03 exceeds [-1,1]).
    6. Generate 7 publication-quality figures.

Outputs:
    artifacts/plots/champ_fig01_trajectory.png  -- XY path, X vs time, Z bounce
    artifacts/plots/champ_fig02_joints.png      -- Joint positions and velocities
    artifacts/plots/champ_fig03_contacts.png    -- Contact raster (3 stacked lanes)
    artifacts/plots/champ_fig04_energy.png      -- Instantaneous power + cumulative work
    artifacts/plots/champ_fig05_phase.png       -- Phase portraits (j0 vs j1)
    artifacts/plots/champ_fig06_rotation.png    -- Angular velocity components (3x3)
    artifacts/plots/champ_fig07_fft.png         -- FFT spectra of joint angles

Notes:
    - brain.nndf is backed up before simulation and restored afterward.
    - CPG Champion uses a hidden-layer topology (7 neurons), so its data comes
      from pre-recorded telemetry rather than live simulation.
    - The Novelty Champion's w03=-1.308 is outside [-1,1], having emerged from
      unclamped perturbation in the Novelty Seeker optimization.

Usage:
    python3 analyze_novelty_champion.py
"""

import json
import shutil
import sys
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
from compute_beer_analytics import (compute_all, load_telemetry, DT,
                                     _fft_peak, _hilbert_analytic)

PLOT_DIR = PROJECT / "artifacts" / "plots"

# The three gaits we're comparing
NOVELTY_CHAMPION_WEIGHTS = {
    "w03": -1.3083167156740476,
    "w04": -0.34279812804233867,
    "w13": 0.8331363773051514,
    "w14": -0.37582983217830773,
    "w23": -0.0369713954829298,
    "w24": 0.4375020967145814,
}

TRIAL3_WEIGHTS = {
    "w03": -0.5971393487736976,
    "w04": -0.4236677331634211,
    "w13": 0.11222931078528431,
    "w14": -0.004679977731207874,
    "w23": 0.2970146930268889,
    "w24": 0.21399448704946855,
}

GAIT_LABELS = {
    "nc": ("Novelty Champion", "#C44E52"),
    "cpg": ("CPG Champion", "#4C72B0"),
    "t3": ("Trial 3", "#55A868"),
}


# ── Simulation ───────────────────────────────────────────────────────────────

def write_brain_6syn(weights):
    """Write a 6-synapse brain.nndf file from a weights dict (3 sensors x 2 motors).

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
        # Full connectivity: each of 3 sensors connects to each of 2 motors
        for s in [0, 1, 2]:
            for m in [3, 4]:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def run_with_telemetry(weights):
    """Run a headless simulation and return full in-memory telemetry arrays.

    Captures position, velocity, orientation, ground contacts, joint states, and
    motor neuron outputs at every timestep for downstream Beer-framework analysis.

    Args:
        weights: Dict of 6-synapse weights (w03..w24).

    Returns:
        Dict of numpy arrays keyed by signal name (t, x, y, z, vx, vy, vz,
        wx, wy, wz, roll, pitch, yaw, contact_torso, contact_back, contact_front,
        j0_pos, j0_vel, j0_tau, j1_pos, j1_vel, j1_tau, m3_out, m4_out).

    Side effects:
        Overwrites brain.nndf via write_brain_6syn.
    """
    write_brain_6syn(weights)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(c.DT)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    # -1 is the base link (Torso); iterate all links to apply uniform friction
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
    roll = np.empty(n_steps); pitch = np.empty(n_steps); yaw = np.empty(n_steps)
    ct = np.empty(n_steps, dtype=bool)
    cb = np.empty(n_steps, dtype=bool)
    cf = np.empty(n_steps, dtype=bool)
    j0p = np.empty(n_steps); j0v = np.empty(n_steps); j0t = np.empty(n_steps)
    j1p = np.empty(n_steps); j1v = np.empty(n_steps); j1t = np.empty(n_steps)
    m3_out = np.empty(n_steps); m4_out = np.empty(n_steps)

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

    # Main simulation loop: apply motor commands before stepping physics,
    # then update the neural network with new sensor readings
    for i in range(n_steps):
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

        # Classify ground contacts by link: cp[3] is the link index on bodyA.
        # -1 = base link (Torso), others matched by their link indices.
        contact_pts = p.getContactPoints(robotId)
        tc = bc = fc = False
        for cp in contact_pts:
            li = cp[3]
            if li == -1: tc = True
            elif li == back_li: bc = True
            elif li == front_li: fc = True
        ct[i] = tc; cb[i] = bc; cf[i] = fc

        # Joint state tuple: (position, velocity, reactionForces, appliedTorque)
        # Index [3] is the motor torque actually applied at the joint
        js0 = p.getJointState(robotId, j0_idx)
        js1 = p.getJointState(robotId, j1_idx)
        j0p[i] = js0[0]; j0v[i] = js0[1]; j0t[i] = js0[3]
        j1p[i] = js1[0]; j1v[i] = js1[1]; j1t[i] = js1[3]

    p.disconnect()

    return {
        "t": t_arr, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "contact_torso": ct, "contact_back": cb, "contact_front": cf,
        "j0_pos": j0p, "j0_vel": j0v, "j0_tau": j0t,
        "j1_pos": j1p, "j1_vel": j1v, "j1_tau": j1t,
        "m3_out": m3_out, "m4_out": m4_out,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_ax(ax):
    """Remove top and right spines from a matplotlib axes for cleaner plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    """Save a figure to PLOT_DIR and close it to free memory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


def work_by_joint(data):
    """Compute per-joint mechanical work from torque and velocity time series.

    Args:
        data: Telemetry dict with "j0_tau", "j0_vel", "j1_tau", "j1_vel" arrays.

    Returns:
        Tuple of (work_j0, work_j1, power_j0_array, power_j1_array) where work
        values are scalar floats (Joule-like units) and power arrays are per-timestep.
    """
    # Instantaneous mechanical power = |torque * angular_velocity| per joint;
    # total work is the time-integral of power (sum * DT approximation)
    pj0 = np.abs(data["j0_tau"] * data["j0_vel"])
    pj1 = np.abs(data["j1_tau"] * data["j1_vel"])
    return float(np.sum(pj0) * DT), float(np.sum(pj1) * DT), pj0, pj1


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run 3-way gait comparison, print metrics table, and generate all 7 figures.

    Side effects:
        - Backs up and restores brain.nndf.
        - Runs 2 headless simulations (Novelty Champion, Trial 3).
        - Loads CPG Champion from saved telemetry.
        - Writes 7 PNG figures to artifacts/plots/.
        - Prints a detailed comparison table to stdout.
    """
    # Backup brain.nndf since run_with_telemetry overwrites it for each gait
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    # ── Run simulations ──────────────────────────────────────────────────────
    print("Running Novelty Champion...")
    nc_data = run_with_telemetry(NOVELTY_CHAMPION_WEIGHTS)
    nc_a = compute_all(nc_data, DT)

    print("Running Trial 3...")
    t3_data = run_with_telemetry(TRIAL3_WEIGHTS)
    t3_a = compute_all(t3_data, DT)

    # CPG Champion uses a hidden-layer topology, so load from saved telemetry
    # rather than re-simulating (its brain.nndf format differs from 6-synapse)
    print("Loading CPG Champion telemetry...")
    cpg_data = load_telemetry("43_hidden_cpg_champion")
    cpg_a = compute_all(cpg_data, DT)

    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # ── Metrics comparison ───────────────────────────────────────────────────
    # 3-way comparison: Novelty Champion (nc), CPG Champion (cpg), Trial 3 (t3).
    # Each row shows the same metric for all three gaits plus the NC/CPG ratio.
    gaits = {
        "nc": (nc_data, nc_a, NOVELTY_CHAMPION_WEIGHTS),
        "cpg": (cpg_data, cpg_a, None),
        "t3": (t3_data, t3_a, TRIAL3_WEIGHTS),
    }

    nc_w0_j0, nc_w0_j1, nc_pj0, nc_pj1 = work_by_joint(nc_data)
    cpg_w_j0, cpg_w_j1, cpg_pj0, cpg_pj1 = work_by_joint(cpg_data)
    t3_w_j0, t3_w_j1, t3_pj0, t3_pj1 = work_by_joint(t3_data)

    print("\n" + "=" * 80)
    print("NOVELTY CHAMPION vs CPG CHAMPION vs TRIAL 3")
    print("=" * 80)

    rows = [
        ("DX (m)",
         nc_a["outcome"]["dx"], cpg_a["outcome"]["dx"], t3_a["outcome"]["dx"]),
        ("DY (m)",
         nc_a["outcome"]["dy"], cpg_a["outcome"]["dy"], t3_a["outcome"]["dy"]),
        ("Net distance (m)",
         np.sqrt(nc_a["outcome"]["dx"]**2 + nc_a["outcome"]["dy"]**2),
         np.sqrt(cpg_a["outcome"]["dx"]**2 + cpg_a["outcome"]["dy"]**2),
         np.sqrt(t3_a["outcome"]["dx"]**2 + t3_a["outcome"]["dy"]**2)),
        ("Mean speed (m/s)",
         nc_a["outcome"]["mean_speed"], cpg_a["outcome"]["mean_speed"],
         t3_a["outcome"]["mean_speed"]),
        ("Speed CV",
         nc_a["outcome"]["speed_cv"], cpg_a["outcome"]["speed_cv"],
         t3_a["outcome"]["speed_cv"]),
        ("Work proxy",
         nc_a["outcome"]["work_proxy"], cpg_a["outcome"]["work_proxy"],
         t3_a["outcome"]["work_proxy"]),
        ("Efficiency (dist/work)",
         nc_a["outcome"]["distance_per_work"], cpg_a["outcome"]["distance_per_work"],
         t3_a["outcome"]["distance_per_work"]),
        ("Yaw net (rad)",
         nc_a["outcome"]["yaw_net_rad"], cpg_a["outcome"]["yaw_net_rad"],
         t3_a["outcome"]["yaw_net_rad"]),
        ("Phase lock",
         nc_a["coordination"]["phase_lock_score"], cpg_a["coordination"]["phase_lock_score"],
         t3_a["coordination"]["phase_lock_score"]),
        ("Delta phi (rad)",
         nc_a["coordination"]["delta_phi_rad"], cpg_a["coordination"]["delta_phi_rad"],
         t3_a["coordination"]["delta_phi_rad"]),
        ("Joint 0 freq (Hz)",
         nc_a["coordination"]["joint_0"]["dominant_freq_hz"],
         cpg_a["coordination"]["joint_0"]["dominant_freq_hz"],
         t3_a["coordination"]["joint_0"]["dominant_freq_hz"]),
        ("Joint 1 freq (Hz)",
         nc_a["coordination"]["joint_1"]["dominant_freq_hz"],
         cpg_a["coordination"]["joint_1"]["dominant_freq_hz"],
         t3_a["coordination"]["joint_1"]["dominant_freq_hz"]),
        ("Contact entropy (bits)",
         nc_a["contact"]["contact_entropy_bits"], cpg_a["contact"]["contact_entropy_bits"],
         t3_a["contact"]["contact_entropy_bits"]),
        ("Duty torso",
         nc_a["contact"]["duty_torso"], cpg_a["contact"]["duty_torso"],
         t3_a["contact"]["duty_torso"]),
        ("Duty back leg",
         nc_a["contact"]["duty_back"], cpg_a["contact"]["duty_back"],
         t3_a["contact"]["duty_back"]),
        ("Duty front leg",
         nc_a["contact"]["duty_front"], cpg_a["contact"]["duty_front"],
         t3_a["contact"]["duty_front"]),
        ("Roll dominance",
         nc_a["rotation_axis"]["axis_dominance"][0],
         cpg_a["rotation_axis"]["axis_dominance"][0],
         t3_a["rotation_axis"]["axis_dominance"][0]),
        ("BackLeg work",
         nc_w0_j0, cpg_w_j0, t3_w_j0),
        ("FrontLeg work",
         nc_w0_j1, cpg_w_j1, t3_w_j1),
    ]

    print(f"\n  {'Metric':<28} {'Nov. Champ':>12} {'CPG Champ':>12} {'Trial 3':>12} {'NC/CPG':>8}")
    print("  " + "-" * 76)
    for label, vnc, vcpg, vt3 in rows:
        # Ratio shows how the Novelty Champion scales relative to CPG Champion;
        # guard against division by near-zero CPG values
        ratio = vnc / vcpg if abs(vcpg) > 1e-12 else 0
        print(f"  {label:<28} {vnc:12.4f} {vcpg:12.4f} {vt3:12.4f} {ratio:7.2f}x")

    # ── Weight analysis ──────────────────────────────────────────────────────
    print("\n  NOVELTY CHAMPION WEIGHT STRUCTURE:")
    w = NOVELTY_CHAMPION_WEIGHTS
    print(f"    Sensor 0 (Torso):    w03={w['w03']:+.4f}  w04={w['w04']:+.4f}  "
          f"sum={w['w03']+w['w04']:+.4f}  diff={w['w03']-w['w04']:+.4f}")
    print(f"    Sensor 1 (BackLeg):  w13={w['w13']:+.4f}  w14={w['w14']:+.4f}  "
          f"sum={w['w13']+w['w14']:+.4f}  diff={w['w13']-w['w14']:+.4f}")
    print(f"    Sensor 2 (FrontLeg): w23={w['w23']:+.4f}  w24={w['w24']:+.4f}  "
          f"sum={w['w23']+w['w24']:+.4f}  diff={w['w23']-w['w24']:+.4f}")

    # Sum of all sensor-to-motor weights: net excitatory/inhibitory drive per motor
    total_to_m3 = w['w03'] + w['w13'] + w['w23']
    total_to_m4 = w['w04'] + w['w14'] + w['w24']
    print(f"\n    Total drive to BackLeg motor (m3): {total_to_m3:+.4f}")
    print(f"    Total drive to FrontLeg motor (m4): {total_to_m4:+.4f}")
    if abs(total_to_m4) > 1e-12:
        print(f"    Ratio |m3/m4|: {abs(total_to_m3/total_to_m4):.2f}")

    print(f"\n    Key observation: w03={w['w03']:+.3f} is OUTSIDE [-1,1]")
    print(f"    This weight emerged from unclamped perturbation in the Novelty Seeker.")

    # ── FIGURES ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    CNC = "#C44E52"   # Novelty Champion
    CCPG = "#4C72B0"  # CPG Champion
    CT3 = "#55A868"   # Trial 3

    nc_t = nc_data["t"]
    cpg_t = cpg_data["t"]
    t3_t = t3_data["t"]

    # ── Fig 1: Trajectories ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.plot(nc_data["x"], nc_data["y"], color=CNC, lw=0.5, alpha=0.8, label="Novelty Champ")
    ax.plot(cpg_data["x"], cpg_data["y"], color=CCPG, lw=0.5, alpha=0.8, label="CPG Champ")
    ax.plot(t3_data["x"], t3_data["y"], color=CT3, lw=0.5, alpha=0.8, label="Trial 3")
    ax.scatter([0], [0], c="black", s=50, zorder=5, marker="o")
    for data, color in [(nc_data, CNC), (cpg_data, CCPG), (t3_data, CT3)]:
        ax.scatter([data["x"][-1]], [data["y"][-1]], c=color, s=80, zorder=5, marker="*")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    clean_ax(ax)

    ax = axes[1]
    ax.plot(nc_t, nc_data["x"], color=CNC, lw=0.8, label="Novelty Champ")
    ax.plot(cpg_t, cpg_data["x"], color=CCPG, lw=0.8, label="CPG Champ")
    ax.plot(t3_t, t3_data["x"], color=CT3, lw=0.8, label="Trial 3")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("X position (m)")
    ax.set_title("X vs Time")
    ax.legend(fontsize=8)
    clean_ax(ax)

    ax = axes[2]
    ax.plot(nc_t, nc_data["z"], color=CNC, lw=0.4, alpha=0.7, label="Novelty Champ")
    ax.plot(cpg_t, cpg_data["z"], color=CCPG, lw=0.4, alpha=0.7, label="CPG Champ")
    ax.plot(t3_t, t3_data["z"], color=CT3, lw=0.4, alpha=0.7, label="Trial 3")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Z height (m)")
    ax.set_title("Vertical Bounce")
    ax.legend(fontsize=8)
    clean_ax(ax)

    fig.suptitle("Three Champions: Trajectory Comparison", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "champ_fig01_trajectory.png")

    # ── Fig 2: Joint dynamics ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for col, (data, t, label, color) in enumerate([
            (nc_data, nc_t, "Novelty Champ", CNC),
            (cpg_data, cpg_t, "CPG Champ", CCPG),
            (t3_data, t3_t, "Trial 3", CT3)]):

        ax = axes[0][col]
        ax.plot(t, data["j0_pos"], color="#DD8452", lw=0.5, label="BackLeg (j0)")
        ax.plot(t, data["j1_pos"], color="#8172B2", lw=0.5, label="FrontLeg (j1)")
        ax.set_title(f"{label} — Joint Positions")
        ax.set_ylabel("Angle (rad)")
        ax.legend(fontsize=7); clean_ax(ax)

        ax = axes[1][col]
        ax.plot(t, data["j0_vel"], color="#DD8452", lw=0.3, alpha=0.7, label="j0 vel")
        ax.plot(t, data["j1_vel"], color="#8172B2", lw=0.3, alpha=0.7, label="j1 vel")
        ax.set_title(f"{label} — Joint Velocities")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity (rad/s)")
        ax.legend(fontsize=7); clean_ax(ax)

    fig.suptitle("Joint Dynamics: Three Champions", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "champ_fig02_joints.png")

    # ── Fig 3: Contact patterns ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

    for ax, (data, label, color) in zip(axes, [
            (nc_data, "Novelty Champion", CNC),
            (cpg_data, "CPG Champion", CCPG),
            (t3_data, "Trial 3", CT3)]):
        t_plot = data["t"]
        ct_v = data["contact_torso"].astype(float)
        cb_v = data["contact_back"].astype(float)
        cf_v = data["contact_front"].astype(float)
        # Stack contact bands vertically (0, 1, 2) with 0.9 height per band;
        # fill_between shows solid color when contact is True (1.0)
        ax.fill_between(t_plot, 0, ct_v * 0.9, alpha=0.5, color="#999999", label="Torso")
        ax.fill_between(t_plot, 1, 1 + cb_v * 0.9, alpha=0.5, color="#DD8452", label="BackLeg")
        ax.fill_between(t_plot, 2, 2 + cf_v * 0.9, alpha=0.5, color="#8172B2", label="FrontLeg")
        ax.set_yticks([0.45, 1.45, 2.45])
        ax.set_yticklabels(["Torso", "BackLeg", "FrontLeg"])
        ax.set_title(f"{label} — Contact Pattern", color=color)
        ax.legend(fontsize=7, loc="upper right")
        clean_ax(ax)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, "champ_fig03_contacts.png")

    # ── Fig 4: Energy ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for col, (data, t, label, color, pj0, pj1, wj0, wj1) in enumerate([
            (nc_data, nc_t, "Novelty Champ", CNC, nc_pj0, nc_pj1, nc_w0_j0, nc_w0_j1),
            (cpg_data, cpg_t, "CPG Champ", CCPG, cpg_pj0, cpg_pj1, cpg_w_j0, cpg_w_j1),
            (t3_data, t3_t, "Trial 3", CT3, t3_pj0, t3_pj1, t3_w_j0, t3_w_j1)]):

        ax = axes[0][col]
        ax.plot(t, pj0, color="#DD8452", lw=0.3, alpha=0.6, label="j0 (BackLeg)")
        ax.plot(t, pj1, color="#8172B2", lw=0.3, alpha=0.6, label="j1 (FrontLeg)")
        ax.set_title(f"{label} — Instantaneous Power")
        ax.set_ylabel("|Power|")
        ax.legend(fontsize=7); clean_ax(ax)

        ax = axes[1][col]
        # Cumulative work = running integral of |power| over time (Riemann sum)
        cum_j0 = np.cumsum(pj0) * DT
        cum_j1 = np.cumsum(pj1) * DT
        ax.plot(t, cum_j0, color="#DD8452", lw=1.5, label=f"j0 ({wj0:.0f})")
        ax.plot(t, cum_j1, color="#8172B2", lw=1.5, label=f"j1 ({wj1:.0f})")
        ax.plot(t, cum_j0 + cum_j1, color=color, lw=2, ls="--",
                label=f"Total ({wj0+wj1:.0f})")
        ax.set_title(f"{label} — Cumulative Work")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Cumulative work")
        ax.legend(fontsize=7); clean_ax(ax)

    fig.suptitle("Energy Expenditure: Three Champions", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "champ_fig04_energy.png")

    # ── Fig 5: Phase portraits ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (data, label, color) in zip(axes, [
            (nc_data, "Novelty Champion", CNC),
            (cpg_data, "CPG Champion", CCPG),
            (t3_data, "Trial 3", CT3)]):
        # Phase portrait: j0 vs j1 angle. A tight ellipse means strong phase-locking.
        ax.plot(data["j0_pos"], data["j1_pos"], color=color, lw=0.3, alpha=0.5)
        # Color by time: subsample to 500 points for readable scatter overlay
        n = len(data["j0_pos"])
        scatter_idx = np.linspace(0, n - 1, 500, dtype=int)
        sc = ax.scatter(data["j0_pos"][scatter_idx], data["j1_pos"][scatter_idx],
                        c=scatter_idx, cmap="viridis", s=5, alpha=0.7, zorder=3)
        ax.scatter([data["j0_pos"][0]], [data["j1_pos"][0]], c="green", s=40,
                   marker="^", zorder=5, label="start")
        ax.set_xlabel("BackLeg angle (rad)")
        ax.set_ylabel("FrontLeg angle (rad)")
        ax.set_title(f"{label}\nPL={compute_all(data, DT)['coordination']['phase_lock_score']:.3f}")
        ax.legend(fontsize=7)
        clean_ax(ax)

    fig.suptitle("Phase Portraits (j0 vs j1, colored by time)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "champ_fig05_phase.png")

    # ── Fig 6: Angular velocity ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    components = [("wx", "Roll"), ("wy", "Pitch"), ("wz", "Yaw")]

    for row, (data, t, label, color) in enumerate([
            (nc_data, nc_t, "Novelty Champ", CNC),
            (cpg_data, cpg_t, "CPG Champ", CCPG),
            (t3_data, t3_t, "Trial 3", CT3)]):
        for col, (key, comp_label) in enumerate(components):
            ax = axes[row][col]
            ax.plot(t, data[key], color=color, lw=0.3, alpha=0.7)
            # RMS quantifies the overall rotational intensity for each axis
            rms = np.sqrt(np.mean(data[key]**2))
            ax.set_title(f"{label} — {comp_label} (RMS={rms:.2f})", fontsize=9)
            if row == 2:
                ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("Angular vel (rad/s)")
            clean_ax(ax)

    fig.suptitle("Angular Velocity Components", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "champ_fig06_rotation.png")

    # ── Fig 7: FFT spectra ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (data, label, color) in zip(axes, [
            (nc_data, "Novelty Champion", CNC),
            (cpg_data, "CPG Champion", CCPG),
            (t3_data, "Trial 3", CT3)]):
        for jkey, jlabel, jcolor in [("j0_pos", "BackLeg", "#DD8452"),
                                      ("j1_pos", "FrontLeg", "#8172B2")]:
            # Remove DC offset before FFT so the zero-frequency bin doesn't dominate
            sig = data[jkey] - np.mean(data[jkey])
            n = len(sig)
            freqs = np.fft.rfftfreq(n, d=DT)
            # Normalize to single-sided amplitude spectrum (factor of 2/n)
            spectrum = np.abs(np.fft.rfft(sig)) * (2.0 / n)
            spectrum[0] = 0
            # Show up to 10 Hz
            max_idx = min(len(freqs), int(10 / (freqs[1] - freqs[0])) + 1)
            ax.plot(freqs[:max_idx], spectrum[:max_idx], color=jcolor, lw=1.2, label=jlabel)

            # Mark peak: skip bin 0 (DC) and search within the displayed range
            peak_idx = np.argmax(spectrum[1:max_idx]) + 1
            ax.annotate(f"{freqs[peak_idx]:.2f} Hz",
                        (freqs[peak_idx], spectrum[peak_idx]),
                        fontsize=7, ha="left", color=jcolor)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{label} — Joint FFT")
        ax.legend(fontsize=8)
        clean_ax(ax)

    fig.suptitle("FFT Spectra of Joint Angles", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "champ_fig07_fft.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
