#!/usr/bin/env python3
"""
random_search_500.py

Run 500 random-search trials with in-memory Beer-framework analytics.
No telemetry files written to disk — data is captured during simulation
and analyzed on the fly.

Outputs:
    artifacts/random_search_500.json         — all 500 trial results
    artifacts/plots/rs_fig01_dx_histogram.png
    artifacts/plots/rs_fig02_phase_lock_histogram.png
    artifacts/plots/rs_fig03_best_of_n.png
    artifacts/plots/rs_fig04_weight_correlations.png
    artifacts/plots/rs_fig05_speed_efficiency.png
    artifacts/plots/rs_fig06_dead_fraction.png
    artifacts/plots/rs_fig07_symmetry.png

Usage:
    python3 random_search_500.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from compute_beer_analytics import compute_all, DT, NumpyEncoder

NUM_TRIALS = 500
SENSOR_NEURONS = [0, 1, 2]
MOTOR_NEURONS = [3, 4]
WEIGHT_NAMES = [f"w{s}{m}" for s in SENSOR_NEURONS for m in MOTOR_NEURONS]
OUT_JSON = PROJECT / "artifacts" / "random_search_500.json"
PLOT_DIR = PROJECT / "artifacts" / "plots"


# ── Simulation with in-memory telemetry ──────────────────────────────────────

def write_brain(weights):
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


def run_trial_inmemory(weights):
    """Run simulation, capture telemetry arrays in memory, return analytics dict."""
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
    max_force = float(getattr(c, "MAX_FORCE", 150.0))
    n_steps = c.SIM_STEPS

    # Pre-allocate arrays
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

    # Link/joint indices
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
    # Base link (Torso) is typically -1 for contacts
    torso_link_idx = link_indices.get("Torso", -1)
    back_link_idx = link_indices.get("BackLeg", -1)
    front_link_idx = link_indices.get("FrontLeg", -1)

    j0_idx = joint_indices.get("Torso_BackLeg", 0)
    j1_idx = joint_indices.get("Torso_FrontLeg", 1)

    for i in range(n_steps):
        # Act
        for neuronName in nn.neurons:
            n_obj = nn.neurons[neuronName]
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

        # Record telemetry in-memory
        t_arr[i] = i * c.DT
        pos, orn = p.getBasePositionAndOrientation(robotId)
        vel_lin, vel_ang = p.getBaseVelocity(robotId)
        rpy_vals = p.getEulerFromQuaternion(orn)

        x[i] = pos[0]; y[i] = pos[1]; z[i] = pos[2]
        vx[i] = vel_lin[0]; vy[i] = vel_lin[1]; vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]; wy[i] = vel_ang[1]; wz[i] = vel_ang[2]
        roll[i] = rpy_vals[0]; pitch[i] = rpy_vals[1]; yaw[i] = rpy_vals[2]

        # Contacts
        contact_pts = p.getContactPoints(robotId)
        torso_contact = False; back_contact = False; front_contact = False
        for cp in contact_pts:
            li = cp[3]  # linkIndexA
            if li == -1:  # base link = Torso
                torso_contact = True
            elif li == back_link_idx:
                back_contact = True
            elif li == front_link_idx:
                front_contact = True
        contact_torso[i] = torso_contact
        contact_back[i] = back_contact
        contact_front[i] = front_contact

        # Joint states
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

    analytics = compute_all(data, DT)
    return analytics


# ── Plotting helpers ─────────────────────────────────────────────────────────

def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_fig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")

def correlation_r(x, y):
    x, y = np.array(x), np.array(y)
    mx, my = np.mean(x), np.mean(y)
    num = np.sum((x - mx) * (y - my))
    den = np.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2))
    return num / den if den > 1e-12 else 0.0


# ── Load zoo for context ─────────────────────────────────────────────────────

def load_zoo_summary():
    with open(PROJECT / "synapse_gait_zoo_v2.json") as f:
        zoo = json.load(f)
    gaits = []
    for cat in zoo["categories"].values():
        for gname, gdata in cat.get("gaits", {}).items():
            a = gdata.get("analytics", {})
            o = a.get("outcome", {})
            coord = a.get("coordination", {})
            contact = a.get("contact", {})
            ra = a.get("rotation_axis", {})
            gaits.append({
                "name": gname,
                "dx": o.get("dx", 0),
                "speed": o.get("mean_speed", 0),
                "efficiency": o.get("distance_per_work", 0),
                "phase_lock": coord.get("phase_lock_score", 0),
                "entropy": contact.get("contact_entropy_bits", 0),
                "roll_dom": ra.get("axis_dominance", [0,0,0])[0],
                "work_proxy": o.get("work_proxy", 0),
            })
    return gaits


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Back up brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    results = []
    print(f"Running {NUM_TRIALS} random-search trials (in-memory analytics)...\n")
    t_total = time.perf_counter()

    for trial in range(NUM_TRIALS):
        weights = {wn: random.uniform(-1, 1) for wn in WEIGHT_NAMES}
        analytics = run_trial_inmemory(weights)

        result = {"weights": weights, "analytics": analytics}
        results.append(result)

        if (trial + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_total
            o = analytics["outcome"]
            print(f"  [{trial+1:3d}/{NUM_TRIALS}] {elapsed:.1f}s  "
                  f"last DX={o['dx']:+7.3f}", flush=True)

    total_elapsed = time.perf_counter() - t_total
    print(f"\nSimulation complete: {NUM_TRIALS} trials in {total_elapsed:.1f}s "
          f"({total_elapsed/NUM_TRIALS:.3f}s/trial)\n")

    # Restore brain.nndf
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # ── Extract flat arrays ──────────────────────────────────────────────────

    dx = np.array([r["analytics"]["outcome"]["dx"] for r in results])
    speed = np.array([r["analytics"]["outcome"]["mean_speed"] for r in results])
    efficiency = np.array([r["analytics"]["outcome"]["distance_per_work"] for r in results])
    work = np.array([r["analytics"]["outcome"]["work_proxy"] for r in results])
    phase_lock = np.array([r["analytics"]["coordination"]["phase_lock_score"] for r in results])
    entropy = np.array([r["analytics"]["contact"]["contact_entropy_bits"] for r in results])
    roll_dom = np.array([r["analytics"]["rotation_axis"]["axis_dominance"][0] for r in results])
    yaw_net = np.array([r["analytics"]["outcome"]["yaw_net_rad"] for r in results])
    abs_dx = np.abs(dx)

    W = np.array([[r["weights"][wn] for wn in WEIGHT_NAMES] for r in results])

    # ── Save JSON ────────────────────────────────────────────────────────────

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Save compact: weights + key metrics only (not full analytics)
    compact = []
    for i, r in enumerate(results):
        o = r["analytics"]["outcome"]
        coord = r["analytics"]["coordination"]
        contact = r["analytics"]["contact"]
        ra = r["analytics"]["rotation_axis"]
        compact.append({
            "trial": i,
            "weights": r["weights"],
            "dx": o["dx"], "speed": o["mean_speed"],
            "efficiency": o["distance_per_work"], "work_proxy": o["work_proxy"],
            "phase_lock": coord["phase_lock_score"],
            "entropy": contact["contact_entropy_bits"],
            "roll_dom": ra["axis_dominance"][0],
            "yaw_net_rad": o["yaw_net_rad"],
        })
    with open(OUT_JSON, "w") as f:
        json.dump(compact, f, indent=2, cls=NumpyEncoder)
    print(f"WROTE {OUT_JSON}\n")

    # ── Load zoo for context ─────────────────────────────────────────────────

    zoo = load_zoo_summary()
    zoo_dx = np.array([g["dx"] for g in zoo])
    zoo_speed = np.array([g["speed"] for g in zoo])
    zoo_eff = np.array([g["efficiency"] for g in zoo])
    zoo_pl = np.array([g["phase_lock"] for g in zoo])
    zoo_ent = np.array([g["entropy"] for g in zoo])

    # ── Print summary stats ──────────────────────────────────────────────────

    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    dead_thresh = 1.0
    n_dead = np.sum(abs_dx < dead_thresh)
    n_forward = np.sum(dx > dead_thresh)
    n_backward = np.sum(dx < -dead_thresh)
    print(f"  Dead (|DX| < {dead_thresh}m): {n_dead}/{NUM_TRIALS} ({n_dead/NUM_TRIALS*100:.1f}%)")
    print(f"  Forward (DX > {dead_thresh}m): {n_forward}/{NUM_TRIALS} ({n_forward/NUM_TRIALS*100:.1f}%)")
    print(f"  Backward (DX < -{dead_thresh}m): {n_backward}/{NUM_TRIALS} ({n_backward/NUM_TRIALS*100:.1f}%)")
    print()
    print(f"  |DX|  — mean: {np.mean(abs_dx):.2f}, median: {np.median(abs_dx):.2f}, "
          f"max: {np.max(abs_dx):.2f}, P90: {np.percentile(abs_dx, 90):.2f}")
    print(f"  Speed — mean: {np.mean(speed):.3f}, max: {np.max(speed):.3f}")
    print(f"  Phase lock — mean: {np.mean(phase_lock):.3f}, "
          f"frac>0.8: {np.mean(phase_lock > 0.8):.2f}, frac<0.2: {np.mean(phase_lock < 0.2):.2f}")
    print(f"  Roll dom — mean: {np.mean(roll_dom):.3f}, min: {np.min(roll_dom):.3f}")
    print()

    # Weight correlations
    print("WEIGHT–METRIC CORRELATIONS (Pearson r)")
    print(f"  {'Weight':<8} {'DX':>8} {'|DX|':>8} {'Speed':>8} {'PhaseLk':>8} {'Entropy':>8}")
    print("  " + "-" * 50)
    for j, wn in enumerate(WEIGHT_NAMES):
        r_dx = correlation_r(W[:, j], dx)
        r_abs = correlation_r(W[:, j], abs_dx)
        r_spd = correlation_r(W[:, j], speed)
        r_pl = correlation_r(W[:, j], phase_lock)
        r_ent = correlation_r(W[:, j], entropy)
        print(f"  {wn:<8} {r_dx:+8.3f} {r_abs:+8.3f} {r_spd:+8.3f} {r_pl:+8.3f} {r_ent:+8.3f}")
    print()

    # Best trial
    best_idx = np.argmax(abs_dx)
    best = results[best_idx]
    bo = best["analytics"]["outcome"]
    print(f"BEST MOVER: trial {best_idx}")
    print(f"  DX={bo['dx']:+.3f}  speed={bo['mean_speed']:.4f}  "
          f"eff={bo['distance_per_work']:.5f}")
    print(f"  Weights: {best['weights']}")
    print(f"  Zoo CPG Champion DX=+50.11 → random best is "
          f"{abs_dx[best_idx]/50.11*100:.1f}% of zoo best")

    # ── FIGURES ───────────────────────────────────────────────────────────────

    print("\nGenerating figures...")

    # Fig 1: DX histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.hist(dx, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("DX (meters)")
    ax.set_ylabel("Count")
    ax.set_title(f"Displacement Distribution (n={NUM_TRIALS})")
    clean_ax(ax)

    ax = axes[1]
    ax.hist(abs_dx, bins=40, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(abs_dx), color="red", lw=1.2, ls="--",
               label=f"median={np.median(abs_dx):.2f}")
    ax.axvline(dead_thresh, color="gray", lw=1, ls=":",
               label=f"|DX|<{dead_thresh}m = 'dead' ({n_dead/NUM_TRIALS*100:.0f}%)")
    ax.set_xlabel("|DX| (meters)")
    ax.set_ylabel("Count")
    ax.set_title("|DX| Distribution")
    ax.legend(fontsize=9)
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "rs_fig01_dx_histogram.png")

    # Fig 2: Phase lock histogram (bimodality test)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(phase_lock, bins=40, color="#C44E52", edgecolor="white", alpha=0.85,
            label=f"Random search (n={NUM_TRIALS})")
    ax.hist(zoo_pl, bins=40, color="#4C72B0", edgecolor="white", alpha=0.5,
            label=f"Zoo (n={len(zoo)})")
    ax.set_xlabel("Phase Lock Score")
    ax.set_ylabel("Count")
    ax.set_title("Phase Lock Distribution: Random Search vs Zoo")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "rs_fig02_phase_lock_histogram.png")

    # Fig 3: Best-of-N curve
    fig, ax = plt.subplots(figsize=(8, 5))
    best_so_far = np.maximum.accumulate(abs_dx)
    ax.plot(np.arange(1, NUM_TRIALS + 1), best_so_far, color="#4C72B0", lw=1.5)
    ax.axhline(np.max(np.abs(zoo_dx)), color="red", lw=1, ls="--",
               label=f"Zoo best |DX|={np.max(np.abs(zoo_dx)):.1f}")
    ax.axhline(np.median(np.abs(zoo_dx)), color="orange", lw=1, ls="--",
               label=f"Zoo median |DX|={np.median(np.abs(zoo_dx)):.1f}")
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Best |DX| so far (meters)")
    ax.set_title("Random Search: Best-of-N Curve")
    ax.legend(fontsize=9)
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "rs_fig03_best_of_n.png")

    # Fig 4: Weight correlations heatmap
    metrics_arr = np.column_stack([dx, abs_dx, speed, phase_lock, entropy, roll_dom])
    metric_labels = ["DX", "|DX|", "Speed", "Phase Lock", "Entropy", "Roll Dom"]
    corr = np.zeros((len(WEIGHT_NAMES), len(metric_labels)))
    for j in range(len(WEIGHT_NAMES)):
        for k in range(len(metric_labels)):
            corr[j, k] = correlation_r(W[:, j], metrics_arr[:, k])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(WEIGHT_NAMES)))
    ax.set_yticklabels(WEIGHT_NAMES, fontsize=10)
    for j in range(corr.shape[0]):
        for k in range(corr.shape[1]):
            ax.text(k, j, f"{corr[j,k]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr[j, k]) > 0.3 else "black")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Weight–Metric Correlations (500 random trials)")
    fig.tight_layout()
    save_fig(fig, "rs_fig04_weight_correlations.png")

    # Fig 5: Speed vs Efficiency — random trials + zoo context
    fig, ax = plt.subplots(figsize=(8, 6))
    # Zoo background
    zoo_eff_clip = np.clip(zoo_eff, 0, np.percentile(zoo_eff, 97))
    ax.scatter(zoo_speed, zoo_eff_clip, c="#CCCCCC", s=30, alpha=0.6, label="Zoo (116)", zorder=1)
    # Random trials
    eff_clip = np.clip(efficiency, 0, np.percentile(zoo_eff, 97))
    sc = ax.scatter(speed, eff_clip, c=abs_dx, cmap="viridis", s=20, alpha=0.8,
                    edgecolors="black", linewidths=0.3, label=f"Random ({NUM_TRIALS})", zorder=2)
    plt.colorbar(sc, ax=ax, label="|DX| (meters)")
    # Label zoo champions
    for g in zoo:
        if g["name"] in ("43_hidden_cpg_champion", "18_curie", "7_fuller_dymaxion"):
            eff_v = min(g["efficiency"], np.percentile(zoo_eff, 97))
            ax.annotate(g["name"].split("_", 1)[1], (g["speed"], eff_v),
                        fontsize=7, ha="left", style="italic",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="gray"))
    ax.set_xlabel("Mean Speed")
    ax.set_ylabel("Efficiency (distance/work)")
    ax.set_title("Speed vs Efficiency: Random Search in Zoo Context")
    ax.legend(fontsize=9)
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "rs_fig05_speed_efficiency.png")

    # Fig 6: Dead fraction — cumulative |DX|
    fig, ax = plt.subplots(figsize=(7, 5))
    sorted_abs = np.sort(abs_dx)
    cdf = np.arange(1, NUM_TRIALS + 1) / NUM_TRIALS
    ax.plot(sorted_abs, cdf, color="#4C72B0", lw=2)
    ax.axvline(1.0, color="gray", ls=":", lw=1, label="|DX|=1m threshold")
    ax.axhline(n_dead / NUM_TRIALS, color="gray", ls="--", lw=0.8)
    ax.text(1.2, n_dead / NUM_TRIALS + 0.02,
            f"{n_dead/NUM_TRIALS*100:.0f}% dead", fontsize=9, color="gray")
    # Zoo median
    ax.axvline(np.median(np.abs(zoo_dx)), color="orange", ls="--", lw=1,
               label=f"Zoo median |DX|={np.median(np.abs(zoo_dx)):.1f}m")
    ax.set_xlabel("|DX| (meters)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of |DX|: How Much of Weight Space Is Dead?")
    ax.legend(fontsize=9)
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "rs_fig06_dead_fraction.png")

    # Fig 7: Symmetry — |w_s3 + w_s4| (antisymmetry score) vs |DX|
    # For each sensor s, compute how antisymmetric the pair (w_s3, w_s4) is
    # Perfect antisymmetry: w_s3 = -w_s4, so w_s3 + w_s4 = 0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    sensor_names = ["Torso (s0)", "BackLeg (s1)", "FrontLeg (s2)"]
    for idx, (s, label) in enumerate(zip(SENSOR_NEURONS, sensor_names)):
        w_s3 = W[:, WEIGHT_NAMES.index(f"w{s}3")]
        w_s4 = W[:, WEIGHT_NAMES.index(f"w{s}4")]
        antisym = np.abs(w_s3 + w_s4)  # 0 = perfect antisymmetry
        ax = axes[idx]
        sc = ax.scatter(antisym, abs_dx, c=phase_lock, cmap="coolwarm", s=12,
                        alpha=0.6, edgecolors="none")
        ax.set_xlabel(f"|w{s}3 + w{s}4| (0=antisymmetric)")
        ax.set_ylabel("|DX|")
        ax.set_title(label)
        clean_ax(ax)
    plt.colorbar(sc, ax=axes[-1], label="Phase Lock")
    fig.suptitle("Weight Antisymmetry vs Displacement", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "rs_fig07_symmetry.png")

    print(f"\nDone. {total_elapsed:.1f}s total.")


if __name__ == "__main__":
    main()
