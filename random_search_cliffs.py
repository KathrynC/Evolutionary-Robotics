#!/usr/bin/env python3
"""
random_search_cliffs.py

Explore the local neighborhood of random points in weight space to
characterize the "cliffiness" of the fitness landscape.

For each of 50 random base points:
  - Simulate to get base DX
  - Perturb in 10 random directions at each of 3 radii (0.05, 0.1, 0.2)
  - Measure |delta_DX| for each perturbation

A "cliff" is defined as a perturbation where |delta_DX| exceeds a
threshold relative to the perturbation radius.

Outputs:
    artifacts/random_search_cliffs.json
    artifacts/plots/cliff_fig01_delta_dx_by_radius.png
    artifacts/plots/cliff_fig02_cliff_probability.png
    artifacts/plots/cliff_fig03_landscape_profiles.png
    artifacts/plots/cliff_fig04_cliff_vs_base_dx.png
    artifacts/plots/cliff_fig05_worst_cliff_histogram.png
    artifacts/random_search_cliffs_analysis.md

Usage:
    python3 random_search_cliffs.py
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

NUM_BASE = 50
NUM_PERTURBATIONS = 10
RADII = [0.05, 0.1, 0.2]
WEIGHT_NAMES = [f"w{s}{m}" for s in [0,1,2] for m in [3,4]]
PLOT_DIR = PROJECT / "artifacts" / "plots"
OUT_JSON = PROJECT / "artifacts" / "random_search_cliffs.json"


def write_brain(weights):
    """Write a brain.nndf file with the given synapse weights."""
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for s in [0,1,2]:
            for m in [3,4]:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def simulate_weights(weights):
    """Run one headless simulation, return (dx, analytics_dict)."""
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

    # Pre-allocate
    t_arr = np.empty(n_steps)
    x = np.empty(n_steps); y = np.empty(n_steps); z_arr = np.empty(n_steps)
    vx = np.empty(n_steps); vy = np.empty(n_steps); vz = np.empty(n_steps)
    wx = np.empty(n_steps); wy = np.empty(n_steps); wz = np.empty(n_steps)
    roll_a = np.empty(n_steps); pitch_a = np.empty(n_steps); yaw_a = np.empty(n_steps)
    ct = np.empty(n_steps, dtype=bool)
    cb = np.empty(n_steps, dtype=bool)
    cf = np.empty(n_steps, dtype=bool)
    j0p = np.empty(n_steps); j0v = np.empty(n_steps); j0t = np.empty(n_steps)
    j1p = np.empty(n_steps); j1v = np.empty(n_steps); j1t = np.empty(n_steps)

    # Get link/joint indices
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

        x[i] = pos[0]; y[i] = pos[1]; z_arr[i] = pos[2]
        vx[i] = vel_lin[0]; vy[i] = vel_lin[1]; vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]; wy[i] = vel_ang[1]; wz[i] = vel_ang[2]
        roll_a[i] = rpy_vals[0]; pitch_a[i] = rpy_vals[1]; yaw_a[i] = rpy_vals[2]

        contact_pts = p.getContactPoints(robotId)
        tc = False; bc = False; fc = False
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

    data = {
        "t": t_arr, "x": x, "y": y, "z": z_arr,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll_a, "pitch": pitch_a, "yaw": yaw_a,
        "contact_torso": ct, "contact_back": cb, "contact_front": cf,
        "j0_pos": j0p, "j0_vel": j0v, "j0_tau": j0t,
        "j1_pos": j1p, "j1_vel": j1v, "j1_tau": j1t,
    }

    analytics = compute_all(data, DT)
    dx = float(x[-1] - x[0])
    return dx, analytics


def random_direction_6d():
    """Return a random unit vector in 6D."""
    v = np.random.randn(6)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        v = np.ones(6)
        norm = np.linalg.norm(v)
    return v / norm


def perturb_weights(base_weights, direction, radius):
    """Return a new weight dict = base + radius * direction."""
    w = {}
    for i, wn in enumerate(WEIGHT_NAMES):
        w[wn] = base_weights[wn] + radius * direction[i]
    return w


def clean_ax(ax):
    """Remove top and right spines from an axes for cleaner plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_fig(fig, name):
    """Save a matplotlib figure to PLOT_DIR and close it."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


def main():
    """Explore cliff structure of the fitness landscape via perturbation analysis."""
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    total_sims = NUM_BASE * (1 + len(RADII) * NUM_PERTURBATIONS)
    print(f"Cliff exploration: {NUM_BASE} base points x {len(RADII)} radii x "
          f"{NUM_PERTURBATIONS} perturbations = {total_sims} simulations")
    print(f"Estimated time: {total_sims * 0.093:.0f}s\n")

    all_results = []
    t_total = time.perf_counter()

    for b in range(NUM_BASE):
        # Random base point
        base_w = {wn: random.uniform(-1, 1) for wn in WEIGHT_NAMES}
        base_dx, base_analytics = simulate_weights(base_w)

        base_result = {
            "base_idx": b,
            "base_weights": base_w,
            "base_dx": base_dx,
            "base_phase_lock": base_analytics["coordination"]["phase_lock_score"],
            "base_speed": base_analytics["outcome"]["mean_speed"],
            "base_entropy": base_analytics["contact"]["contact_entropy_bits"],
            "perturbations": {},
        }

        # Generate random directions once and reuse across all radii, so we
        # can compare how the same direction behaves at different step sizes.
        directions = [random_direction_6d() for _ in range(NUM_PERTURBATIONS)]

        for radius in RADII:
            r_key = f"r{radius}"
            perturbs = []

            for d_idx, direction in enumerate(directions):
                # Perturb: new_weights = base + radius * unit_direction
                pw = perturb_weights(base_w, direction, radius)
                p_dx, p_analytics = simulate_weights(pw)
                # delta_dx captures how much the behavioral outcome shifted
                delta_dx = p_dx - base_dx
                delta_pl = (p_analytics["coordination"]["phase_lock_score"]
                            - base_result["base_phase_lock"])
                delta_speed = (p_analytics["outcome"]["mean_speed"]
                               - base_result["base_speed"])

                perturbs.append({
                    "dx": p_dx,
                    "delta_dx": delta_dx,
                    "delta_phase_lock": delta_pl,
                    "delta_speed": delta_speed,
                    "phase_lock": p_analytics["coordination"]["phase_lock_score"],
                })

            base_result["perturbations"][r_key] = perturbs

        all_results.append(base_result)

        if (b + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_total
            done = (b + 1) * (1 + len(RADII) * NUM_PERTURBATIONS)
            rate = elapsed / done
            remaining = rate * (total_sims - done)
            print(f"  [{b+1:3d}/{NUM_BASE}] {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining  "
                  f"base DX={base_dx:+7.2f}", flush=True)

    total_elapsed = time.perf_counter() - t_total

    # Restore brain.nndf
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    # ── Analysis ─────────────────────────────────────────────────────────────

    # Collect delta_dx arrays by radius for aggregate cliff statistics.
    # max_delta_by_base tracks the worst cliff seen at each base point,
    # answering: "if I'm standing here, how bad could a tiny step be?"
    deltas_by_radius = {r: [] for r in RADII}
    delta_pl_by_radius = {r: [] for r in RADII}
    delta_speed_by_radius = {r: [] for r in RADII}
    max_delta_by_base = {r: [] for r in RADII}  # worst cliff per base per radius
    base_dxs = []
    base_pls = []

    for res in all_results:
        base_dxs.append(res["base_dx"])
        base_pls.append(res["base_phase_lock"])
        for radius in RADII:
            r_key = f"r{radius}"
            dd = [p["delta_dx"] for p in res["perturbations"][r_key]]
            dp = [p["delta_phase_lock"] for p in res["perturbations"][r_key]]
            ds = [p["delta_speed"] for p in res["perturbations"][r_key]]
            deltas_by_radius[radius].extend(dd)
            delta_pl_by_radius[radius].extend(dp)
            delta_speed_by_radius[radius].extend(ds)
            max_delta_by_base[radius].append(np.max(np.abs(dd)))

    base_dxs = np.array(base_dxs)
    base_pls = np.array(base_pls)

    # A "cliff" is a perturbation where |delta_DX| exceeds one of these
    # thresholds (in meters). Larger thresholds indicate more catastrophic
    # behavioral shifts from small weight changes.
    CLIFF_THRESHOLDS = [5, 10, 20]  # meters of DX change

    print(f"\n{'='*80}")
    print(f"CLIFF ANALYSIS ({NUM_BASE} base points, {total_elapsed:.1f}s)")
    print(f"{'='*80}")

    for radius in RADII:
        dd = np.abs(deltas_by_radius[radius])
        print(f"\n  Radius = {radius}:")
        print(f"    |delta_DX| — mean: {np.mean(dd):.2f}, median: {np.median(dd):.2f}, "
              f"P90: {np.percentile(dd, 90):.2f}, max: {np.max(dd):.2f}")
        print(f"    |delta_DX|/radius — mean: {np.mean(dd)/radius:.1f}, "
              f"P90: {np.percentile(dd, 90)/radius:.1f}")
        for thresh in CLIFF_THRESHOLDS:
            frac = np.mean(dd > thresh)
            print(f"    Cliff (|delta_DX|>{thresh}m): {frac*100:.1f}% of perturbations")
        # Fraction of BASE POINTS with at least one cliff
        md = np.array(max_delta_by_base[radius])
        for thresh in CLIFF_THRESHOLDS:
            frac_base = np.mean(md > thresh)
            print(f"    Base points with any cliff >{thresh}m: "
                  f"{frac_base*100:.1f}% ({int(np.sum(md > thresh))}/{NUM_BASE})")

    # Gradient magnitude (|delta_DX|/radius) by radius — if this increases
    # with radius, the landscape is locally nonlinear (super-linear response).
    print(f"\n  Gradient magnitude scaling:")
    for radius in RADII:
        dd = np.abs(deltas_by_radius[radius])
        grad = dd / radius
        print(f"    r={radius}: mean |grad|={np.mean(grad):.1f}, "
              f"median={np.median(grad):.1f}")

    # ── FIGURES ───────────────────────────────────────────────────────────────

    print("\nGenerating figures...")

    colors = {"0.05": "#4C72B0", "0.1": "#55A868", "0.2": "#C44E52"}

    # Fig 1: |delta_DX| distributions by radius
    fig, ax = plt.subplots(figsize=(8, 5))
    for radius in RADII:
        dd = np.abs(deltas_by_radius[radius])
        ax.hist(dd, bins=50, alpha=0.5, label=f"r={radius} (n={len(dd)})",
                color=colors[str(radius)], edgecolor="white")
    ax.set_xlabel("|delta DX| (meters)")
    ax.set_ylabel("Count")
    ax.set_title("Behavioral Change from Small Weight Perturbations")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cliff_fig01_delta_dx_by_radius.png")

    # Fig 2: Cliff probability vs radius for different thresholds
    fig, ax = plt.subplots(figsize=(7, 5))
    for thresh in CLIFF_THRESHOLDS:
        probs = []
        for radius in RADII:
            dd = np.abs(deltas_by_radius[radius])
            probs.append(np.mean(dd > thresh) * 100)
        ax.plot(RADII, probs, "o-", lw=2, markersize=8, label=f"|delta DX| > {thresh}m")
    ax.set_xlabel("Perturbation radius")
    ax.set_ylabel("Cliff probability (%)")
    ax.set_title("How Often Do Small Perturbations Cause Large Behavioral Shifts?")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cliff_fig02_cliff_probability.png")

    # Fig 3: Landscape profiles — for 8 selected base points, show DX vs perturbation.
    # Each line is a 1D transect along one random direction in 6D weight space,
    # plotting DX at the base point (radius=0) and at each perturbation radius.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # Pick 8 evenly spaced base points
    profile_indices = np.linspace(0, NUM_BASE - 1, 8, dtype=int)
    for ax_idx, bi in enumerate(profile_indices):
        ax = axes[ax_idx // 4][ax_idx % 4]
        res = all_results[bi]
        base_dx_val = res["base_dx"]

        # For each perturbation direction, we have the same direction at 3 radii
        # Plot: -r, 0, +r along that direction
        for d_idx in range(min(3, NUM_PERTURBATIONS)):  # show 3 directions
            xs = [0]
            ys = [base_dx_val]
            for radius in RADII:
                r_key = f"r{radius}"
                p_dx = res["perturbations"][r_key][d_idx]["dx"]
                xs.append(radius)
                ys.append(p_dx)
            ax.plot(xs, ys, "o-", lw=1, markersize=4, alpha=0.7)

        ax.axhline(base_dx_val, color="black", lw=0.5, ls=":")
        ax.set_title(f"Base {bi} (DX={base_dx_val:+.1f})", fontsize=9)
        ax.set_xlabel("Radius", fontsize=8)
        ax.set_ylabel("DX", fontsize=8)
        clean_ax(ax)

    fig.suptitle("Local Landscape Profiles (3 directions per base point)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "cliff_fig03_landscape_profiles.png")

    # Fig 4: Worst cliff at each base point vs base |DX|
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: worst cliff vs |base_DX|
    ax = axes[0]
    for radius in RADII:
        md = np.array(max_delta_by_base[radius])
        ax.scatter(np.abs(base_dxs), md, s=30, alpha=0.7,
                   color=colors[str(radius)], label=f"r={radius}")
    ax.set_xlabel("|Base DX| (meters)")
    ax.set_ylabel("Worst |delta DX| among neighbors")
    ax.set_title("Are High-Performance Points Near Cliffs?")
    ax.legend(fontsize=9)
    clean_ax(ax)

    # Right: worst cliff vs base phase lock
    ax = axes[1]
    for radius in RADII:
        md = np.array(max_delta_by_base[radius])
        ax.scatter(base_pls, md, s=30, alpha=0.7,
                   color=colors[str(radius)], label=f"r={radius}")
    ax.set_xlabel("Base Phase Lock Score")
    ax.set_ylabel("Worst |delta DX| among neighbors")
    ax.set_title("Are Phase-Locked Points Near Cliffs?")
    ax.legend(fontsize=9)
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "cliff_fig04_cliff_vs_base_dx.png")

    # Fig 5: Histogram of worst-cliff-per-base at each radius
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, radius in enumerate(RADII):
        ax = axes[idx]
        md = np.array(max_delta_by_base[radius])
        ax.hist(md, bins=20, color=colors[str(radius)], edgecolor="white", alpha=0.85)
        ax.axvline(np.median(md), color="black", ls="--", lw=1,
                   label=f"median={np.median(md):.1f}")
        frac_cliff = np.mean(md > 10) * 100
        ax.set_title(f"r={radius}  ({frac_cliff:.0f}% have cliff >10m)", fontsize=10)
        ax.set_xlabel("Worst |delta DX| among 10 neighbors")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        clean_ax(ax)
    fig.suptitle("Distribution of Worst Nearby Cliff per Base Point", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "cliff_fig05_worst_cliff_histogram.png")

    # ── Summary stats for the write-up ────────────────────────────────────────
    print(f"\nDone. {total_elapsed:.1f}s total ({total_sims} simulations).")


if __name__ == "__main__":
    main()
