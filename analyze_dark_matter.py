#!/usr/bin/env python3
"""
analyze_dark_matter.py

Deep analysis of "dead" gaits — trials with |DX| < 1m from the 500
random-search trials. These gaits don't go anywhere, but they're not
doing nothing. Full telemetry reveals what they actually are: spinners,
rockers, vibrators, circlers, and the truly inert.

Outputs:
    artifacts/dark_matter.json
    artifacts/plots/dark_fig01_overview.png
    artifacts/plots/dark_fig02_xy_trajectories.png
    artifacts/plots/dark_fig03_clusters.png
    artifacts/plots/dark_fig04_joint_gallery.png
    artifacts/plots/dark_fig05_contact_patterns.png
    artifacts/plots/dark_fig06_phase_portraits.png

Usage:
    python3 analyze_dark_matter.py
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
OUT_JSON = PROJECT / "artifacts" / "dark_matter.json"
RS_JSON = PROJECT / "artifacts" / "random_search_500.json"

DEAD_THRESHOLD = 1.0  # meters


# ── Simulation ───────────────────────────────────────────────────────────────

def write_brain(weights):
    """Write a brain.nndf file from a weight dict (w03, w04, w13, etc.)."""
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        # 3 sensors (0,1,2) x 2 motors (3,4) = 6 synapses
        for s in [0, 1, 2]:
            for m in [3, 4]:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def simulate_full(weights):
    """Run simulation, return full telemetry data dict."""
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

    # Pre-allocate arrays for every telemetry channel
    t_arr = np.empty(n_steps)
    x = np.empty(n_steps); y = np.empty(n_steps); z = np.empty(n_steps)
    vx = np.empty(n_steps); vy = np.empty(n_steps); vz = np.empty(n_steps)
    wx = np.empty(n_steps); wy = np.empty(n_steps); wz = np.empty(n_steps)
    roll = np.empty(n_steps); pitch = np.empty(n_steps); yaw = np.empty(n_steps)
    ct = np.empty(n_steps, dtype=bool)   # torso contact
    cb = np.empty(n_steps, dtype=bool)   # back leg contact
    cf = np.empty(n_steps, dtype=bool)   # front leg contact
    j0p = np.empty(n_steps); j0v = np.empty(n_steps); j0t = np.empty(n_steps)
    j1p = np.empty(n_steps); j1v = np.empty(n_steps); j1t = np.empty(n_steps)

    # Build name-to-index maps for links and joints (handles bytes vs str)
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
        roll[i] = rpy_vals[0]; pitch[i] = rpy_vals[1]; yaw[i] = rpy_vals[2]

        # Determine which links are touching the ground this step.
        # cp[3] is the link index on body A; -1 means the base link (Torso).
        contact_pts = p.getContactPoints(robotId)
        tc = bc = fc = False
        for cp in contact_pts:
            li = cp[3]
            if li == -1: tc = True
            elif li == back_li: bc = True
            elif li == front_li: fc = True
        ct[i] = tc; cb[i] = bc; cf[i] = fc

        # Joint state: [0]=position, [1]=velocity, [3]=applied torque
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
    }


# ── Extended behavioral descriptors ─────────────────────────────────────────

def compute_dark_descriptors(data):
    """Compute extended behavioral descriptors for dead-gait analysis."""
    x, y, z = data["x"], data["y"], data["z"]
    vx, vy = data["vx"], data["vy"]
    wx, wy, wz = data["wx"], data["wy"], data["wz"]
    j0p, j1p = data["j0_pos"], data["j1_pos"]
    j0t, j1t = data["j0_tau"], data["j0_vel"]
    n = len(x)  # number of simulation steps

    # Displacement
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    net_disp = np.sqrt(dx**2 + dy**2)

    # Path length (total distance traveled, not net)
    diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    path_length = float(np.sum(diffs))

    # Sinuosity: path_length / net_displacement (1.0 = straight line, high = winding)
    sinuosity = path_length / net_disp if net_disp > 0.01 else float('inf')

    # Max displacement from origin at any point
    dist_from_origin = np.sqrt(x**2 + y**2)
    max_excursion = float(np.max(dist_from_origin))

    # Speed statistics
    speed = np.sqrt(vx**2 + vy**2)
    mean_speed = float(np.mean(speed))

    # Rotation: total absolute yaw change (cumulative, not net)
    yaw_rate = wz
    total_yaw = float(np.sum(np.abs(yaw_rate)) * DT)
    yaw_net = float(np.trapezoid(wz, dx=DT))

    # Angular velocity magnitudes
    roll_rms = float(np.sqrt(np.mean(wx**2)))
    pitch_rms = float(np.sqrt(np.mean(wy**2)))
    yaw_rms = float(np.sqrt(np.mean(wz**2)))

    # Vertical (z) dynamics
    z_mean = float(np.mean(z))
    z_range = float(np.max(z) - np.min(z))
    z_std = float(np.std(z))

    # Joint oscillation
    j0_range = float(np.max(j0p) - np.min(j0p))
    j1_range = float(np.max(j1p) - np.min(j1p))
    j0_freq, j0_amp = _fft_peak(j0p, DT)
    j1_freq, j1_amp = _fft_peak(j1p, DT)

    # Energy
    power_j0 = np.abs(data["j0_tau"] * data["j0_vel"])
    power_j1 = np.abs(data["j1_tau"] * data["j1_vel"])
    work_total = float(np.sum(power_j0 + power_j1) * DT)

    # Contact
    ct = data["contact_torso"].astype(float)
    cb = data["contact_back"].astype(float)
    cf = data["contact_front"].astype(float)
    duty_torso = float(np.mean(ct))
    duty_back = float(np.mean(cb))
    duty_front = float(np.mean(cf))

    # Contact state entropy: encode 3 binary contact channels as a 3-bit
    # integer (0-7), then compute Shannon entropy over the 8 possible states.
    # High entropy = many distinct contact patterns; low = stuck in one state.
    state = ct.astype(int) * 4 + cb.astype(int) * 2 + cf.astype(int)
    counts = np.bincount(state.astype(int), minlength=8)
    probs = counts / n
    nonzero = probs[probs > 0]
    contact_entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    # Phase lock + heading metrics from Beer analytics
    analytics = compute_all(data, DT)
    phase_lock = analytics["coordination"]["phase_lock_score"]
    delta_phi = analytics["coordination"]["delta_phi_rad"]
    path_straightness = analytics["outcome"]["path_straightness"]
    heading_consistency = analytics["outcome"]["heading_consistency"]

    # XY trajectory shape: measure how circular the path is by computing
    # the coefficient of variation of radii from the centroid.
    # Low radius_cv = circular orbit; high = irregular trajectory.
    cx, cy = np.mean(x), np.mean(y)
    radii = np.sqrt((x - cx)**2 + (y - cy)**2)
    mean_radius = float(np.mean(radii))
    radius_cv = float(np.std(radii) / mean_radius) if mean_radius > 0.001 else 0.0

    return {
        "dx": dx, "dy": dy, "net_disp": net_disp,
        "path_length": path_length, "sinuosity": sinuosity,
        "max_excursion": max_excursion,
        "mean_speed": mean_speed,
        "total_yaw": total_yaw, "yaw_net": yaw_net,
        "roll_rms": roll_rms, "pitch_rms": pitch_rms, "yaw_rms": yaw_rms,
        "z_mean": z_mean, "z_range": z_range, "z_std": z_std,
        "j0_range": j0_range, "j1_range": j1_range,
        "j0_freq": j0_freq, "j0_amp": j0_amp,
        "j1_freq": j1_freq, "j1_amp": j1_amp,
        "work_total": work_total,
        "duty_torso": duty_torso, "duty_back": duty_back, "duty_front": duty_front,
        "contact_entropy": contact_entropy,
        "phase_lock": phase_lock, "delta_phi": delta_phi,
        "mean_radius": mean_radius, "radius_cv": radius_cv,
        "path_straightness": path_straightness,
        "heading_consistency": heading_consistency,
    }


# ── Classification ───────────────────────────────────────────────────────────

def classify_dark_gait(desc):
    """Heuristic classification of dead gait into behavioral type.

    Categories are tested in priority order; the first match wins.
    Returns one of: Frozen, Spinner, Circler, Rocker, Vibrator, Twitcher,
    Canceller, or Other.
    """
    # Frozen (truly inert): both joints barely move and body has near-zero
    # speed. These gaits do essentially nothing -- the NN outputs settle
    # to a fixed point immediately.
    if desc["j0_range"] < 0.05 and desc["j1_range"] < 0.05 and desc["mean_speed"] < 0.05:
        return "Frozen"

    # Spinner: accumulates large yaw rotation (>3 rad total) but goes
    # nowhere. Caused by asymmetric joint torques creating a net moment
    # about the vertical axis.
    if desc["total_yaw"] > 3.0 and desc["net_disp"] < 0.5:
        return "Spinner"

    # Circler: travels a significant path (>2m) and wanders far from the
    # origin (>0.5m excursion) but ends up near where it started. Unlike
    # Spinners, these actually translate through space in a loop.
    if desc["path_length"] > 2.0 and desc["max_excursion"] > 0.5 and desc["net_disp"] < 1.0:
        return "Circler"

    # Rocker: joints oscillate (>0.1 rad range) causing visible body
    # pitching (z_range > 0.05m) but almost no lateral translation.
    # The robot tips back and forth without going anywhere.
    if desc["j0_range"] > 0.1 and desc["mean_speed"] < 0.3 and desc["z_range"] > 0.05:
        return "Rocker"

    # Vibrator: at least one joint oscillates at high frequency (>2 Hz)
    # but the vibrations cancel out or are too small to produce movement.
    # Distinguished from Rockers by frequency rather than amplitude.
    if (desc["j0_freq"] > 2.0 or desc["j1_freq"] > 2.0) and desc["mean_speed"] < 0.2:
        return "Vibrator"

    # Twitcher: joints move (>0.05 rad) but translation is minimal.
    # Catch-all for gaits with visible motion that don't fit the
    # more specific categories above.
    if desc["j0_range"] > 0.05 and desc["mean_speed"] < 0.3:
        return "Twitcher"

    # Canceller: actually walks but reverses direction, ending near the
    # origin. Detected by high sinuosity (path_length >> net_displacement).
    if desc["path_length"] > 1.0 and desc["sinuosity"] > 5.0:
        return "Canceller"

    return "Other"


# ── Clustering (numpy k-means) ──────────────────────────────────────────────

def kmeans(data, k=6, max_iter=50):
    """Simple k-means. Returns labels array."""
    n = len(data)
    if n <= k:
        return np.arange(n)

    # Normalize
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0
    normed = (data - mins) / ranges

    indices = np.random.choice(n, k, replace=False)
    centroids = normed[indices].copy()

    for _ in range(max_iter):
        dists = np.zeros((n, k))
        for ci in range(k):
            dists[:, ci] = np.linalg.norm(normed - centroids[ci], axis=1)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(centroids)
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                new_centroids[ci] = normed[mask].mean(axis=0)
            else:
                new_centroids[ci] = centroids[ci]

        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    return labels


# ── Plot helpers ─────────────────────────────────────────────────────────────

def clean_ax(ax):
    """Remove top and right spines from a matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    """Save a figure to PLOT_DIR and close it to free memory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


TYPE_COLORS = {
    "Frozen": "#999999",
    "Spinner": "#C44E52",
    "Circler": "#4C72B0",
    "Rocker": "#55A868",
    "Vibrator": "#8172B2",
    "Twitcher": "#CCB974",
    "Canceller": "#DD8452",
    "Other": "#64B5CD",
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Re-simulate dead gaits, classify them, write JSON results, and generate figures."""
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    # Load 500 trials
    with open(RS_JSON) as f:
        all_trials = json.load(f)

    # Filter to dead gaits
    dead_trials = [t for t in all_trials if abs(t["dx"]) < DEAD_THRESHOLD]
    n_dead = len(dead_trials)
    print(f"Found {n_dead} dead gaits (|DX| < {DEAD_THRESHOLD}m) out of {len(all_trials)}")

    # Re-simulate with full telemetry
    print(f"Re-simulating {n_dead} dead gaits with full telemetry...")
    t_start = time.perf_counter()

    results = []
    telemetry_cache = []

    for i, trial in enumerate(dead_trials):
        data = simulate_full(trial["weights"])
        desc = compute_dark_descriptors(data)
        gait_type = classify_dark_gait(desc)

        results.append({
            "trial_idx": trial["trial"],
            "weights": trial["weights"],
            "type": gait_type,
            "descriptors": desc,
        })
        telemetry_cache.append(data)

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{n_dead}] last type: {gait_type}")

    elapsed = time.perf_counter() - t_start
    print(f"Done in {elapsed:.1f}s ({elapsed/n_dead:.3f}s/sim)")

    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # ── Census ───────────────────────────────────────────────────────────────
    type_counts = {}
    for r in results:
        t = r["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'='*60}")
    print("DARK MATTER CENSUS")
    print(f"{'='*60}")
    print(f"  Total dead gaits: {n_dead} / {len(all_trials)} ({n_dead/len(all_trials)*100:.1f}%)")
    print(f"\n  {'Type':<15} {'Count':>6} {'Fraction':>10}")
    print("  " + "-" * 35)
    for t in sorted(type_counts.keys(), key=lambda k: -type_counts[k]):
        print(f"  {t:<15} {type_counts[t]:>6} {type_counts[t]/n_dead*100:>9.1f}%")

    # ── Summary stats by type ────────────────────────────────────────────────
    print(f"\n  {'Type':<12} {'Speed':>8} {'TotYaw':>8} {'Work':>8} {'PathLen':>8} "
          f"{'PhasLk':>7} {'Strtns':>7} {'HdgCon':>7}")
    print("  " + "-" * 72)
    for gtype in sorted(type_counts.keys(), key=lambda k: -type_counts[k]):
        members = [r for r in results if r["type"] == gtype]
        avg = lambda key: np.mean([m["descriptors"][key] for m in members])
        print(f"  {gtype:<12} {avg('mean_speed'):>8.3f} {avg('total_yaw'):>8.2f} "
              f"{avg('work_total'):>8.0f} {avg('path_length'):>8.2f} "
              f"{avg('phase_lock'):>7.3f} {avg('path_straightness'):>7.3f} "
              f"{avg('heading_consistency'):>7.3f}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "n_total": len(all_trials),
            "n_dead": n_dead,
            "threshold": DEAD_THRESHOLD,
            "time_s": elapsed,
        },
        "census": type_counts,
        "gaits": [{
            "trial_idx": r["trial_idx"],
            "weights": r["weights"],
            "type": r["type"],
            "descriptors": r["descriptors"],
        } for r in results],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    # ── FIGURES ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    # ── Fig 1: Overview — scatter matrix of key descriptors ──────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    types = [r["type"] for r in results]
    colors = [TYPE_COLORS.get(t, "#999999") for t in types]

    plot_pairs = [
        ("mean_speed", "total_yaw", "Speed vs Total Yaw"),
        ("work_total", "path_length", "Energy vs Path Length"),
        ("path_straightness", "heading_consistency", "Path Straightness vs Heading Consistency"),
        ("yaw_rms", "heading_consistency", "Yaw RMS vs Heading Consistency"),
        ("phase_lock", "contact_entropy", "Phase Lock vs Contact Entropy"),
        ("max_excursion", "sinuosity", "Max Excursion vs Sinuosity"),
    ]

    for ax, (kx, ky, title) in zip(axes.flat, plot_pairs):
        xs = [r["descriptors"][kx] for r in results]
        ys_raw = [r["descriptors"][ky] for r in results]
        # Cap sinuosity at 100 for display; dead gaits can have infinite sinuosity
        if ky == "sinuosity":
            ys = [min(y, 100) for y in ys_raw]
        else:
            ys = ys_raw
        for gtype in TYPE_COLORS:
            mask = [i for i, t in enumerate(types) if t == gtype]
            if mask:
                ax.scatter([xs[i] for i in mask], [ys[i] for i in mask],
                           c=TYPE_COLORS[gtype], s=30, alpha=0.7, label=gtype,
                           edgecolors="white", linewidths=0.3)
        ax.set_xlabel(kx)
        ax.set_ylabel(ky)
        ax.set_title(title, fontsize=10)
        clean_ax(ax)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(TYPE_COLORS),
               fontsize=8, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"Dark Matter Overview: {n_dead} Dead Gaits (|DX| < {DEAD_THRESHOLD}m)",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    save_fig(fig, "dark_fig01_overview.png")

    # ── Fig 2: XY trajectories — gallery by type ────────────────────────────
    unique_types = sorted(type_counts.keys(), key=lambda k: -type_counts[k])
    n_types = len(unique_types)
    cols = min(n_types, 4)
    rows = (n_types + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, gtype in enumerate(unique_types):
        ax = axes[idx // cols][idx % cols]
        members = [(i, r) for i, r in enumerate(results) if r["type"] == gtype]

        # Plot up to 8 trajectories per type; fade later ones for legibility
        for mi, (ridx, r) in enumerate(members[:8]):
            data = telemetry_cache[ridx]
            alpha = 0.6 if mi < 4 else 0.3
            ax.plot(data["x"], data["y"], lw=0.8, alpha=alpha,
                    color=TYPE_COLORS.get(gtype, "#999999"))
        ax.scatter([0], [0], c="black", s=30, zorder=5, marker="o")
        ax.set_title(f"{gtype} (n={len(members)})", fontsize=10,
                     color=TYPE_COLORS.get(gtype, "black"))
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        clean_ax(ax)

    # Hide unused axes
    for idx in range(n_types, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("XY Trajectories by Dark Matter Type", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "dark_fig02_xy_trajectories.png")

    # ── Fig 3: PCA clustering ────────────────────────────────────────────────
    # Build feature matrix for PCA from 17 behavioral descriptors
    feature_keys = ["mean_speed", "total_yaw", "yaw_rms", "roll_rms", "pitch_rms",
                    "z_range", "j0_range", "j1_range", "j0_freq", "j1_freq",
                    "work_total", "phase_lock", "contact_entropy", "path_length",
                    "max_excursion", "path_straightness", "heading_consistency"]
    feat_matrix = np.array([[r["descriptors"][k] for k in feature_keys] for r in results])

    # Min-max normalize so each feature contributes equally to PCA
    mins = feat_matrix.min(axis=0)
    maxs = feat_matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0  # avoid division by zero for constant features
    normed = (feat_matrix - mins) / ranges

    # Manual PCA via eigendecomposition of the covariance matrix.
    # eigh returns eigenvalues in ascending order, so we reverse to get
    # largest-variance components first.
    mean = normed.mean(axis=0)
    centered = normed - mean
    cov = np.dot(centered.T, centered) / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_sorted = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx_sorted]
    eigvals = eigvals[idx_sorted]
    projected = centered @ eigvecs[:, :2]  # project onto top 2 principal components
    var_exp = eigvals[:2] / eigvals.sum() * 100  # % variance explained

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by heuristic type
    ax = axes[0]
    for gtype in unique_types:
        mask = [i for i, r in enumerate(results) if r["type"] == gtype]
        if mask:
            ax.scatter(projected[mask, 0], projected[mask, 1],
                       c=TYPE_COLORS.get(gtype, "#999999"), s=40, alpha=0.7,
                       label=gtype, edgecolors="white", linewidths=0.3)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
    ax.set_title("Heuristic Classification")
    ax.legend(fontsize=8)
    clean_ax(ax)

    # Right: colored by k-means
    k_labels = kmeans(feat_matrix, k=6)
    ax = axes[1]
    km_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#DD8452",
                 "#64B5CD", "#999999"]
    for ki in range(6):
        mask = k_labels == ki
        if np.any(mask):
            ax.scatter(projected[mask, 0], projected[mask, 1],
                       c=km_colors[ki % len(km_colors)], s=40, alpha=0.7,
                       label=f"Cluster {ki}", edgecolors="white", linewidths=0.3)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
    ax.set_title("K-means Clustering (k=6)")
    ax.legend(fontsize=8)
    clean_ax(ax)

    fig.suptitle("Dark Matter in Behavioral PCA Space", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "dark_fig03_clusters.png")

    # ── Fig 4: Joint angle gallery — one representative per type ─────────────
    fig, axes = plt.subplots(n_types, 2, figsize=(14, 3 * n_types))
    if n_types == 1:
        axes = axes[np.newaxis, :]

    for idx, gtype in enumerate(unique_types):
        members = [(i, r) for i, r in enumerate(results) if r["type"] == gtype]
        # Pick the member with median work as a representative example
        works = [r["descriptors"]["work_total"] for _, r in members]
        med_idx = np.argsort(works)[len(works) // 2]
        ridx, rep = members[med_idx]
        data = telemetry_cache[ridx]

        ax = axes[idx][0]
        ax.plot(data["t"], data["j0_pos"], color="#DD8452", lw=0.6, label="BackLeg")
        ax.plot(data["t"], data["j1_pos"], color="#8172B2", lw=0.6, label="FrontLeg")
        ax.set_title(f"{gtype} — Joint Angles (trial {rep['trial_idx']})",
                     fontsize=9, color=TYPE_COLORS.get(gtype, "black"))
        ax.set_ylabel("Angle (rad)")
        ax.legend(fontsize=7); clean_ax(ax)

        ax = axes[idx][1]
        ax.plot(data["t"], data["z"], color="#4C72B0", lw=0.5, alpha=0.8)
        ax2 = ax.twinx()
        speed = np.sqrt(data["vx"]**2 + data["vy"]**2)
        ax2.plot(data["t"], speed, color="#C44E52", lw=0.3, alpha=0.5)
        ax.set_title(f"{gtype} — Height (blue) & Speed (red)", fontsize=9)
        ax.set_ylabel("Z height (m)")
        ax2.set_ylabel("Speed (m/s)")
        clean_ax(ax)

    axes[-1][0].set_xlabel("Time (s)")
    axes[-1][1].set_xlabel("Time (s)")
    fig.suptitle("Dark Matter Gallery: Representative Gaits by Type", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "dark_fig04_joint_gallery.png")

    # ── Fig 5: Contact patterns — one per type ──────────────────────────────
    # Each row shows a stacked "raster" of ground contact for the 3 links
    # over time: filled = touching ground, blank = airborne. Rows are offset
    # vertically (0, 1, 2) so the three channels don't overlap.
    fig, axes = plt.subplots(n_types, 1, figsize=(14, 2 * n_types), sharex=True)
    if n_types == 1:
        axes = [axes]

    for idx, gtype in enumerate(unique_types):
        members = [(i, r) for i, r in enumerate(results) if r["type"] == gtype]
        # Pick the median-energy member as the representative
        works = [r["descriptors"]["work_total"] for _, r in members]
        med_idx = np.argsort(works)[len(works) // 2]
        ridx, rep = members[med_idx]
        data = telemetry_cache[ridx]

        ax = axes[idx]
        t_plot = data["t"]
        ct_v = data["contact_torso"].astype(float)
        cb_v = data["contact_back"].astype(float)
        cf_v = data["contact_front"].astype(float)
        # Each link's contact band is drawn in its own vertical lane
        # (0-0.9, 1-1.9, 2-2.9) so they stack without overlapping
        ax.fill_between(t_plot, 0, ct_v * 0.9, alpha=0.5, color="#999999")
        ax.fill_between(t_plot, 1, 1 + cb_v * 0.9, alpha=0.5, color="#DD8452")
        ax.fill_between(t_plot, 2, 2 + cf_v * 0.9, alpha=0.5, color="#8172B2")
        ax.set_yticks([0.45, 1.45, 2.45])
        ax.set_yticklabels(["Torso", "Back", "Front"])
        ax.set_title(f"{gtype} (trial {rep['trial_idx']})", fontsize=9,
                     color=TYPE_COLORS.get(gtype, "black"))
        clean_ax(ax)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Dark Matter Contact Patterns", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "dark_fig05_contact_patterns.png")

    # ── Fig 6: Phase portraits — one per type ────────────────────────────────
    # Phase portrait: plot joint0 angle vs joint1 angle over time.
    # A tight ellipse/cycle means phase-locked coordination; a filled blob
    # means uncorrelated joint motion. Points are colored by time (viridis)
    # to show trajectory evolution.
    cols_pp = min(n_types, 4)
    rows_pp = (n_types + cols_pp - 1) // cols_pp
    fig, axes = plt.subplots(rows_pp, cols_pp, figsize=(4 * cols_pp, 4 * rows_pp))
    # Normalize axes to 2D array regardless of grid dimensions
    if rows_pp == 1 and cols_pp == 1:
        axes = np.array([[axes]])
    elif rows_pp == 1:
        axes = axes[np.newaxis, :]
    elif cols_pp == 1:
        axes = axes[:, np.newaxis]

    for idx, gtype in enumerate(unique_types):
        ax = axes[idx // cols_pp][idx % cols_pp]
        members = [(i, r) for i, r in enumerate(results) if r["type"] == gtype]
        # Pick the median-energy member as the representative
        works = [r["descriptors"]["work_total"] for _, r in members]
        med_idx = np.argsort(works)[len(works) // 2]
        ridx, rep = members[med_idx]
        data = telemetry_cache[ridx]

        # Subsample to 500 points for the scatter overlay to avoid clutter
        scatter_idx = np.linspace(0, len(data["j0_pos"]) - 1, 500, dtype=int)
        # Faint line shows the full trajectory; scatter points add time coloring
        ax.plot(data["j0_pos"], data["j1_pos"], color=TYPE_COLORS.get(gtype, "#999999"),
                lw=0.3, alpha=0.3)
        ax.scatter(data["j0_pos"][scatter_idx], data["j1_pos"][scatter_idx],
                   c=scatter_idx, cmap="viridis", s=5, alpha=0.6)
        ax.set_title(f"{gtype}\nPL={rep['descriptors']['phase_lock']:.3f}", fontsize=9,
                     color=TYPE_COLORS.get(gtype, "black"))
        ax.set_xlabel("BackLeg (rad)")
        ax.set_ylabel("FrontLeg (rad)")
        clean_ax(ax)

    for idx in range(n_types, rows_pp * cols_pp):
        axes[idx // cols_pp][idx % cols_pp].set_visible(False)

    fig.suptitle("Dark Matter Phase Portraits (j0 vs j1, colored by time)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "dark_fig06_phase_portraits.png")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
