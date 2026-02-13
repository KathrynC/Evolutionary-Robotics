#!/usr/bin/env python3
"""
atlas_cliffiness.py

Atlas of Cliffiness in Gaitspace — comprehensive spatial atlas showing
where cliffs are, what predicts them, and which weight dimensions are
the cliffiest in the 6D fitness landscape of the 3-link walker.

Simulation budget: ~6,400 sims (~10 minutes)

Part 1: Cliffiness Probing (3,000 sims)
    Load 500 base points from random_search_500.json, probe 6 random
    directions at r=0.05 each. Reconstruct full gradient vector via
    np.linalg.solve (6 probes in 6D = unique solution).

Part 2: 2D Slice Heatmaps (3,200 sims)
    Two 40x40 grids through Novelty Champion:
    Slice 1: w23 vs w13 (cliffiest synapse + key back-leg input)
    Slice 2: w03 vs w24 (largest weight + cross-leg synapse)

Part 3: Cliff Anatomy (200 sims)
    Top 10 cliffiest points: 20-point DX profile along worst cliff
    direction (radii -0.2 to +0.2).

Outputs:
    artifacts/atlas_cliffiness.json
    artifacts/plots/atlas_fig01_scatter_pca.png
    artifacts/plots/atlas_fig02_cliff_vs_metrics.png
    artifacts/plots/atlas_fig03_slice_dx.png
    artifacts/plots/atlas_fig04_slice_cliff.png
    artifacts/plots/atlas_fig05_per_weight_cliff.png
    artifacts/plots/atlas_fig06_champion_context.png
    artifacts/plots/atlas_fig07_cliff_anatomy.png

Usage:
    python3 atlas_cliffiness.py
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
from compute_beer_analytics import NumpyEncoder

# ── Constants ────────────────────────────────────────────────────────────────

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]
WEIGHT_LABELS = {
    "w03": "Torso->Back",
    "w04": "Torso->Front",
    "w13": "BackLeg->Back",
    "w14": "BackLeg->Front",
    "w23": "FrontLeg->Back",
    "w24": "FrontLeg->Front",
}

NOVELTY_CHAMPION = {
    "w03": -1.3083167156740476,
    "w04": -0.34279812804233867,
    "w13": 0.8331363773051514,
    "w14": -0.37582983217830773,
    "w23": -0.0369713954829298,
    "w24": 0.4375020967145814,
}

TRIAL3 = {
    "w03": -0.5971393487736976,
    "w04": -0.4236677331634211,
    "w13": 0.11222931078528431,
    "w14": -0.004679977731207874,
    "w23": 0.2970146930268889,
    "w24": 0.21399448704946855,
}

R_PROBE = 0.05
N_GRID = 40
N_ANATOMY_POINTS = 20
N_TOP_CLIFFS = 10

IN_JSON = PROJECT / "artifacts" / "random_search_500.json"
OUT_JSON = PROJECT / "artifacts" / "atlas_cliffiness.json"
PLOT_DIR = PROJECT / "artifacts" / "plots"


# ── Simulation ───────────────────────────────────────────────────────────────

def write_brain_standard(weights):
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


def simulate_dx_only(weights):
    """Minimal sim loop — skips all telemetry recording, only returns DX."""
    write_brain_standard(weights)

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

    x_first = None
    for i in range(n_steps):
        for nName in nn.neurons:
            n_obj = nn.neurons[nName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes,
                                                n_obj.Get_Value(), max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes,
                                                p.POSITION_CONTROL,
                                                n_obj.Get_Value(), max_force)
        p.stepSimulation()
        nn.Update()

        if x_first is None:
            pos, _ = p.getBasePositionAndOrientation(robotId)
            x_first = pos[0]

    pos, _ = p.getBasePositionAndOrientation(robotId)
    x_last = pos[0]
    p.disconnect()
    return x_last - x_first


# ── Helpers ──────────────────────────────────────────────────────────────────

def random_direction_6d():
    """Return a random unit vector in 6D."""
    v = np.random.randn(6)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        v = np.ones(6)
        norm = np.linalg.norm(v)
    return v / norm


def perturb_weights(base_weights, direction, radius):
    """Return new weight dict = base + radius * direction."""
    w = {}
    for i, wn in enumerate(WEIGHT_NAMES):
        w[wn] = base_weights[wn] + radius * direction[i]
    return w


def correlation_r(x, y):
    """Pearson correlation coefficient."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mx, my = np.mean(x), np.mean(y)
    num = np.sum((x - mx) * (y - my))
    den = np.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2))
    return float(num / den) if den > 1e-12 else 0.0


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


def probe_gradient(weights, base_dx):
    """Probe 6 random directions at R_PROBE, reconstruct gradient via linalg.solve."""
    directions = np.array([random_direction_6d() for _ in range(6)])  # 6x6
    delta_dxs = np.empty(6)
    for k in range(6):
        pw = perturb_weights(weights, directions[k], R_PROBE)
        delta_dxs[k] = simulate_dx_only(pw) - base_dx
    try:
        grad = np.linalg.solve(directions, delta_dxs / R_PROBE)
    except np.linalg.LinAlgError:
        grad = np.zeros(6)
    return grad, delta_dxs, directions


# ── Part 1: Cliffiness Probing ──────────────────────────────────────────────

def probe_cliffiness(base_points):
    """For each base point, probe 6 directions and reconstruct gradient."""
    n = len(base_points)
    total_sims = n * 6
    print(f"\n{'='*80}")
    print(f"PART 1: Cliffiness Probing ({n} points x 6 probes = {total_sims} sims)")
    print(f"{'='*80}")

    results = []
    sim_count = 0
    t0 = time.perf_counter()

    for idx, bp in enumerate(base_points):
        base_w = bp["weights"]
        base_dx = bp["dx"]

        grad, delta_dxs, directions = probe_gradient(base_w, base_dx)
        sim_count += 6

        cliffiness = float(np.max(np.abs(delta_dxs)))
        grad_mag = float(np.linalg.norm(grad))
        partials = {wn: float(np.abs(grad[i])) for i, wn in enumerate(WEIGHT_NAMES)}

        results.append({
            "idx": idx,
            "base_dx": base_dx,
            "cliffiness": cliffiness,
            "gradient_magnitude": grad_mag,
            "gradient_vector": grad.tolist(),
            "partials": partials,
            "delta_dxs": delta_dxs.tolist(),
            "directions": directions.tolist(),
            "weights": base_w,
            "speed": bp.get("speed", 0),
            "work_proxy": bp.get("work_proxy", 0),
            "phase_lock": bp.get("phase_lock", 0),
            "entropy": bp.get("entropy", 0),
            "yaw_net_rad": bp.get("yaw_net_rad", 0),
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / sim_count
            remaining = rate * (total_sims - sim_count)
            print(f"  [{idx+1:3d}/{n}] {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining  "
                  f"cliff={cliffiness:.2f}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Part 1 complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")
    return results


# ── Part 2: 2D Slice Heatmaps ──────────────────────────────────────────────

def compute_slice(vary_keys, center_weights, x_range=(-1, 1), y_range=(-1, 1),
                  n_grid=N_GRID):
    """Compute a 2D grid of DX values, varying two weights."""
    k0, k1 = vary_keys
    x_vals = np.linspace(x_range[0], x_range[1], n_grid)
    y_vals = np.linspace(y_range[0], y_range[1], n_grid)
    dx_grid = np.empty((n_grid, n_grid))
    total = n_grid * n_grid

    print(f"    Slice: {k0} vs {k1} ({total} sims)")
    t0 = time.perf_counter()
    sim_count = 0

    for i, v0 in enumerate(x_vals):
        for j, v1 in enumerate(y_vals):
            w = dict(center_weights)
            w[k0] = v0
            w[k1] = v1
            dx_grid[i, j] = simulate_dx_only(w)
            sim_count += 1

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / sim_count
            remaining = rate * (total - sim_count)
            print(f"      row {i+1}/{n_grid}, {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"    Slice done: {total} sims in {elapsed:.1f}s")

    # Compute cliffiness (gradient magnitude from 4-neighbor finite differences)
    step_x = (x_range[1] - x_range[0]) / (n_grid - 1)
    step_y = (y_range[1] - y_range[0]) / (n_grid - 1)
    cliff_grid = np.zeros((n_grid, n_grid))

    for i in range(n_grid):
        for j in range(n_grid):
            if i == 0:
                gx = (dx_grid[1, j] - dx_grid[0, j]) / step_x
            elif i == n_grid - 1:
                gx = (dx_grid[-1, j] - dx_grid[-2, j]) / step_x
            else:
                gx = (dx_grid[i+1, j] - dx_grid[i-1, j]) / (2 * step_x)

            if j == 0:
                gy = (dx_grid[i, 1] - dx_grid[i, 0]) / step_y
            elif j == n_grid - 1:
                gy = (dx_grid[i, -1] - dx_grid[i, -2]) / step_y
            else:
                gy = (dx_grid[i, j+1] - dx_grid[i, j-1]) / (2 * step_y)

            cliff_grid[i, j] = np.sqrt(gx**2 + gy**2)

    return {
        "vary_keys": vary_keys,
        "x_vals": x_vals.tolist(),
        "y_vals": y_vals.tolist(),
        "dx_grid": dx_grid.tolist(),
        "cliff_grid": cliff_grid.tolist(),
        "center_pos": (float(center_weights[k0]), float(center_weights[k1])),
    }


def compute_slices():
    """Compute both 2D slices through the Novelty Champion."""
    total = 2 * N_GRID * N_GRID
    print(f"\n{'='*80}")
    print(f"PART 2: 2D Slice Heatmaps (2 x {N_GRID}x{N_GRID} = {total} sims)")
    print(f"{'='*80}")

    # Slice 1: w23 vs w13, range [-1, 1] for both
    slice1 = compute_slice(["w23", "w13"], NOVELTY_CHAMPION)

    # Slice 2: w03 vs w24 — extend w03 range to [-1.5, 1] to include NC at w03=-1.308
    slice2 = compute_slice(["w03", "w24"], NOVELTY_CHAMPION,
                           x_range=(-1.5, 1), y_range=(-1, 1))

    return [slice1, slice2]


# ── Part 3: Cliff Anatomy ──────────────────────────────────────────────────

def cliff_anatomy(probe_results, n_top=N_TOP_CLIFFS, n_points=N_ANATOMY_POINTS):
    """Profile DX along worst cliff direction for top N cliffiest points."""
    total_sims = n_top * n_points
    print(f"\n{'='*80}")
    print(f"PART 3: Cliff Anatomy ({n_top} points x {n_points} radii = {total_sims} sims)")
    print(f"{'='*80}")

    sorted_by_cliff = sorted(probe_results, key=lambda x: x["cliffiness"],
                              reverse=True)
    top = sorted_by_cliff[:n_top]

    radii = np.linspace(-0.2, 0.2, n_points)
    profiles = []
    t0 = time.perf_counter()
    sim_count = 0

    for rank, pt in enumerate(top):
        grad = np.array(pt["gradient_vector"])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            cliff_dir = grad / grad_norm
        else:
            cliff_dir = random_direction_6d()

        base_w = pt["weights"]
        dxs = np.empty(n_points)

        for k, r in enumerate(radii):
            pw = perturb_weights(base_w, cliff_dir, r)
            dxs[k] = simulate_dx_only(pw)
            sim_count += 1

        profiles.append({
            "rank": rank,
            "idx": pt["idx"],
            "base_dx": pt["base_dx"],
            "cliffiness": pt["cliffiness"],
            "gradient_magnitude": pt["gradient_magnitude"],
            "cliff_direction": cliff_dir.tolist(),
            "radii": radii.tolist(),
            "dxs": dxs.tolist(),
        })

        elapsed = time.perf_counter() - t0
        print(f"  [{rank+1}/{n_top}] base_DX={pt['base_dx']:+.1f}  "
              f"cliff={pt['cliffiness']:.1f}  "
              f"DX range={dxs.min():+.1f} to {dxs.max():+.1f}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Part 3 complete: {sim_count} sims in {elapsed:.1f}s")
    return profiles


# ── Figures ──────────────────────────────────────────────────────────────────

def make_figures(probe_results, slices, anatomy, champion_probe):
    """Generate all 7 figures."""
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")

    cliffiness = np.array([r["cliffiness"] for r in probe_results])
    dx = np.array([r["base_dx"] for r in probe_results])
    abs_dx = np.abs(dx)
    speed = np.array([r["speed"] for r in probe_results])
    work = np.array([r["work_proxy"] for r in probe_results])
    phase_lock = np.array([r["phase_lock"] for r in probe_results])
    entropy = np.array([r["entropy"] for r in probe_results])
    abs_yaw = np.abs(np.array([r["yaw_net_rad"] for r in probe_results]))

    W = np.array([[r["weights"][wn] for wn in WEIGHT_NAMES] for r in probe_results])

    nc_w = np.array([NOVELTY_CHAMPION[wn] for wn in WEIGHT_NAMES])
    t3_w = np.array([TRIAL3[wn] for wn in WEIGHT_NAMES])

    # ── Fig 1: PCA scatter ──────────────────────────────────────────────────
    # Manual PCA via covariance eigendecomposition
    W_mean = W.mean(axis=0)
    W_centered = W - W_mean
    cov = np.dot(W_centered.T, W_centered) / max(len(W) - 1, 1)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    order = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = eig_vecs[:, order]

    pc = W_centered @ eig_vecs[:, :2]
    var_explained = eig_vals[:2] / eig_vals.sum() * 100

    nc_pc = (nc_w - W_mean) @ eig_vecs[:, :2]
    t3_pc = (t3_w - W_mean) @ eig_vecs[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=cliffiness, cmap="inferno",
                    s=15, alpha=0.7, edgecolors="none")
    ax.scatter(*nc_pc, marker="*", s=200, c="cyan", edgecolors="black",
               linewidths=1, zorder=5, label="Novelty Champion")
    ax.scatter(*t3_pc, marker="^", s=120, c="lime", edgecolors="black",
               linewidths=1, zorder=5, label="Trial 3")
    plt.colorbar(sc, ax=ax, label="Cliffiness (max |delta DX|)")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)")
    ax.set_title("Weight Space PCA — Colored by Cliffiness")
    ax.legend(fontsize=8)
    clean_ax(ax)

    ax = axes[1]
    vmax = np.percentile(abs_dx, 95)
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=dx, cmap="RdBu_r",
                    s=15, alpha=0.7, edgecolors="none",
                    vmin=-vmax, vmax=vmax)
    ax.scatter(*nc_pc, marker="*", s=200, c="cyan", edgecolors="black",
               linewidths=1, zorder=5, label="Novelty Champion")
    ax.scatter(*t3_pc, marker="^", s=120, c="lime", edgecolors="black",
               linewidths=1, zorder=5, label="Trial 3")
    plt.colorbar(sc, ax=ax, label="DX (meters)")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)")
    ax.set_title("Weight Space PCA — Colored by DX")
    ax.legend(fontsize=8)
    clean_ax(ax)

    fig.suptitle("Atlas of Cliffiness: PCA of 500 Random Weight Points", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "atlas_fig01_scatter_pca.png")

    # ── Fig 2: Cliffiness vs metrics ────────────────────────────────────────
    metrics = [
        (abs_dx, "|DX|", "m"),
        (speed, "Speed", "m/s"),
        (work, "Work Proxy", ""),
        (phase_lock, "Phase Lock", ""),
        (entropy, "Entropy", "bits"),
        (abs_yaw, "|Yaw|", "rad"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax_idx, (metric, label, unit) in enumerate(metrics):
        ax = axes[ax_idx // 3][ax_idx % 3]
        ax.scatter(metric, cliffiness, s=10, alpha=0.5, c="#4C72B0",
                   edgecolors="none")
        r = correlation_r(metric, cliffiness)
        xlabel = f"{label}" + (f" ({unit})" if unit else "")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cliffiness (max |delta DX|)")
        ax.set_title(f"r = {r:+.3f}")
        if abs(r) > 0.05:
            z = np.polyfit(metric, cliffiness, 1)
            x_line = np.linspace(metric.min(), metric.max(), 100)
            ax.plot(x_line, z[0] * x_line + z[1], color="#C44E52",
                    lw=1.5, ls="--")
        clean_ax(ax)

    fig.suptitle("Cliffiness vs Behavioral Metrics (500 random points)",
                 fontsize=13)
    fig.tight_layout()
    save_fig(fig, "atlas_fig02_cliff_vs_metrics.png")

    # ── Fig 3: Slice DX heatmaps ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, sl in zip(axes, slices):
        k0, k1 = sl["vary_keys"]
        x_vals = np.array(sl["x_vals"])
        y_vals = np.array(sl["y_vals"])
        dx_g = np.array(sl["dx_grid"])
        cx, cy = sl["center_pos"]

        vmax_sl = np.percentile(np.abs(dx_g), 95)
        im = ax.imshow(dx_g.T, origin="lower", aspect="auto",
                       extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
                       cmap="RdBu_r", vmin=-vmax_sl, vmax=vmax_sl)
        ax.scatter([cx], [cy], marker="*", s=200, c="cyan",
                   edgecolors="black", linewidths=1, zorder=5,
                   label="Novelty Champion")
        plt.colorbar(im, ax=ax, label="DX (m)")
        ax.set_xlabel(k0)
        ax.set_ylabel(k1)
        ax.set_title(f"DX Landscape: {k0} vs {k1}")
        ax.legend(fontsize=8)

    fig.suptitle("2D Slices Through Novelty Champion (DX)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "atlas_fig03_slice_dx.png")

    # ── Fig 4: Slice cliffiness heatmaps ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, sl in zip(axes, slices):
        k0, k1 = sl["vary_keys"]
        x_vals = np.array(sl["x_vals"])
        y_vals = np.array(sl["y_vals"])
        cliff_g = np.array(sl["cliff_grid"])
        cx, cy = sl["center_pos"]

        im = ax.imshow(cliff_g.T, origin="lower", aspect="auto",
                       extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
                       cmap="inferno")
        ax.scatter([cx], [cy], marker="*", s=200, c="cyan",
                   edgecolors="black", linewidths=1, zorder=5,
                   label="Novelty Champion")
        plt.colorbar(im, ax=ax, label="Cliffiness (|grad DX|)")
        ax.set_xlabel(k0)
        ax.set_ylabel(k1)
        ax.set_title(f"Cliffiness: {k0} vs {k1}")
        ax.legend(fontsize=8)

    fig.suptitle("2D Slices Through Novelty Champion (Cliffiness)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "atlas_fig04_slice_cliff.png")

    # ── Fig 5: Per-weight cliffiness ────────────────────────────────────────
    partials_matrix = np.array([[r["partials"][wn] for wn in WEIGHT_NAMES]
                                 for r in probe_results])  # (500, 6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors_box = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#DD8452",
                  "#937860"]
    bp_plot = ax.boxplot([partials_matrix[:, i] for i in range(6)],
                         tick_labels=WEIGHT_NAMES, patch_artist=True)
    for patch, color in zip(bp_plot["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("|dDX/dw_i|")
    ax.set_title("Per-Weight Sensitivity (500 random points)")
    clean_ax(ax)

    ax = axes[1]
    nc_partials = [champion_probe["nc"]["partials"][wn] for wn in WEIGHT_NAMES]
    t3_partials = [champion_probe["t3"]["partials"][wn] for wn in WEIGHT_NAMES]
    mean_partials = partials_matrix.mean(axis=0)

    x_pos = np.arange(6)
    width = 0.25
    ax.bar(x_pos - width, mean_partials, width, label="Population mean",
           color="#999999", alpha=0.8)
    ax.bar(x_pos, nc_partials, width, label="Novelty Champion",
           color="#C44E52", alpha=0.8)
    ax.bar(x_pos + width, t3_partials, width, label="Trial 3",
           color="#55A868", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(WEIGHT_NAMES)
    ax.set_ylabel("|dDX/dw_i|")
    ax.set_title("Champion vs Population Sensitivity")
    ax.legend(fontsize=8)
    clean_ax(ax)

    fig.suptitle("Per-Weight Cliffiness Analysis", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "atlas_fig05_per_weight_cliff.png")

    # ── Fig 6: Champion context (zoomed slices) ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for col, sl in enumerate(slices):
        k0, k1 = sl["vary_keys"]
        x_vals = np.array(sl["x_vals"])
        y_vals = np.array(sl["y_vals"])
        dx_g = np.array(sl["dx_grid"])
        cliff_g = np.array(sl["cliff_grid"])
        cx, cy = sl["center_pos"]

        # Zoom: find grid indices within +-0.3 of champion
        zoom = 0.3
        xi = np.where((x_vals >= cx - zoom) & (x_vals <= cx + zoom))[0]
        yi = np.where((y_vals >= cy - zoom) & (y_vals <= cy + zoom))[0]

        if len(xi) < 3 or len(yi) < 3:
            ci = np.argmin(np.abs(x_vals - cx))
            cj = np.argmin(np.abs(y_vals - cy))
            xi = np.arange(max(0, ci - 5), min(len(x_vals), ci + 6))
            yi = np.arange(max(0, cj - 5), min(len(y_vals), cj + 6))

        dx_zoom = dx_g[np.ix_(xi, yi)]
        cliff_zoom = cliff_g[np.ix_(xi, yi)]
        extent_zoom = [x_vals[xi[0]], x_vals[xi[-1]],
                       y_vals[yi[0]], y_vals[yi[-1]]]

        # Top row: DX
        ax = axes[0][col]
        vmax_z = max(np.percentile(np.abs(dx_zoom), 95), 1)
        im = ax.imshow(dx_zoom.T, origin="lower", aspect="auto",
                       extent=extent_zoom, cmap="RdBu_r",
                       vmin=-vmax_z, vmax=vmax_z)
        ax.scatter([cx], [cy], marker="*", s=200, c="cyan",
                   edgecolors="black", linewidths=1, zorder=5)
        plt.colorbar(im, ax=ax, label="DX")
        ax.set_xlabel(k0)
        ax.set_ylabel(k1)
        ax.set_title(f"Zoomed DX: {k0} vs {k1}")

        # Bottom row: Cliffiness
        ax = axes[1][col]
        im = ax.imshow(cliff_zoom.T, origin="lower", aspect="auto",
                       extent=extent_zoom, cmap="inferno")
        ax.scatter([cx], [cy], marker="*", s=200, c="cyan",
                   edgecolors="black", linewidths=1, zorder=5)
        plt.colorbar(im, ax=ax, label="Cliffiness")
        ax.set_xlabel(k0)
        ax.set_ylabel(k1)
        ax.set_title(f"Zoomed Cliffiness: {k0} vs {k1}")

    fig.suptitle("Champion Neighborhood Detail", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "atlas_fig06_champion_context.png")

    # ── Fig 7: Cliff anatomy ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(19, 7))
    for i, prof in enumerate(anatomy):
        ax = axes[i // 5][i % 5]
        radii_arr = np.array(prof["radii"])
        dxs = np.array(prof["dxs"])
        base_dx_val = prof["base_dx"]

        ax.plot(radii_arr, dxs, "o-", color="#4C72B0", lw=1.5, markersize=4)
        ax.axhline(base_dx_val, color="gray", lw=0.8, ls=":")
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"#{prof['rank']+1} cliff={prof['cliffiness']:.1f}\n"
                     f"base DX={base_dx_val:+.1f}", fontsize=9)
        ax.set_xlabel("Radius along cliff dir", fontsize=8)
        ax.set_ylabel("DX (m)", fontsize=8)
        clean_ax(ax)

    fig.suptitle("Cliff Anatomy: DX Profiles Along Worst Direction (Top 10)",
                 fontsize=13)
    fig.tight_layout()
    save_fig(fig, "atlas_fig07_cliff_anatomy.png")


# ── Console Output ──────────────────────────────────────────────────────────

def print_analysis(probe_results, slices, champion_probe):
    """Print summary tables to console."""
    cliffiness = np.array([r["cliffiness"] for r in probe_results])
    grad_mag = np.array([r["gradient_magnitude"] for r in probe_results])
    dx = np.array([r["base_dx"] for r in probe_results])
    abs_dx = np.abs(dx)
    speed = np.array([r["speed"] for r in probe_results])
    work = np.array([r["work_proxy"] for r in probe_results])
    phase_lock = np.array([r["phase_lock"] for r in probe_results])
    entropy = np.array([r["entropy"] for r in probe_results])
    abs_yaw = np.abs(np.array([r["yaw_net_rad"] for r in probe_results]))

    print(f"\n{'='*80}")
    print("ATLAS OF CLIFFINESS — SUMMARY")
    print(f"{'='*80}")

    # Cliffiness summary stats
    print(f"\n  CLIFFINESS STATS (max |delta DX| at r={R_PROBE}):")
    print(f"    Mean:   {np.mean(cliffiness):.2f} m")
    print(f"    Median: {np.median(cliffiness):.2f} m")
    print(f"    P90:    {np.percentile(cliffiness, 90):.2f} m")
    print(f"    Max:    {np.max(cliffiness):.2f} m")

    # Cliff thresholds
    print(f"\n  CLIFF PREVALENCE:")
    for thresh in [5, 10, 20]:
        frac = np.mean(cliffiness > thresh) * 100
        count = int(np.sum(cliffiness > thresh))
        print(f"    >{thresh:2d}m cliff: {frac:.1f}% ({count}/{len(cliffiness)})")

    # Gradient magnitude
    print(f"\n  GRADIENT MAGNITUDE:")
    print(f"    Mean:   {np.mean(grad_mag):.1f}")
    print(f"    Median: {np.median(grad_mag):.1f}")
    print(f"    P90:    {np.percentile(grad_mag, 90):.1f}")
    print(f"    Max:    {np.max(grad_mag):.1f}")

    # Per-weight sensitivity ranking
    partials_matrix = np.array([[r["partials"][wn] for wn in WEIGHT_NAMES]
                                 for r in probe_results])
    mean_partials = partials_matrix.mean(axis=0)
    rank_order = np.argsort(mean_partials)[::-1]

    print(f"\n  PER-WEIGHT SENSITIVITY RANKING (mean |dDX/dw_i|):")
    print(f"    {'Rank':<6} {'Weight':<8} {'Label':<18} "
          f"{'Mean |dDX/dw|':>14} {'Median':>10} {'P90':>10}")
    print("    " + "-" * 68)
    for rank, idx in enumerate(rank_order):
        wn = WEIGHT_NAMES[idx]
        print(f"    {rank+1:<6} {wn:<8} {WEIGHT_LABELS[wn]:<18} "
              f"{mean_partials[idx]:14.2f} "
              f"{np.median(partials_matrix[:, idx]):10.2f} "
              f"{np.percentile(partials_matrix[:, idx], 90):10.2f}")

    # Correlation table
    metrics = [
        (abs_dx, "|DX|"),
        (speed, "Speed"),
        (work, "Work"),
        (phase_lock, "Phase Lock"),
        (entropy, "Entropy"),
        (abs_yaw, "|Yaw|"),
    ]
    print(f"\n  CLIFFINESS-vs-METRIC CORRELATIONS (Pearson r):")
    print(f"    {'Metric':<14} {'r':>8}")
    print("    " + "-" * 24)
    for metric, label in metrics:
        r = correlation_r(metric, cliffiness)
        print(f"    {label:<14} {r:+8.3f}")

    # Champion comparison
    print(f"\n  CHAMPION LOCAL CLIFFINESS:")
    for key, label in [("nc", "Novelty Champion"), ("t3", "Trial 3")]:
        cp = champion_probe[key]
        print(f"    {label}:")
        print(f"      Base DX:     {cp['base_dx']:+.2f} m")
        print(f"      Cliffiness:  {cp['cliffiness']:.2f} m  "
              f"(population median: {np.median(cliffiness):.2f})")
        print(f"      Gradient mag: {cp['gradient_magnitude']:.1f}  "
              f"(population median: {np.median(grad_mag):.1f})")
        pct = np.mean(cliffiness < cp["cliffiness"]) * 100
        print(f"      Percentile:  {pct:.0f}%")

    # Slice summary
    print(f"\n  SLICE SUMMARY:")
    for sl in slices:
        k0, k1 = sl["vary_keys"]
        dx_g = np.array(sl["dx_grid"])
        cliff_g = np.array(sl["cliff_grid"])
        print(f"    {k0} vs {k1}:")
        print(f"      DX range: [{dx_g.min():+.1f}, {dx_g.max():+.1f}]")
        print(f"      Cliffiness — mean: {cliff_g.mean():.1f}, "
              f"max: {cliff_g.max():.1f}, "
              f"P90: {np.percentile(cliff_g, 90):.1f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    np.random.seed(42)

    # Back up brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    # Load base points
    print(f"Loading {IN_JSON} ...")
    with open(IN_JSON) as f:
        base_points = json.load(f)
    print(f"  Loaded {len(base_points)} base points")

    # Sim budget
    n_bp = len(base_points)
    budget = (5 + n_bp * 6 + 14 + 2 * N_GRID * N_GRID + N_TOP_CLIFFS * N_ANATOMY_POINTS)
    print(f"  Total simulation budget: ~{budget} sims")

    # ── Determinism check ───────────────────────────────────────────────────
    print("\nDeterminism check (5 base points)...")
    check_indices = np.random.choice(len(base_points), 5, replace=False)
    max_err = 0
    for ci in check_indices:
        bp = base_points[ci]
        dx_sim = simulate_dx_only(bp["weights"])
        err = abs(dx_sim - bp["dx"])
        max_err = max(max_err, err)
        status = "OK" if err < 0.01 else "MISMATCH"
        print(f"  Trial {bp['trial']}: stored={bp['dx']:+.4f}  "
              f"sim={dx_sim:+.4f}  err={err:.6f}  [{status}]")
    if max_err > 0.01:
        print(f"  WARNING: max error {max_err:.6f} > 0.01")
    else:
        print(f"  All checks passed (max err={max_err:.6f})")

    # ── Part 1: Cliffiness Probing ──────────────────────────────────────────
    probe_results = probe_cliffiness(base_points)

    # ── Champion probes ─────────────────────────────────────────────────────
    print("\nProbing champions (NC + T3, 7 sims each)...")
    champion_probe = {}
    for key, weights, label in [("nc", NOVELTY_CHAMPION, "Novelty Champion"),
                                 ("t3", TRIAL3, "Trial 3")]:
        base_dx = simulate_dx_only(weights)
        grad, delta_dxs, directions = probe_gradient(weights, base_dx)

        champion_probe[key] = {
            "base_dx": float(base_dx),
            "cliffiness": float(np.max(np.abs(delta_dxs))),
            "gradient_magnitude": float(np.linalg.norm(grad)),
            "gradient_vector": grad.tolist(),
            "partials": {wn: float(np.abs(grad[i]))
                         for i, wn in enumerate(WEIGHT_NAMES)},
        }
        print(f"  {label}: DX={base_dx:+.2f}  "
              f"cliff={champion_probe[key]['cliffiness']:.2f}")

    # ── Part 2: 2D Slices ──────────────────────────────────────────────────
    slices = compute_slices()

    # ── Part 3: Cliff Anatomy ──────────────────────────────────────────────
    anatomy = cliff_anatomy(probe_results)

    # ── Restore brain.nndf ──────────────────────────────────────────────────
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # ── Console output ──────────────────────────────────────────────────────
    print_analysis(probe_results, slices, champion_probe)

    # ── Figures ─────────────────────────────────────────────────────────────
    make_figures(probe_results, slices, anatomy, champion_probe)

    # ── Save JSON ───────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "n_base_points": len(base_points),
            "r_probe": R_PROBE,
            "n_grid": N_GRID,
            "n_anatomy_points": N_ANATOMY_POINTS,
            "n_top_cliffs": N_TOP_CLIFFS,
            "weight_names": WEIGHT_NAMES,
        },
        "probe_results": probe_results,
        "champion_probe": champion_probe,
        "slices": slices,
        "anatomy": anatomy,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
