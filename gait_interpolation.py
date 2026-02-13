#!/usr/bin/env python3
"""
gait_interpolation.py

Gait Interpolation — What Lies Between Champions?

Linearly interpolate in 6D weight space between pairs of high-performing
gaits and simulate at each point. Map the fitness landscape along these
privileged transects — are champion-to-champion corridors smoother than
random directions?

Questions:
  1. Is the landscape between two champions a smooth valley, a cliff,
     or fractal noise?
  2. Are there intermediate gaits that outperform the endpoints?
  3. Do champion transects show any smoothness floor, or is the fractal
     structure universal?
  4. Is the interpolation path special compared to random directions?

Simulation budget: ~2,400 sims (~3 min)

Part 1: Pairwise Transects (6 pairs x 80 points = 480 sims)
Part 2: Grand Tour (200 points through 6 champions in sequence)
Part 3: Midpoint Probing (6 midpoints x 8 directions x 20 points = 960 sims)
Part 4: Transect Roughness vs Random Baseline (6 pairs x ~120 pts = 720 sims)

Outputs:
    artifacts/gait_interpolation.json
    artifacts/plots/interp_fig01_pairwise_transects.png
    artifacts/plots/interp_fig02_grand_tour.png
    artifacts/plots/interp_fig03_midpoint_landscape.png
    artifacts/plots/interp_fig04_roughness_comparison.png
    artifacts/plots/interp_fig05_weight_trajectories.png
    artifacts/plots/interp_fig06_verdict.png

Usage:
    python3 gait_interpolation.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as c
from pyrosim.neuralNetwork import NEURAL_NETWORK
import pyrosim.pyrosim as pyrosim

# ── Config ──────────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent
PLOT_DIR = PROJECT / "artifacts" / "plots"
OUT_JSON = PROJECT / "artifacts" / "gait_interpolation.json"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]

# ── Champion gaits (6-synapse standard topology) ────────────────────────────

CHAMPIONS = {
    "Novelty Champion": {
        "w03": -1.3083167156740476, "w04": -0.34279812804233867,
        "w13": 0.8331363773051514, "w14": -0.37582983217830773,
        "w23": -0.0369713954829298, "w24": 0.4375020967145814,
    },
    "Trial 3": {
        "w03": -0.5971393487736976, "w04": -0.4236677331634211,
        "w13": 0.11222931078528431, "w14": -0.004679977731207874,
        "w23": 0.2970146930268889, "w24": 0.21399448704946855,
    },
    "Pelton": {
        "w03": -0.3, "w04": 1.0, "w13": -1.0, "w14": 0.3,
        "w23": -0.3, "w24": 1.0,
    },
    "Curie": {
        "w03": -0.3, "w04": 0.9, "w13": -0.9, "w14": 0.3,
        "w23": -0.3, "w24": 0.9,
    },
    "Noether": {
        "w03": -0.7, "w04": 0.3, "w13": -0.5, "w14": 0.5,
        "w23": -0.3, "w24": 0.7,
    },
    "Original": {
        "w03": 1.0, "w04": -1.0, "w13": 1.0, "w14": -1.0,
        "w23": 1.0, "w24": -1.0,
    },
}

# Pairs to interpolate between (chosen for diversity)
PAIRS = [
    ("Novelty Champion", "Trial 3"),         # Both forward walkers, very different weights
    ("Novelty Champion", "Pelton"),           # Top performer → designed champion
    ("Novelty Champion", "Original"),         # Top performer → near-zero DX
    ("Pelton", "Curie"),                      # Very similar weights (nearby in weight space)
    ("Pelton", "Noether"),                    # Different directions of travel
    ("Trial 3", "Original"),                  # Random-search find → hand-designed
]

GRAND_TOUR_ORDER = ["Novelty Champion", "Pelton", "Curie", "Noether",
                     "Trial 3", "Original"]

N_INTERP = 80       # Points per pairwise transect
N_GRAND = 200       # Total points for grand tour
N_MIDPOINT_DIR = 8  # Directions at each midpoint
N_MIDPOINT_PTS = 20 # Points per midpoint direction
MIDPOINT_RADIUS = 0.15  # Radius of midpoint probes

RNG_SEED = 123


# ── Simulation ──────────────────────────────────────────────────────────────

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
    """Minimal sim — returns DX."""
    write_brain_standard(weights)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(c.DT)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    mu = float(getattr(c, "ROBOT_FRICTION", 2.5))
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


# ── Helpers ─────────────────────────────────────────────────────────────────

def weights_to_vec(w):
    return np.array([w[k] for k in WEIGHT_NAMES])


def vec_to_weights(v):
    return {WEIGHT_NAMES[i]: float(v[i]) for i in range(6)}


def interpolate_weights(w1, w2, t):
    """Linear interpolation: t=0 → w1, t=1 → w2."""
    v1 = weights_to_vec(w1)
    v2 = weights_to_vec(w2)
    v = v1 + t * (v2 - v1)
    return vec_to_weights(v)


def compute_roughness(dx_values, t_values):
    """
    Roughness = mean |second difference| of DX profile.
    Also returns max_step and sign_change_rate.
    """
    dxs = np.array(dx_values)
    n = len(dxs)
    if n < 3:
        return {"roughness": 0.0, "max_step": 0.0, "sign_change_rate": 0.0}

    first_diff = np.diff(dxs)
    second_diff = np.diff(first_diff)

    roughness = float(np.mean(np.abs(second_diff)))
    max_step = float(np.max(np.abs(first_diff)))

    signs = np.sign(first_diff)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    sign_change_rate = float(sign_changes / (n - 2))

    # Normalized roughness: divide by mean DX range per step
    dt_vals = np.diff(t_values)
    dx_per_t = np.abs(first_diff) / np.where(dt_vals > 0, dt_vals, 1)
    mean_gradient = float(np.mean(dx_per_t))

    return {
        "roughness": roughness,
        "max_step": max_step,
        "sign_change_rate": sign_change_rate,
        "mean_gradient": mean_gradient,
    }


def perpendicular_basis(direction, rng, n_dirs=8):
    """
    Generate n_dirs evenly-spaced directions in the plane
    perpendicular to `direction` (6D).
    """
    d = direction / (np.linalg.norm(direction) + EPS)
    # Find a vector not parallel to d
    basis = []
    for _ in range(100):
        v = rng.randn(6)
        v = v - np.dot(v, d) * d
        norm = np.linalg.norm(v)
        if norm > 0.1:
            v = v / norm
            # Check independence from existing basis vectors
            independent = True
            for b in basis:
                if abs(np.dot(v, b)) > 0.9:
                    independent = False
                    break
            if independent:
                basis.append(v)
                if len(basis) >= 2:
                    break

    if len(basis) < 2:
        # Fallback
        basis = [rng.randn(6) for _ in range(2)]
        for i, b in enumerate(basis):
            b = b - np.dot(b, d) * d
            for j in range(i):
                b = b - np.dot(b, basis[j]) * basis[j]
            basis[i] = b / (np.linalg.norm(b) + EPS)

    e1, e2 = basis[0], basis[1]
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dirs = []
    for angle in angles:
        v = np.cos(angle) * e1 + np.sin(angle) * e2
        v = v / (np.linalg.norm(v) + EPS)
        dirs.append(v)
    return dirs


# ── Plotting helpers ────────────────────────────────────────────────────────

def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def save_fig(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  WROTE {path}")


PAIR_COLORS = ["#E24A33", "#348ABD", "#988ED5", "#55A868", "#FBC15E", "#777777"]


# ── Figures ─────────────────────────────────────────────────────────────────

def fig01_pairwise_transects(transect_data):
    """6-panel gallery: DX profile along each pairwise interpolation."""
    n_pairs = len(transect_data)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle("Pairwise Champion Interpolations", fontsize=14, fontweight="bold")

    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

    for idx, (pair_key, pdata) in enumerate(transect_data.items()):
        ax = axes_flat[idx]
        t_vals = pdata["t_values"]
        dx_vals = pdata["dx_values"]
        name_a, name_b = pair_key.split(" → ")

        color = PAIR_COLORS[idx % len(PAIR_COLORS)]
        ax.plot(t_vals, dx_vals, color=color, lw=1.2, alpha=0.9)
        ax.fill_between(t_vals, dx_vals, alpha=0.15, color=color)

        # Mark endpoints
        ax.plot(0, dx_vals[0], "o", color="black", markersize=8, zorder=5)
        ax.plot(1, dx_vals[-1], "s", color="black", markersize=8, zorder=5)
        ax.annotate(name_a, (0, dx_vals[0]), textcoords="offset points",
                    xytext=(-5, 10), fontsize=7, ha="right")
        ax.annotate(name_b, (1, dx_vals[-1]), textcoords="offset points",
                    xytext=(5, 10), fontsize=7, ha="left")

        # Mark max and min along transect
        i_max = np.argmax(dx_vals)
        i_min = np.argmin(dx_vals)
        ax.plot(t_vals[i_max], dx_vals[i_max], "^", color="green", markersize=6, zorder=5)
        ax.plot(t_vals[i_min], dx_vals[i_min], "v", color="red", markersize=6, zorder=5)

        roughness = pdata["roughness"]
        ax.set_title(f"{name_a} → {name_b}\n"
                     f"roughness={roughness['roughness']:.1f}  "
                     f"max_step={roughness['max_step']:.1f}m  "
                     f"sign_flip={roughness['sign_change_rate']:.2f}",
                     fontsize=8)
        ax.set_xlabel("t  (0=A, 1=B)", fontsize=8)
        ax.set_ylabel("DX (m)", fontsize=8)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        clean_ax(ax)

    # Hide unused
    for idx in range(n_pairs, len(list(axes_flat))):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "interp_fig01_pairwise_transects.png")


def fig02_grand_tour(tour_data):
    """Grand tour: DX along the path visiting all champions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grand Tour Through Champion Space", fontsize=14, fontweight="bold")

    # Left: DX profile
    ax = axes[0]
    cum_t = tour_data["cumulative_t"]
    dxs = tour_data["dx_values"]
    ax.plot(cum_t, dxs, color="#E24A33", lw=1.2)
    ax.fill_between(cum_t, dxs, alpha=0.15, color="#E24A33")

    # Mark champion positions
    for ci, name in enumerate(GRAND_TOUR_ORDER):
        t_pos = ci / max(len(GRAND_TOUR_ORDER) - 1, 1)
        actual_t = t_pos * cum_t[-1]
        # Find nearest point
        idx = np.argmin(np.abs(np.array(cum_t) - actual_t))
        ax.plot(cum_t[idx], dxs[idx], "o", color="black", markersize=8, zorder=5)
        ax.annotate(name, (cum_t[idx], dxs[idx]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=7, ha="center", rotation=30)

    ax.set_xlabel("Position along tour", fontsize=9)
    ax.set_ylabel("DX (m)", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title("DX Along Grand Tour", fontsize=10)
    clean_ax(ax)

    # Right: weight trajectories
    ax = axes[1]
    weights_arr = np.array(tour_data["weight_vectors"])  # (N, 6)
    for wi, wname in enumerate(WEIGHT_NAMES):
        ax.plot(cum_t, weights_arr[:, wi], lw=1.2, label=wname, alpha=0.8)
    ax.set_xlabel("Position along tour", fontsize=9)
    ax.set_ylabel("Weight value", fontsize=9)
    ax.set_title("Weight Trajectories", fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "interp_fig02_grand_tour.png")


def fig03_midpoint_landscape(midpoint_data, transect_data):
    """Midpoint probes: 2x3 grid showing local landscape at transect midpoints."""
    n_mid = len(midpoint_data)
    n_cols = 3
    n_rows = (n_mid + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle("Midpoint Landscape Probes (r=0.15)", fontsize=14, fontweight="bold")
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

    for idx, (pair_key, mdata) in enumerate(midpoint_data.items()):
        ax = axes_flat[idx]

        # Plot transect DX in background
        tdata = transect_data[pair_key]
        ax.plot(tdata["t_values"], tdata["dx_values"], color="gray",
                lw=0.8, alpha=0.4, label="transect")

        # Midpoint marker
        mid_dx = mdata["midpoint_dx"]
        ax.axhline(mid_dx, color="black", lw=0.5, ls=":")

        # Plot each perpendicular direction
        for di, direction_data in enumerate(mdata["directions"]):
            r_vals = direction_data["r_values"]
            dx_vals = direction_data["dx_values"]
            # Map r_vals to a local x range around t=0.5
            local_t = 0.5 + np.array(r_vals) / MIDPOINT_RADIUS * 0.15
            color = plt.cm.hsv(di / N_MIDPOINT_DIR)
            ax.plot(local_t, dx_vals, color=color, lw=0.8, alpha=0.7)

        name_a, name_b = pair_key.split(" → ")
        perp_roughness = mdata["mean_perp_roughness"]
        trans_roughness = tdata["roughness"]["roughness"]
        ax.set_title(f"{name_a} → {name_b} midpoint\n"
                     f"perp roughness={perp_roughness:.1f}  "
                     f"transect roughness={trans_roughness:.1f}",
                     fontsize=8)
        ax.set_xlabel("position (0.5 = midpoint)", fontsize=8)
        ax.set_ylabel("DX (m)", fontsize=8)
        clean_ax(ax)

    for idx in range(n_mid, len(list(axes_flat))):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "interp_fig03_midpoint_landscape.png")


def fig04_roughness_comparison(transect_data, random_roughness):
    """Compare transect roughness to random-direction roughness."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Are Champion Transects Smoother Than Random?",
                 fontsize=14, fontweight="bold")

    # Collect transect roughness values
    trans_rough = []
    trans_labels = []
    trans_sign_rates = []
    trans_max_steps = []
    for pair_key, pdata in transect_data.items():
        r = pdata["roughness"]
        trans_rough.append(r["roughness"])
        trans_sign_rates.append(r["sign_change_rate"])
        trans_max_steps.append(r["max_step"])
        trans_labels.append(pair_key.replace(" → ", "\n→ "))

    # Collect random roughness
    rand_rough = [r["roughness"] for r in random_roughness]
    rand_sign_rates = [r["sign_change_rate"] for r in random_roughness]
    rand_max_steps = [r["max_step"] for r in random_roughness]

    # Left: roughness bar comparison
    ax = axes[0]
    ax.set_title("Mean |2nd Diff| (Roughness)", fontsize=10)
    x = np.arange(len(trans_labels))
    ax.barh(x, trans_rough, color=[PAIR_COLORS[i % len(PAIR_COLORS)]
            for i in range(len(trans_labels))], height=0.7, label="Transect")
    ax.axvline(np.mean(rand_rough), color="gray", lw=2, ls="--",
               label=f"Random mean={np.mean(rand_rough):.1f}")
    ax.axvline(np.median(rand_rough), color="gray", lw=1, ls=":",
               label=f"Random median={np.median(rand_rough):.1f}")
    ax.set_yticks(x)
    ax.set_yticklabels(trans_labels, fontsize=7)
    ax.set_xlabel("Roughness", fontsize=9)
    ax.legend(fontsize=7)
    ax.invert_yaxis()
    clean_ax(ax)

    # Center: sign change rate
    ax = axes[1]
    ax.set_title("Sign Change Rate", fontsize=10)
    ax.barh(x, trans_sign_rates, color=[PAIR_COLORS[i % len(PAIR_COLORS)]
            for i in range(len(trans_labels))], height=0.7)
    ax.axvline(np.mean(rand_sign_rates), color="gray", lw=2, ls="--",
               label=f"Random mean={np.mean(rand_sign_rates):.2f}")
    ax.set_yticks(x)
    ax.set_yticklabels(trans_labels, fontsize=7)
    ax.set_xlabel("Sign change rate", fontsize=9)
    ax.legend(fontsize=7)
    ax.invert_yaxis()
    clean_ax(ax)

    # Right: max step
    ax = axes[2]
    ax.set_title("Max Single Step (m)", fontsize=10)
    ax.barh(x, trans_max_steps, color=[PAIR_COLORS[i % len(PAIR_COLORS)]
            for i in range(len(trans_labels))], height=0.7)
    ax.axvline(np.mean(rand_max_steps), color="gray", lw=2, ls="--",
               label=f"Random mean={np.mean(rand_max_steps):.1f}")
    ax.set_yticks(x)
    ax.set_yticklabels(trans_labels, fontsize=7)
    ax.set_xlabel("Max step (m)", fontsize=9)
    ax.legend(fontsize=7)
    ax.invert_yaxis()
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "interp_fig04_roughness_comparison.png")


def fig05_weight_trajectories(transect_data):
    """Weight-space distance vs DX for all transects."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Weight-Space Geometry of Interpolation",
                 fontsize=14, fontweight="bold")

    # Left: |DX| vs weight distance from start
    ax = axes[0]
    ax.set_title("|DX| vs Weight Distance from A", fontsize=10)
    for idx, (pair_key, pdata) in enumerate(transect_data.items()):
        t_vals = np.array(pdata["t_values"])
        dx_vals = np.array(pdata["dx_values"])
        w_vecs = np.array(pdata["weight_vectors"])
        # Distance from start
        dists = np.linalg.norm(w_vecs - w_vecs[0], axis=1)
        color = PAIR_COLORS[idx % len(PAIR_COLORS)]
        label = pair_key.replace(" → ", "→")
        ax.plot(dists, dx_vals, color=color, lw=1, alpha=0.8, label=label)
    ax.set_xlabel("Weight distance from A", fontsize=9)
    ax.set_ylabel("DX (m)", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.legend(fontsize=6, ncol=2)
    clean_ax(ax)

    # Right: Scatter of all interpolated DX values
    ax = axes[1]
    ax.set_title("DX Distribution Along Transects", fontsize=10)
    all_dxs = []
    all_labels = []
    for pair_key, pdata in transect_data.items():
        all_dxs.append(pdata["dx_values"])
        all_labels.append(pair_key.replace(" → ", "\n→ "))
    bp = ax.boxplot(all_dxs, vert=True, patch_artist=True, showfliers=True,
                     flierprops=dict(markersize=2))
    for bi, box in enumerate(bp["boxes"]):
        box.set_facecolor(PAIR_COLORS[bi % len(PAIR_COLORS)])
        box.set_alpha(0.6)
    ax.set_xticklabels(all_labels, fontsize=6, rotation=0)
    ax.set_ylabel("DX (m)", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "interp_fig05_weight_trajectories.png")


def fig06_verdict(transect_data, random_roughness, midpoint_data):
    """Summary verdict panel."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Gait Interpolation Verdict", fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Top-left: overlay of all transects
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("All Transects Overlaid", fontsize=10)
    for idx, (pair_key, pdata) in enumerate(transect_data.items()):
        color = PAIR_COLORS[idx % len(PAIR_COLORS)]
        ax.plot(pdata["t_values"], pdata["dx_values"], color=color, lw=1, alpha=0.7)
    ax.set_xlabel("t (0=A, 1=B)", fontsize=8)
    ax.set_ylabel("DX (m)", fontsize=8)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    clean_ax(ax)

    # Top-center: roughness distribution comparison
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Roughness: Transect vs Random", fontsize=10)
    trans_r = [pdata["roughness"]["roughness"] for pdata in transect_data.values()]
    rand_r = [r["roughness"] for r in random_roughness]
    bins = np.linspace(0, max(max(trans_r), max(rand_r)) * 1.1, 20)
    ax.hist(rand_r, bins=bins, color="gray", alpha=0.5, label="Random dirs",
            density=True)
    for i, tr in enumerate(trans_r):
        ax.axvline(tr, color=PAIR_COLORS[i % len(PAIR_COLORS)], lw=2, alpha=0.8)
    ax.set_xlabel("Roughness", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)
    clean_ax(ax)

    # Top-right: best intermediate discovery
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Best Intermediate Gaits", fontsize=10)
    discoveries = []
    for pair_key, pdata in transect_data.items():
        dx_arr = np.array(pdata["dx_values"])
        endpoint_best = max(abs(dx_arr[0]), abs(dx_arr[-1]))
        interp_best_idx = np.argmax(np.abs(dx_arr))
        interp_best = abs(dx_arr[interp_best_idx])
        excess = interp_best - endpoint_best
        discoveries.append((pair_key, endpoint_best, interp_best, excess,
                           pdata["t_values"][interp_best_idx]))
    discoveries.sort(key=lambda x: x[3], reverse=True)
    labels = []
    excesses = []
    colors_list = []
    for i, (pk, ep, ip, ex, t_at) in enumerate(discoveries):
        labels.append(pk.replace(" → ", "\n→ "))
        excesses.append(ex)
        colors_list.append("#55A868" if ex > 0 else "#E24A33")
    ax.barh(range(len(labels)), excesses, color=colors_list, height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Excess |DX| over best endpoint (m)", fontsize=8)
    ax.axvline(0, color="black", lw=0.5)
    ax.invert_yaxis()
    clean_ax(ax)

    # Bottom-left: perp vs transect roughness at midpoints
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Midpoint: Perpendicular vs Transect", fontsize=10)
    perp_rough = []
    trans_rough_at_mid = []
    mid_labels = []
    for pair_key, mdata in midpoint_data.items():
        perp_rough.append(mdata["mean_perp_roughness"])
        trans_rough_at_mid.append(transect_data[pair_key]["roughness"]["roughness"])
        mid_labels.append(pair_key.replace(" → ", "→"))
    x = np.arange(len(mid_labels))
    w = 0.35
    ax.bar(x - w/2, trans_rough_at_mid, w, label="Along transect", color="#348ABD")
    ax.bar(x + w/2, perp_rough, w, label="Perpendicular", color="#E24A33")
    ax.set_xticks(x)
    ax.set_xticklabels(mid_labels, fontsize=6, rotation=20)
    ax.set_ylabel("Roughness", fontsize=8)
    ax.legend(fontsize=7)
    clean_ax(ax)

    # Bottom-center and right: verdict text
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis("off")

    # Compute verdict values
    mean_trans_rough = np.mean(trans_r)
    mean_rand_rough = np.mean(rand_r)
    ratio = mean_trans_rough / mean_rand_rough if mean_rand_rough > 0 else 0
    n_smoother = sum(1 for tr in trans_r if tr < mean_rand_rough)
    n_discoveries = sum(1 for _, _, _, ex, _ in discoveries if ex > 1.0)
    mean_perp = np.mean(perp_rough) if perp_rough else 0
    mean_along = np.mean(trans_rough_at_mid) if trans_rough_at_mid else 0

    verdict_lines = [
        "INTERPOLATION STRUCTURE",
        f"  Mean transect roughness: {mean_trans_rough:.1f}",
        f"  Mean random roughness: {mean_rand_rough:.1f}",
        f"  Ratio (transect/random): {ratio:.2f}",
        f"  Transects smoother than random: {n_smoother}/{len(trans_r)}",
        "",
        "INTERMEDIATE DISCOVERIES",
        f"  Transects with |DX| exceeding endpoints: {n_discoveries}/{len(discoveries)}",
    ]
    for pk, ep, ip, ex, t_at in discoveries:
        if ex > 1.0:
            verdict_lines.append(
                f"    {pk}: +{ex:.1f}m at t={t_at:.2f}")
    verdict_lines += [
        "",
        "MIDPOINT ISOTROPY",
        f"  Mean perpendicular roughness: {mean_perp:.1f}",
        f"  Mean transect roughness: {mean_along:.1f}",
        f"  Ratio (perp/transect): {mean_perp/mean_along:.2f}" if mean_along > 0 else "",
        "",
        "KEY FINDING",
    ]
    if ratio < 0.5:
        verdict_lines.append("  Champion transects are SMOOTHER than random —")
        verdict_lines.append("  there are privileged corridors in weight space.")
    elif ratio > 1.5:
        verdict_lines.append("  Champion transects are ROUGHER than random —")
        verdict_lines.append("  champions sit on cliff edges, not in valleys.")
    else:
        verdict_lines.append("  Champion transects are EQUALLY ROUGH as random —")
        verdict_lines.append("  no privileged paths exist. Fractal is universal.")

    ax.text(0.05, 0.95, "\n".join(verdict_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    fig.tight_layout()
    save_fig(fig, "interp_fig06_verdict.png")


# ── JSON encoder ────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 6)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.RandomState(RNG_SEED)

    total_sims = 0
    t_global = time.time()

    # Verify champion DX values
    print("Champion verification:")
    for name, weights in CHAMPIONS.items():
        dx = simulate_dx_only(weights)
        total_sims += 1
        print(f"  {name}: DX={dx:+.2f}m")
    print()

    # ── Part 1: Pairwise Transects ──────────────────────────────────────────
    print("=" * 80)
    print(f"PART 1: Pairwise Champion Transects ({len(PAIRS)} pairs x {N_INTERP} pts = "
          f"{len(PAIRS) * N_INTERP} sims)")
    print("=" * 80)
    t0 = time.time()

    transect_data = {}
    for pi, (name_a, name_b) in enumerate(PAIRS):
        t_start = time.time()
        w_a = CHAMPIONS[name_a]
        w_b = CHAMPIONS[name_b]
        t_vals = np.linspace(0, 1, N_INTERP)
        dx_vals = []
        w_vecs = []

        for t in t_vals:
            w = interpolate_weights(w_a, w_b, t)
            dx = simulate_dx_only(w)
            dx_vals.append(dx)
            w_vecs.append(weights_to_vec(w))
            total_sims += 1

        elapsed = time.time() - t_start
        roughness = compute_roughness(dx_vals, t_vals.tolist())
        pair_key = f"{name_a} → {name_b}"
        transect_data[pair_key] = {
            "t_values": t_vals.tolist(),
            "dx_values": dx_vals,
            "weight_vectors": [v.tolist() for v in w_vecs],
            "roughness": roughness,
            "name_a": name_a,
            "name_b": name_b,
        }

        dx_range = max(dx_vals) - min(dx_vals)
        print(f"  [{pi+1}/{len(PAIRS)}] {pair_key:45s} "
              f"DX range=[{min(dx_vals):+.1f}, {max(dx_vals):+.1f}]  "
              f"roughness={roughness['roughness']:.1f}  ({elapsed:.1f}s)")

    print(f"  Part 1 complete: {len(PAIRS) * N_INTERP} sims in {time.time()-t0:.1f}s")

    # ── Part 2: Grand Tour ──────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"PART 2: Grand Tour ({N_GRAND} points through {len(GRAND_TOUR_ORDER)} champions)")
    print("=" * 80)
    t0 = time.time()

    n_segments = len(GRAND_TOUR_ORDER) - 1
    pts_per_seg = N_GRAND // n_segments
    tour_dx = []
    tour_weights = []
    tour_t = []
    cumulative = 0.0

    for si in range(n_segments):
        name_a = GRAND_TOUR_ORDER[si]
        name_b = GRAND_TOUR_ORDER[si + 1]
        w_a = CHAMPIONS[name_a]
        w_b = CHAMPIONS[name_b]

        for pi in range(pts_per_seg):
            t = pi / max(pts_per_seg - 1, 1)
            w = interpolate_weights(w_a, w_b, t)
            dx = simulate_dx_only(w)
            tour_dx.append(dx)
            tour_weights.append(weights_to_vec(w).tolist())
            tour_t.append(cumulative + t / n_segments)
            total_sims += 1

        cumulative += 1.0 / n_segments
        print(f"  Segment {si+1}/{n_segments}: {name_a} → {name_b}  ({pts_per_seg} pts)")

    tour_data = {
        "cumulative_t": tour_t,
        "dx_values": tour_dx,
        "weight_vectors": tour_weights,
        "order": GRAND_TOUR_ORDER,
    }
    print(f"  Part 2 complete: {N_GRAND} sims in {time.time()-t0:.1f}s")

    # ── Part 3: Midpoint Probing ────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"PART 3: Midpoint Probing ({len(PAIRS)} midpoints x {N_MIDPOINT_DIR} dirs x "
          f"{N_MIDPOINT_PTS} pts = {len(PAIRS) * N_MIDPOINT_DIR * N_MIDPOINT_PTS} sims)")
    print("=" * 80)
    t0 = time.time()

    midpoint_data = {}
    for pi, (name_a, name_b) in enumerate(PAIRS):
        w_a = CHAMPIONS[name_a]
        w_b = CHAMPIONS[name_b]
        midpoint_w = interpolate_weights(w_a, w_b, 0.5)
        mid_vec = weights_to_vec(midpoint_w)
        mid_dx = simulate_dx_only(midpoint_w)
        total_sims += 1

        # Direction along transect
        transect_dir = weights_to_vec(w_b) - weights_to_vec(w_a)
        transect_dir = transect_dir / (np.linalg.norm(transect_dir) + EPS)

        # Perpendicular directions
        perp_dirs = perpendicular_basis(transect_dir, rng, N_MIDPOINT_DIR)

        directions = []
        perp_roughnesses = []
        for di, pdir in enumerate(perp_dirs):
            r_vals = np.linspace(-MIDPOINT_RADIUS, MIDPOINT_RADIUS, N_MIDPOINT_PTS)
            dx_vals = []
            for r in r_vals:
                w_vec = mid_vec + r * pdir
                w = vec_to_weights(w_vec)
                dx = simulate_dx_only(w)
                dx_vals.append(dx)
                total_sims += 1
            roughness = compute_roughness(dx_vals, r_vals.tolist())
            perp_roughnesses.append(roughness["roughness"])
            directions.append({
                "r_values": r_vals.tolist(),
                "dx_values": dx_vals,
                "roughness": roughness,
            })

        pair_key = f"{name_a} → {name_b}"
        midpoint_data[pair_key] = {
            "midpoint_weights": midpoint_w,
            "midpoint_dx": mid_dx,
            "directions": directions,
            "mean_perp_roughness": float(np.mean(perp_roughnesses)),
        }
        print(f"  [{pi+1}/{len(PAIRS)}] {pair_key:45s} "
              f"mid DX={mid_dx:+.1f}m  perp_roughness={np.mean(perp_roughnesses):.1f}")

    print(f"  Part 3 complete: {len(PAIRS) * N_MIDPOINT_DIR * N_MIDPOINT_PTS + len(PAIRS)} "
          f"sims in {time.time()-t0:.1f}s")

    # ── Part 4: Random Baseline ─────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PART 4: Random Direction Baseline")
    print("=" * 80)
    t0 = time.time()

    # For each pair, simulate along a random direction from the midpoint
    # with same length as the transect, using same number of points
    random_roughness = []
    n_random_dirs = 20  # 20 random directions per pair origin
    n_pts_random = 40

    for pi, (name_a, name_b) in enumerate(PAIRS):
        w_a = CHAMPIONS[name_a]
        w_b = CHAMPIONS[name_b]
        start_vec = weights_to_vec(w_a)
        end_vec = weights_to_vec(w_b)
        transect_length = np.linalg.norm(end_vec - start_vec)

        for ri in range(n_random_dirs):
            rand_dir = rng.randn(6)
            rand_dir = rand_dir / (np.linalg.norm(rand_dir) + EPS)

            t_vals = np.linspace(0, 1, n_pts_random)
            dx_vals = []
            for t in t_vals:
                w_vec = start_vec + t * transect_length * rand_dir
                w = vec_to_weights(w_vec)
                dx = simulate_dx_only(w)
                dx_vals.append(dx)
                total_sims += 1

            roughness = compute_roughness(dx_vals, t_vals.tolist())
            random_roughness.append(roughness)

        print(f"  [{pi+1}/{len(PAIRS)}] {name_a} → {name_b}: "
              f"{n_random_dirs} random dirs, mean roughness="
              f"{np.mean([r['roughness'] for r in random_roughness[-n_random_dirs:]]):.1f}")

    print(f"  Part 4 complete: {len(PAIRS) * n_random_dirs * n_pts_random} sims "
          f"in {time.time()-t0:.1f}s")

    # ── Analysis ────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("INTERPOLATION ANALYSIS")
    print("=" * 80)

    # Roughness comparison
    trans_rough = [pdata["roughness"]["roughness"] for pdata in transect_data.values()]
    rand_rough = [r["roughness"] for r in random_roughness]
    ratio = np.mean(trans_rough) / np.mean(rand_rough) if np.mean(rand_rough) > 0 else 0

    print(f"\n  ROUGHNESS COMPARISON:")
    print(f"    Mean transect roughness: {np.mean(trans_rough):.2f}")
    print(f"    Mean random roughness:   {np.mean(rand_rough):.2f}")
    print(f"    Ratio (transect/random): {ratio:.3f}")
    print(f"    Transects smoother: {sum(1 for t in trans_rough if t < np.mean(rand_rough))}"
          f"/{len(trans_rough)}")

    # Intermediate discoveries
    print(f"\n  INTERMEDIATE DISCOVERIES:")
    for pair_key, pdata in transect_data.items():
        dx_arr = np.array(pdata["dx_values"])
        endpoint_max = max(abs(dx_arr[0]), abs(dx_arr[-1]))
        i_best = np.argmax(np.abs(dx_arr))
        interp_max = abs(dx_arr[i_best])
        excess = interp_max - endpoint_max
        if excess > 0.5:
            print(f"    {pair_key}: |DX| peak {interp_max:.1f}m at t={pdata['t_values'][i_best]:.2f} "
                  f"(+{excess:.1f}m over endpoints)")
        else:
            print(f"    {pair_key}: no intermediate peak above endpoints")

    # Sign change rates
    print(f"\n  SIGN CHANGE RATES:")
    for pair_key, pdata in transect_data.items():
        r = pdata["roughness"]
        print(f"    {pair_key}: {r['sign_change_rate']:.3f} "
              f"(max step: {r['max_step']:.1f}m)")
    print(f"    Random mean: {np.mean([r['sign_change_rate'] for r in random_roughness]):.3f}")

    # Midpoint isotropy
    perp_means = [mdata["mean_perp_roughness"] for mdata in midpoint_data.values()]
    print(f"\n  MIDPOINT PERPENDICULAR ROUGHNESS:")
    print(f"    Mean: {np.mean(perp_means):.2f}")
    print(f"    Transect mean: {np.mean(trans_rough):.2f}")

    # ── Figures ─────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    fig01_pairwise_transects(transect_data)
    fig02_grand_tour(tour_data)
    fig03_midpoint_landscape(midpoint_data, transect_data)
    fig04_roughness_comparison(transect_data, random_roughness)
    fig05_weight_trajectories(transect_data)
    fig06_verdict(transect_data, random_roughness, midpoint_data)

    # ── Save JSON ───────────────────────────────────────────────────────────
    json_out = {
        "champions": {n: {k: v for k, v in w.items()} for n, w in CHAMPIONS.items()},
        "pairs": [list(p) for p in PAIRS],
        "transects": transect_data,
        "grand_tour": tour_data,
        "midpoint_probes": {k: {kk: vv for kk, vv in v.items() if kk != "directions"}
                           for k, v in midpoint_data.items()},
        "random_roughness_stats": {
            "mean": float(np.mean(rand_rough)),
            "std": float(np.std(rand_rough)),
            "median": float(np.median(rand_rough)),
            "n": len(rand_rough),
        },
        "verdict": {
            "transect_random_ratio": ratio,
            "n_smoother": int(sum(1 for t in trans_rough if t < np.mean(rand_rough))),
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    elapsed_total = time.time() - t_global
    print(f"\nTotal: {total_sims} sims in {elapsed_total:.1f}s "
          f"({elapsed_total/60:.1f} min, {total_sims/elapsed_total:.0f} sims/s)")


if __name__ == "__main__":
    main()
