#!/usr/bin/env python3
"""
cliff_taxonomy.py

Cliff Taxonomy — Adaptive Probing & Shape Classification

Deeper probing of the cliffiest weight-space points discovered by atlas_cliffiness.py,
with quantitative shape features and rule-based cliff type classification.

Simulation budget: ~3,300 sims (~4 min)

Part A: Extended Cliff Profiles (1,500 sims)
    Top 50 cliffiest points, 30-point DX profile along gradient direction.

Part B: Perpendicular Profiles (600 sims)
    Top 30 cliffiest points, 20-point DX profile perpendicular to gradient.

Part C: Fine-Grained Cliff Edges (800 sims)
    Top 20 cliffiest points, 40-point DX profile along gradient (r=±0.05).

Part D: Multi-Scale Probing (360 sims)
    Top 20 cliffiest points, 6 random probes at 3 radii (0.01, 0.005, 0.001).

Outputs:
    artifacts/cliff_taxonomy.json
    artifacts/plots/tax_fig01_profile_gallery.png
    artifacts/plots/tax_fig02_taxonomy_summary.png
    artifacts/plots/tax_fig03_perpendicular.png
    artifacts/plots/tax_fig04_cliff_edges.png
    artifacts/plots/tax_fig05_feature_space.png
    artifacts/plots/tax_fig06_multiscale.png

Usage:
    python3 cliff_taxonomy.py
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
from matplotlib.patches import FancyBboxPatch

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

IN_JSON = PROJECT / "artifacts" / "atlas_cliffiness.json"
OUT_JSON = PROJECT / "artifacts" / "cliff_taxonomy.json"
PLOT_DIR = PROJECT / "artifacts" / "plots"

# Taxonomy type colors
TYPE_COLORS = {
    "Step": "#E24A33",
    "Precipice": "#348ABD",
    "Canyon": "#988ED5",
    "Slope": "#777777",
    "Fractal": "#FBC15E",
}

# ── Simulation (reused from atlas_cliffiness.py) ────────────────────────────

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


# ── Helpers (reused from atlas_cliffiness.py) ────────────────────────────────

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


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


# ── New Helpers ──────────────────────────────────────────────────────────────

def perpendicular_direction(grad):
    """Gram-Schmidt: random vector orthogonalized against grad, normalized."""
    v = np.random.randn(6)
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-12:
        return random_direction_6d()
    g_hat = grad / grad_norm
    v = v - np.dot(v, g_hat) * g_hat
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        # Degenerate case: try again with different random vector
        v = np.random.randn(6)
        v = v - np.dot(v, g_hat) * g_hat
        norm = np.linalg.norm(v)
    return v / norm


def compute_cliff_features(radii, dxs, base_dx):
    """Extract shape descriptors from a DX profile.

    Args:
        radii: array of radius values (negative = opposite direction)
        dxs: array of DX values at each radius
        base_dx: DX at the unperturbed point

    Returns:
        dict of feature values
    """
    radii = np.asarray(radii)
    dxs = np.asarray(dxs)
    n = len(dxs)

    # Total range of DX across profile
    dx_range = float(np.max(dxs) - np.min(dxs))
    if dx_range < 1e-12:
        dx_range = 1e-12

    # Asymmetry: range on negative side vs positive side
    neg_mask = radii < -1e-12
    pos_mask = radii > 1e-12
    neg_dxs = dxs[neg_mask] if np.any(neg_mask) else np.array([base_dx])
    pos_dxs = dxs[pos_mask] if np.any(pos_mask) else np.array([base_dx])
    neg_range = float(np.max(neg_dxs) - np.min(neg_dxs))
    pos_range = float(np.max(pos_dxs) - np.min(pos_dxs))
    denom = max(pos_range, 1e-12)
    asymmetry_index = float(neg_range / denom)

    # Max step: largest consecutive jump
    steps = np.abs(np.diff(dxs))
    max_step = float(np.max(steps)) if len(steps) > 0 else 0.0
    max_step_idx = int(np.argmax(steps)) if len(steps) > 0 else 0

    # Step location: radius where max step occurs (midpoint of the two samples)
    if len(steps) > 0:
        step_location = float((radii[max_step_idx] + radii[max_step_idx + 1]) / 2)
    else:
        step_location = 0.0

    # Transition width: radius range over which 80% of total DX change occurs
    sorted_dxs = np.sort(dxs)
    dx_min, dx_max = sorted_dxs[0], sorted_dxs[-1]
    threshold_lo = dx_min + 0.1 * (dx_max - dx_min)
    threshold_hi = dx_min + 0.9 * (dx_max - dx_min)
    in_transition = np.where((dxs >= threshold_lo) & (dxs <= threshold_hi))[0]
    if len(in_transition) >= 2:
        # Find the radii span where the transition happens
        transition_radii = radii[in_transition]
        transition_width = float(np.max(transition_radii) - np.min(transition_radii))
    else:
        transition_width = float(np.abs(radii[-1] - radii[0]))

    # Recovery ratio: how much DX recovers from cliff
    # base = DX at r=0 (or closest), cliff_bottom = min or max DX, far_end = DX at edges
    center_idx = np.argmin(np.abs(radii))
    dx_base = dxs[center_idx]

    # Determine cliff direction: which end has larger deviation from base?
    dx_first = dxs[0]
    dx_last = dxs[-1]
    dev_first = abs(dx_first - dx_base)
    dev_last = abs(dx_last - dx_base)

    if dev_first > dev_last:
        # Cliff is on negative side
        cliff_extreme = dx_first
        far_end = dx_last
    else:
        # Cliff is on positive side
        cliff_extreme = dx_last
        far_end = dx_first

    cliff_depth = abs(cliff_extreme - dx_base)
    if cliff_depth > 1e-12:
        recovery_ratio = float(abs(far_end - dx_base) / cliff_depth)
    else:
        recovery_ratio = 0.0
    recovery_ratio = min(recovery_ratio, 1.0)

    # Cliff polarity: sign of DX change at max step
    if len(steps) > 0:
        cliff_polarity = float(np.sign(dxs[max_step_idx + 1] - dxs[max_step_idx]))
    else:
        cliff_polarity = 0.0

    # Roughness: std of second differences
    if n >= 3:
        second_diffs = np.diff(dxs, n=2)
        roughness = float(np.std(second_diffs))
    else:
        roughness = 0.0

    # Sharpness ratio: max_step / total range
    sharpness = float(max_step / dx_range) if dx_range > 1e-12 else 0.0

    return {
        "asymmetry_index": round(asymmetry_index, 4),
        "max_step": round(max_step, 4),
        "step_location": round(step_location, 6),
        "transition_width": round(transition_width, 6),
        "recovery_ratio": round(recovery_ratio, 4),
        "cliff_polarity": cliff_polarity,
        "roughness": round(roughness, 4),
        "sharpness": round(sharpness, 4),
        "dx_range": round(dx_range, 4),
    }


def classify_cliff(features):
    """Rule-based taxonomy assignment.

    Types:
        Step:      sharp + one-sided + no recovery
        Precipice: sharp + one-sided + partial recovery
        Canyon:    sharp + two-sided + recovery
        Slope:     gradual (no discontinuity)
        Fractal:   ragged + multiple comparable steps
    """
    sharpness = features["sharpness"]
    asym = features["asymmetry_index"]
    rec = features["recovery_ratio"]
    roughness = features["roughness"]
    dx_range = features["dx_range"]

    # Roughness threshold: scale by range
    roughness_norm = roughness / max(dx_range, 1e-6)

    # Check for Fractal first: high roughness with no single dominant step
    if roughness_norm > 0.15 and sharpness < 0.5:
        return "Fractal"

    # Slope: gradual change
    if sharpness < 0.3:
        return "Slope"

    # Sharp cliff: sharpness >= 0.3
    one_sided = asym > 3 or asym < 1 / 3  # asymmetric in either direction

    if one_sided:
        if rec < 0.3:
            return "Step"
        elif rec < 0.7:
            return "Precipice"
        else:
            return "Canyon"
    else:
        # Two-sided
        if rec > 0.5:
            return "Canyon"
        elif rec < 0.3:
            return "Step"
        else:
            return "Precipice"


# ── Part A: Extended Cliff Profiles ──────────────────────────────────────────

def part_a_extended_profiles(sorted_probes, n_top=50, n_points=30):
    """30-point DX profile along gradient direction for top 50 cliffiest points."""
    total_sims = n_top * n_points
    print(f"\n{'='*80}")
    print(f"PART A: Extended Cliff Profiles ({n_top} points x {n_points} radii = {total_sims} sims)")
    print(f"{'='*80}")

    radii = np.linspace(-0.2, 0.2, n_points)
    profiles = []
    t0 = time.perf_counter()
    sim_count = 0

    for rank, pt in enumerate(sorted_probes[:n_top]):
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
            "gradient_vector": pt["gradient_vector"],
            "cliff_direction": cliff_dir.tolist(),
            "weights": pt["weights"],
            "radii": radii.tolist(),
            "dxs": dxs.tolist(),
        })

        if (rank + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / sim_count
            remaining = rate * (total_sims - sim_count)
            print(f"  [{rank+1:3d}/{n_top}] {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining  "
                  f"DX range=[{dxs.min():+.1f}, {dxs.max():+.1f}]", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Part A complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")
    return profiles


# ── Part B: Perpendicular Profiles ──────────────────────────────────────────

def part_b_perpendicular(sorted_probes, n_top=30, n_points=20):
    """20-point DX profile perpendicular to gradient for top 30 cliffiest points."""
    total_sims = n_top * n_points
    print(f"\n{'='*80}")
    print(f"PART B: Perpendicular Profiles ({n_top} points x {n_points} radii = {total_sims} sims)")
    print(f"{'='*80}")

    radii = np.linspace(-0.2, 0.2, n_points)
    perp_profiles = []
    t0 = time.perf_counter()
    sim_count = 0

    for rank, pt in enumerate(sorted_probes[:n_top]):
        grad = np.array(pt["gradient_vector"])
        perp_dir = perpendicular_direction(grad)

        base_w = pt["weights"]
        dxs = np.empty(n_points)

        for k, r in enumerate(radii):
            pw = perturb_weights(base_w, perp_dir, r)
            dxs[k] = simulate_dx_only(pw)
            sim_count += 1

        perp_profiles.append({
            "rank": rank,
            "idx": pt["idx"],
            "base_dx": pt["base_dx"],
            "cliffiness": pt["cliffiness"],
            "perp_direction": perp_dir.tolist(),
            "radii": radii.tolist(),
            "dxs": dxs.tolist(),
        })

        if (rank + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / sim_count
            remaining = rate * (total_sims - sim_count)
            print(f"  [{rank+1:3d}/{n_top}] {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining  "
                  f"DX range=[{dxs.min():+.1f}, {dxs.max():+.1f}]", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Part B complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")
    return perp_profiles


# ── Part C: Fine-Grained Cliff Edges ────────────────────────────────────────

def part_c_fine_grained(sorted_probes, n_top=20, n_points=40):
    """40-point DX profile along gradient, zoomed to r=±0.05."""
    total_sims = n_top * n_points
    print(f"\n{'='*80}")
    print(f"PART C: Fine-Grained Cliff Edges ({n_top} points x {n_points} radii = {total_sims} sims)")
    print(f"{'='*80}")

    radii = np.linspace(-0.05, 0.05, n_points)
    fine_profiles = []
    t0 = time.perf_counter()
    sim_count = 0

    for rank, pt in enumerate(sorted_probes[:n_top]):
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

        fine_profiles.append({
            "rank": rank,
            "idx": pt["idx"],
            "base_dx": pt["base_dx"],
            "cliffiness": pt["cliffiness"],
            "cliff_direction": cliff_dir.tolist(),
            "radii": radii.tolist(),
            "dxs": dxs.tolist(),
        })

        if (rank + 1) % 5 == 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / sim_count
            remaining = rate * (total_sims - sim_count)
            print(f"  [{rank+1:3d}/{n_top}] {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining  "
                  f"DX range=[{dxs.min():+.1f}, {dxs.max():+.1f}]", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Part C complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")
    return fine_profiles


# ── Part D: Multi-Scale Probing ──────────────────────────────────────────────

def part_d_multiscale(sorted_probes, n_top=20, n_probes_per_radius=6):
    """6 random probes at 3 radii for top 20 cliffiest points."""
    radii_scales = [0.01, 0.005, 0.001]
    total_sims = n_top * n_probes_per_radius * len(radii_scales)
    print(f"\n{'='*80}")
    print(f"PART D: Multi-Scale Probing ({n_top} pts x {n_probes_per_radius} probes x "
          f"{len(radii_scales)} radii = {total_sims} sims)")
    print(f"{'='*80}")

    multiscale = []
    t0 = time.perf_counter()
    sim_count = 0

    for rank, pt in enumerate(sorted_probes[:n_top]):
        base_w = pt["weights"]
        base_dx = pt["base_dx"]

        scale_results = {}
        for r_scale in radii_scales:
            delta_dxs = []
            for _ in range(n_probes_per_radius):
                d = random_direction_6d()
                pw = perturb_weights(base_w, d, r_scale)
                dx_probe = simulate_dx_only(pw)
                delta_dxs.append(dx_probe - base_dx)
                sim_count += 1

            delta_dxs = np.array(delta_dxs)
            scale_results[str(r_scale)] = {
                "max_abs_delta": float(np.max(np.abs(delta_dxs))),
                "mean_abs_delta": float(np.mean(np.abs(delta_dxs))),
                "delta_dxs": delta_dxs.tolist(),
            }

        multiscale.append({
            "rank": rank,
            "idx": pt["idx"],
            "base_dx": base_dx,
            "cliffiness": pt["cliffiness"],
            "scales": scale_results,
        })

        if (rank + 1) % 5 == 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / sim_count
            remaining = rate * (total_sims - sim_count)
            print(f"  [{rank+1:3d}/{n_top}] {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Part D complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")
    return multiscale


# ── Feature Extraction & Classification ──────────────────────────────────────

def extract_all_features(profiles, perp_profiles):
    """Compute cliff features for all profiled points and assign taxonomy types."""
    print(f"\n{'='*80}")
    print("FEATURE EXTRACTION & CLASSIFICATION")
    print(f"{'='*80}")

    # Build perpendicular lookup by idx
    perp_by_idx = {}
    for pp in perp_profiles:
        perp_by_idx[pp["idx"]] = pp

    for prof in profiles:
        radii = np.array(prof["radii"])
        dxs = np.array(prof["dxs"])
        base_dx = prof["base_dx"]

        features = compute_cliff_features(radii, dxs, base_dx)

        # Dominant weight: weight index with largest |gradient component|
        grad = np.array(prof["gradient_vector"])
        dom_idx = int(np.argmax(np.abs(grad)))
        features["dominant_weight"] = WEIGHT_NAMES[dom_idx]
        features["dominant_weight_idx"] = dom_idx

        # Perpendicular ratio
        if prof["idx"] in perp_by_idx:
            pp = perp_by_idx[prof["idx"]]
            perp_dxs = np.array(pp["dxs"])
            perp_range = float(np.max(perp_dxs) - np.min(perp_dxs))
            grad_range = features["dx_range"]
            features["perpendicular_ratio"] = round(
                perp_range / max(grad_range, 1e-12), 4)
        else:
            features["perpendicular_ratio"] = None

        # Classify
        cliff_type = classify_cliff(features)
        prof["features"] = features
        prof["type"] = cliff_type

    # Summary
    type_counts = {}
    for prof in profiles:
        t = prof["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Classification complete: {len(profiles)} profiles")
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:12s}: {cnt:3d} ({100*cnt/len(profiles):.1f}%)")

    return profiles


# ── Figures ──────────────────────────────────────────────────────────────────

def fig01_profile_gallery(profiles):
    """5x10 grid: all 50 cliff profiles, border colored by taxonomy type."""
    n_rows, n_cols = 5, 10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 9))

    for i, prof in enumerate(profiles):
        row, col = i // n_cols, i % n_cols
        ax = axes[row][col]

        radii = np.array(prof["radii"])
        dxs = np.array(prof["dxs"])
        cliff_type = prof["type"]
        color = TYPE_COLORS.get(cliff_type, "#333333")

        ax.plot(radii, dxs, "-", color=color, lw=1.5)
        ax.axhline(prof["base_dx"], color="gray", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.3, ls=":")

        ax.set_title(f"#{i+1} {cliff_type}\ncliff={prof['cliffiness']:.0f}",
                      fontsize=7, color=color, fontweight="bold")
        ax.tick_params(labelsize=5)

        # Colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        if row == n_rows - 1:
            ax.set_xlabel("r", fontsize=6)
        if col == 0:
            ax.set_ylabel("DX", fontsize=6)

    fig.suptitle("Cliff Profile Gallery — Top 50 Points (colored by taxonomy type)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "tax_fig01_profile_gallery.png")


def fig02_taxonomy_summary(profiles):
    """1x3: type distribution, prototypical profile per type, feature radar per type."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    # Count types
    type_counts = {}
    type_profiles = {}
    for prof in profiles:
        t = prof["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        if t not in type_profiles:
            type_profiles[t] = []
        type_profiles[t].append(prof)

    types_sorted = sorted(type_counts.keys(), key=lambda t: -type_counts[t])

    # Left: bar chart of type distribution
    ax = axes[0]
    x_pos = np.arange(len(types_sorted))
    counts = [type_counts[t] for t in types_sorted]
    colors = [TYPE_COLORS.get(t, "#333") for t in types_sorted]
    bars = ax.bar(x_pos, counts, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(types_sorted, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Taxonomy Distribution")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
    clean_ax(ax)

    # Center: prototypical profile per type (the one closest to mean features)
    ax = axes[1]
    for t in types_sorted:
        tps = type_profiles[t]
        # Pick the profile closest to median cliffiness within type
        cliff_vals = [tp["cliffiness"] for tp in tps]
        median_cliff = np.median(cliff_vals)
        proto = min(tps, key=lambda tp: abs(tp["cliffiness"] - median_cliff))
        radii = np.array(proto["radii"])
        dxs = np.array(proto["dxs"])
        # Normalize for comparison
        dxs_norm = (dxs - np.mean(dxs)) / max(np.std(dxs), 1e-12)
        ax.plot(radii, dxs_norm, "-", color=TYPE_COLORS.get(t, "#333"),
                lw=2, label=f"{t} (n={type_counts[t]})")
    ax.set_xlabel("Radius along gradient")
    ax.set_ylabel("Normalized DX")
    ax.set_title("Prototypical Profile per Type")
    ax.legend(fontsize=8)
    ax.axvline(0, color="gray", lw=0.3, ls=":")
    clean_ax(ax)

    # Right: feature radar (bar chart of mean features per type)
    ax = axes[2]
    feature_names = ["sharpness", "asymmetry_index", "recovery_ratio", "roughness"]
    feature_labels = ["Sharpness", "Asymmetry", "Recovery", "Roughness"]

    bar_width = 0.8 / max(len(types_sorted), 1)
    x_feat = np.arange(len(feature_names))

    for ti, t in enumerate(types_sorted):
        tps = type_profiles[t]
        means = []
        for fn in feature_names:
            vals = [tp["features"][fn] for tp in tps]
            # Cap asymmetry at 10 for display
            if fn == "asymmetry_index":
                vals = [min(v, 10) for v in vals]
            means.append(np.mean(vals))
        # Normalize each feature to [0,1] range across types for radar display
        ax.bar(x_feat + ti * bar_width, means, bar_width,
               color=TYPE_COLORS.get(t, "#333"), label=t, edgecolor="black", lw=0.3)

    ax.set_xticks(x_feat + bar_width * (len(types_sorted) - 1) / 2)
    ax.set_xticklabels(feature_labels, fontsize=9)
    ax.set_ylabel("Feature Value")
    ax.set_title("Mean Features by Type")
    ax.legend(fontsize=7, loc="upper right")
    clean_ax(ax)

    fig.suptitle("Cliff Taxonomy Summary", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "tax_fig02_taxonomy_summary.png")


def fig03_perpendicular(profiles, perp_profiles):
    """3x5: gradient vs perpendicular profiles for top 15 points."""
    n_show = min(15, len(perp_profiles))
    n_rows, n_cols = 3, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 9))

    # Build profile lookup by idx
    prof_by_idx = {p["idx"]: p for p in profiles}

    for i in range(n_show):
        row, col = i // n_cols, i % n_cols
        ax = axes[row][col]

        pp = perp_profiles[i]
        idx = pp["idx"]
        grad_prof = prof_by_idx.get(idx)

        if grad_prof is not None:
            grad_radii = np.array(grad_prof["radii"])
            grad_dxs = np.array(grad_prof["dxs"])
            ax.plot(grad_radii, grad_dxs, "o-", color="#E24A33", lw=1.5,
                    markersize=3, label="Gradient dir")

        perp_radii = np.array(pp["radii"])
        perp_dxs = np.array(pp["dxs"])
        ax.plot(perp_radii, perp_dxs, "s-", color="#348ABD", lw=1.5,
                markersize=3, label="Perpendicular")

        ax.axhline(pp["base_dx"], color="gray", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.3, ls=":")

        perp_range = float(np.max(perp_dxs) - np.min(perp_dxs))
        if grad_prof is not None:
            grad_range = float(np.max(grad_dxs) - np.min(grad_dxs))
            ratio = perp_range / max(grad_range, 1e-12)
            structure = "ridge" if ratio > 0.5 else "face"
        else:
            ratio = 0
            structure = "?"

        cliff_type = grad_prof["type"] if grad_prof else "?"
        ax.set_title(f"#{i+1} {cliff_type}\nperp_ratio={ratio:.2f} ({structure})",
                      fontsize=8)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=7)
        if row == n_rows - 1:
            ax.set_xlabel("Radius", fontsize=7)
        if col == 0:
            ax.set_ylabel("DX (m)", fontsize=7)
        clean_ax(ax)

    # Hide unused subplots
    for i in range(n_show, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle("Gradient vs Perpendicular Profiles — Ridge vs Face Structure",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "tax_fig03_perpendicular.png")


def fig04_cliff_edges(fine_profiles):
    """4x5: fine-grained cliff profiles for top 20."""
    n_show = min(20, len(fine_profiles))
    n_rows, n_cols = 4, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 13))

    for i in range(n_show):
        row, col = i // n_cols, i % n_cols
        ax = axes[row][col]

        fp = fine_profiles[i]
        radii = np.array(fp["radii"])
        dxs = np.array(fp["dxs"])

        ax.plot(radii, dxs, "o-", color="#4C72B0", lw=1.5, markersize=3)
        ax.axhline(fp["base_dx"], color="gray", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.3, ls=":")

        # Compute local transition width
        features = compute_cliff_features(radii, dxs, fp["base_dx"])
        ax.set_title(f"#{i+1} cliff={fp['cliffiness']:.0f}\n"
                      f"width={features['transition_width']:.4f} "
                      f"sharp={features['sharpness']:.2f}",
                      fontsize=8)
        ax.tick_params(labelsize=6)
        if row == n_rows - 1:
            ax.set_xlabel("Radius (zoomed)", fontsize=7)
        if col == 0:
            ax.set_ylabel("DX (m)", fontsize=7)
        clean_ax(ax)

    # Hide unused
    for i in range(n_show, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle("Fine-Grained Cliff Edges (r = +/-0.05, 40 points)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "tax_fig04_cliff_edges.png")


def fig05_feature_space(profiles):
    """2x2 scatter plots of shape features, colored by type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    pairs = [
        ("asymmetry_index", "sharpness", "Asymmetry Index", "Sharpness"),
        ("recovery_ratio", "roughness", "Recovery Ratio", "Roughness"),
        ("transition_width", "max_step", "Transition Width", "Max Step (m)"),
        ("sharpness", "recovery_ratio", "Sharpness", "Recovery Ratio"),
    ]

    for ax_idx, (fx, fy, xlabel, ylabel) in enumerate(pairs):
        ax = axes[ax_idx // 2][ax_idx % 2]

        for t in TYPE_COLORS:
            xs = [p["features"][fx] for p in profiles if p["type"] == t]
            ys = [p["features"][fy] for p in profiles if p["type"] == t]
            if xs:
                # Cap asymmetry at 10 for display
                if fx == "asymmetry_index":
                    xs = [min(x, 10) for x in xs]
                if fy == "asymmetry_index":
                    ys = [min(y, 10) for y in ys]
                ax.scatter(xs, ys, c=TYPE_COLORS[t], label=t, s=40, alpha=0.7,
                           edgecolors="black", lw=0.3)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        clean_ax(ax)

    fig.suptitle("Shape Feature Space (colored by taxonomy type)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "tax_fig05_feature_space.png")


def fig06_multiscale(multiscale, profiles):
    """1x2: multi-scale cliffiness (left); dominant weight by type (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: multi-scale cliffiness
    ax = axes[0]
    radii_scales = [0.01, 0.005, 0.001]
    for ms in multiscale:
        max_deltas = [ms["scales"][str(r)]["max_abs_delta"] for r in radii_scales]
        ax.plot(radii_scales, max_deltas, "o-", alpha=0.4, lw=1, markersize=4,
                color="#4C72B0")

    # Mean line
    mean_deltas = []
    for r in radii_scales:
        vals = [ms["scales"][str(r)]["max_abs_delta"] for ms in multiscale]
        mean_deltas.append(np.mean(vals))
    ax.plot(radii_scales, mean_deltas, "s-", color="#E24A33", lw=3, markersize=8,
            label="Mean", zorder=5)

    ax.set_xlabel("Probe Radius")
    ax.set_ylabel("Max |delta DX| (m)")
    ax.set_title("Multi-Scale Cliffiness (Top 20)")
    ax.set_xscale("log")
    ax.legend()
    clean_ax(ax)

    # Right: dominant weight by cliff type
    ax = axes[1]
    type_weight_counts = {}
    for prof in profiles:
        t = prof["type"]
        dw = prof["features"]["dominant_weight"]
        if t not in type_weight_counts:
            type_weight_counts[t] = {wn: 0 for wn in WEIGHT_NAMES}
        type_weight_counts[t][dw] = type_weight_counts[t].get(dw, 0) + 1

    types_with_data = sorted(type_weight_counts.keys())
    bar_width = 0.8 / max(len(WEIGHT_NAMES), 1)
    x_type = np.arange(len(types_with_data))

    for wi, wn in enumerate(WEIGHT_NAMES):
        counts = [type_weight_counts[t].get(wn, 0) for t in types_with_data]
        ax.bar(x_type + wi * bar_width, counts, bar_width, label=wn,
               edgecolor="black", lw=0.3)

    ax.set_xticks(x_type + bar_width * (len(WEIGHT_NAMES) - 1) / 2)
    ax.set_xticklabels(types_with_data, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Dominant Weight by Cliff Type")
    ax.legend(fontsize=7, ncol=2)
    clean_ax(ax)

    fig.suptitle("Multi-Scale Structure & Weight Attribution",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "tax_fig06_multiscale.png")


# ── Console Output ──────────────────────────────────────────────────────────

def print_analysis(profiles, perp_profiles, fine_profiles, multiscale):
    """Print taxonomy tables and statistics."""
    print(f"\n{'='*80}")
    print("CLIFF TAXONOMY — RESULTS")
    print(f"{'='*80}")

    # ── Taxonomy Distribution ──
    type_counts = {}
    for prof in profiles:
        t = prof["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    n_total = len(profiles)
    print(f"\n  TAXONOMY DISTRIBUTION ({n_total} profiles):")
    print(f"    {'Type':<12} {'Count':>6} {'Pct':>8}")
    print("    " + "-" * 28)
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<12} {cnt:6d} {100*cnt/n_total:7.1f}%")

    # ── Feature Statistics per Type ──
    feature_names = ["sharpness", "asymmetry_index", "recovery_ratio",
                     "roughness", "max_step", "transition_width", "dx_range"]
    print(f"\n  FEATURE STATISTICS BY TYPE:")
    print(f"    {'Type':<12}", end="")
    for fn in feature_names:
        print(f" {fn[:10]:>12}", end="")
    print()
    print("    " + "-" * (12 + 12 * len(feature_names)))

    type_profiles = {}
    for prof in profiles:
        t = prof["type"]
        if t not in type_profiles:
            type_profiles[t] = []
        type_profiles[t].append(prof)

    for t in sorted(type_profiles.keys()):
        tps = type_profiles[t]
        print(f"    {t:<12}", end="")
        for fn in feature_names:
            vals = [tp["features"][fn] for tp in tps]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f" {mean_val:5.2f}+{std_val:4.2f}", end="")
        print()

    # ── Multi-Scale Roughness ──
    print(f"\n  MULTI-SCALE ROUGHNESS (does cliffiness increase at smaller scales?):")
    radii_scales = [0.01, 0.005, 0.001]
    print(f"    {'Radius':>10} {'Mean |dDX|':>12} {'Max |dDX|':>12}")
    print("    " + "-" * 36)
    for r in radii_scales:
        vals = [ms["scales"][str(r)]["max_abs_delta"] for ms in multiscale]
        print(f"    {r:10.4f} {np.mean(vals):12.3f} {np.max(vals):12.3f}")

    # Fractal test: does gradient magnitude (|dDX|/r) increase at smaller scales?
    print(f"\n    Scale-normalized gradient (|dDX|/r):")
    for r in radii_scales:
        vals = [ms["scales"][str(r)]["max_abs_delta"] / r for ms in multiscale]
        print(f"      r={r:.4f}:  mean grad = {np.mean(vals):.1f}  "
              f"max grad = {np.max(vals):.1f}")

    increasing = True
    prev_mean = 0
    for r in radii_scales:
        vals = [ms["scales"][str(r)]["max_abs_delta"] / r for ms in multiscale]
        curr_mean = np.mean(vals)
        if curr_mean < prev_mean and prev_mean > 0:
            increasing = False
        prev_mean = curr_mean
    print(f"    Fractal structure: {'YES — gradient increases at smaller scales' if increasing else 'NO — gradient saturates or decreases'}")

    # ── Cliff Dimensionality ──
    print(f"\n  CLIFF DIMENSIONALITY (ridge vs face):")
    ridge_count = 0
    face_count = 0
    for prof in profiles:
        pr = prof["features"].get("perpendicular_ratio")
        if pr is not None:
            if pr > 0.5:
                ridge_count += 1
            else:
                face_count += 1
    total_dim = ridge_count + face_count
    if total_dim > 0:
        print(f"    Ridges (perp_ratio > 0.5): {ridge_count}/{total_dim} "
              f"({100*ridge_count/total_dim:.1f}%)")
        print(f"    Faces  (perp_ratio <= 0.5): {face_count}/{total_dim} "
              f"({100*face_count/total_dim:.1f}%)")
    else:
        print(f"    No perpendicular data available")

    # ── Per-Weight Cliff Frequency by Type ──
    print(f"\n  PER-WEIGHT CLIFF CREATION FREQUENCY BY TYPE:")
    print(f"    {'Type':<12}", end="")
    for wn in WEIGHT_NAMES:
        print(f" {wn:>6}", end="")
    print()
    print("    " + "-" * (12 + 7 * len(WEIGHT_NAMES)))
    for t in sorted(type_profiles.keys()):
        tps = type_profiles[t]
        print(f"    {t:<12}", end="")
        for wn in WEIGHT_NAMES:
            cnt = sum(1 for tp in tps if tp["features"]["dominant_weight"] == wn)
            print(f" {cnt:6d}", end="")
        print()

    # ── Top 5 Most Extreme Cliffs ──
    print(f"\n  TOP 5 MOST EXTREME CLIFFS:")
    print(f"    {'Rank':>5} {'Type':<12} {'Cliffiness':>11} {'Sharpness':>10} "
          f"{'Asymmetry':>10} {'Recovery':>9} {'DomWeight':<10} {'DX Range':>10}")
    print("    " + "-" * 79)
    for i, prof in enumerate(profiles[:5]):
        f = prof["features"]
        print(f"    {i+1:5d} {prof['type']:<12} {prof['cliffiness']:11.2f} "
              f"{f['sharpness']:10.3f} {f['asymmetry_index']:10.3f} "
              f"{f['recovery_ratio']:9.3f} {f['dominant_weight']:<10} "
              f"{f['dx_range']:10.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    np.random.seed(42)

    # Back up brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    # Load atlas data
    print(f"Loading {IN_JSON} ...")
    with open(IN_JSON) as f:
        atlas = json.load(f)
    probe_results = atlas["probe_results"]
    print(f"  Loaded {len(probe_results)} probe results")

    # Sort by cliffiness (descending)
    sorted_probes = sorted(probe_results, key=lambda x: x["cliffiness"],
                           reverse=True)

    # Sim budget
    budget = 50 * 30 + 30 * 20 + 20 * 40 + 20 * 6 * 3
    print(f"  Total simulation budget: ~{budget} sims")

    # ── Determinism check ────────────────────────────────────────────────────
    print("\nDeterminism check (3 atlas anatomy profiles)...")
    anatomy = atlas.get("anatomy", [])
    for i, anat in enumerate(anatomy[:3]):
        base_w = sorted_probes[i]["weights"]
        dx_sim = simulate_dx_only(base_w)
        stored_dx = sorted_probes[i]["base_dx"]
        err = abs(dx_sim - stored_dx)
        status = "OK" if err < 0.01 else "MISMATCH"
        print(f"  Rank {i}: stored={stored_dx:+.4f}  sim={dx_sim:+.4f}  "
              f"err={err:.6f}  [{status}]")

    # ── Part A: Extended Cliff Profiles ──────────────────────────────────────
    profiles = part_a_extended_profiles(sorted_probes)

    # ── Part B: Perpendicular Profiles ──────────────────────────────────────
    perp_profiles = part_b_perpendicular(sorted_probes)

    # ── Part C: Fine-Grained Cliff Edges ────────────────────────────────────
    fine_profiles = part_c_fine_grained(sorted_probes)

    # ── Part D: Multi-Scale Probing ─────────────────────────────────────────
    multiscale = part_d_multiscale(sorted_probes)

    # ── Restore brain.nndf ──────────────────────────────────────────────────
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # ── Feature Extraction & Classification ─────────────────────────────────
    profiles = extract_all_features(profiles, perp_profiles)

    # ── Console Output ──────────────────────────────────────────────────────
    print_analysis(profiles, perp_profiles, fine_profiles, multiscale)

    # ── Figures ─────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")
    fig01_profile_gallery(profiles)
    fig02_taxonomy_summary(profiles)
    fig03_perpendicular(profiles, perp_profiles)
    fig04_cliff_edges(fine_profiles)
    fig05_feature_space(profiles)
    fig06_multiscale(multiscale, profiles)

    # ── Save JSON ───────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Prepare output with all data
    output = {
        "meta": {
            "n_profiles": len(profiles),
            "n_perp_profiles": len(perp_profiles),
            "n_fine_profiles": len(fine_profiles),
            "n_multiscale": len(multiscale),
            "weight_names": WEIGHT_NAMES,
            "type_colors": TYPE_COLORS,
        },
        "profiles": profiles,
        "perp_profiles": perp_profiles,
        "fine_profiles": fine_profiles,
        "multiscale": multiscale,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
