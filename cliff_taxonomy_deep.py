#!/usr/bin/env python3
"""
cliff_taxonomy_deep.py

Step Zone Deep Resolution — probing the richest Type 3 (Wolfram) zones
of the fitness landscape at ultra-fine scales.

Builds on cliff_taxonomy.py results. Targets the 10 most chaotic Step-type
cliff profiles with three complementary probing strategies.

Simulation budget: ~2,420 sims (~3 min)

Phase 1: Logarithmic Zoom Cascade (1,200 sims)
    10 Steps x 6 zoom levels x 20 points.
    Scales: r = ±{0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003}
    Centered on each profile's primary step location.
    Measures fractal dimension across 2.5 extra decades.

Phase 2: Directional Fan (720 sims)
    10 Steps x 8 directions x 9 points at r = ±0.005.
    Directions evenly spaced in the plane perpendicular to gradient.
    Tests isotropy of the chaos.

Phase 3: 2D Micro-Grid (500 sims)
    5 Steps x 10x10 grid in gradient + perpendicular plane.
    Extent r = ±0.005. Direct visualization of local cliff texture.

Outputs:
    artifacts/cliff_taxonomy_deep.json
    artifacts/plots/deep_fig01_zoom_cascade.png
    artifacts/plots/deep_fig02_fractal_dimension.png
    artifacts/plots/deep_fig03_directional_fan.png
    artifacts/plots/deep_fig04_isotropy.png
    artifacts/plots/deep_fig05_micro_grids.png
    artifacts/plots/deep_fig06_smoothness_verdict.png

Usage:
    python3 cliff_taxonomy_deep.py
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
from matplotlib.colors import Normalize

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from compute_beer_analytics import NumpyEncoder

# ── Constants ────────────────────────────────────────────────────────────────

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]
WEIGHT_LABELS = {
    "w03": "Torso->Back",  "w04": "Torso->Front",
    "w13": "BackLeg->Back", "w14": "BackLeg->Front",
    "w23": "FrontLeg->Back", "w24": "FrontLeg->Front",
}

TAX_JSON = PROJECT / "artifacts" / "cliff_taxonomy.json"
OUT_JSON = PROJECT / "artifacts" / "cliff_taxonomy_deep.json"
PLOT_DIR = PROJECT / "artifacts" / "plots"

ZOOM_SCALES = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
N_ZOOM_PTS = 20
N_FAN_DIRS = 8
N_FAN_PTS = 9
FAN_RADIUS = 0.005
GRID_N = 10
GRID_RADIUS = 0.005

# ── Simulation (reused from atlas/taxonomy) ──────────────────────────────────

def write_brain_standard(weights):
    """Write a 3-sensor, 2-motor brain.nndf file from a weight dict."""
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
    """Run a headless PyBullet sim and return the robot's x-displacement."""
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

def perturb_weights(base_weights, direction, radius):
    """Offset base_weights along a 6D direction vector scaled by radius. Returns new weight dict."""
    w = {}
    for i, wn in enumerate(WEIGHT_NAMES):
        w[wn] = base_weights[wn] + radius * direction[i]
    return w


def clean_ax(ax):
    """Remove top and right spines from a matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    """Save a matplotlib figure to PLOT_DIR and close it."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


def perpendicular_basis(grad):
    """Build an orthonormal 2D basis for the plane perpendicular to grad.

    Returns (e1, e2) — two 6D unit vectors orthogonal to grad and each other.
    """
    g_hat = grad / max(np.linalg.norm(grad), 1e-12)

    # First perpendicular via Gram-Schmidt on a random vector
    v1 = np.random.randn(6)
    v1 = v1 - np.dot(v1, g_hat) * g_hat
    if np.linalg.norm(v1) < 1e-12:
        v1 = np.random.randn(6)
        v1 = v1 - np.dot(v1, g_hat) * g_hat
    e1 = v1 / np.linalg.norm(v1)

    # Second perpendicular via Gram-Schmidt on another random vector
    v2 = np.random.randn(6)
    v2 = v2 - np.dot(v2, g_hat) * g_hat
    v2 = v2 - np.dot(v2, e1) * e1
    if np.linalg.norm(v2) < 1e-12:
        v2 = np.random.randn(6)
        v2 = v2 - np.dot(v2, g_hat) * g_hat
        v2 = v2 - np.dot(v2, e1) * e1
    e2 = v2 / np.linalg.norm(v2)

    return e1, e2


def compute_chaos_score(dxs):
    """Chaos score: sign_change_rate * spectral_ratio / (1 + autocorr)."""
    dxs = np.asarray(dxs)
    dxs_c = dxs - np.mean(dxs)
    var = np.var(dxs_c)

    # Sign-change rate in the first derivative: high = oscillatory/chaotic
    d1 = np.diff(dxs)
    sign_changes = np.sum(np.diff(np.sign(d1)) != 0)
    scr = sign_changes / max(len(d1), 1)

    # Lag-1 autocorrelation: low/negative = uncorrelated = more chaotic
    if var > 1e-12:
        autocorr = np.mean(dxs_c[:-1] * dxs_c[1:]) / var
    else:
        autocorr = 0

    # Spectral ratio: high-frequency power vs low-frequency power
    # High ratio means energy is spread to small scales (chaotic signature)
    fft = np.fft.rfft(dxs_c)
    power = np.abs(fft) ** 2
    nf = len(power)
    low = np.sum(power[:max(nf // 3, 1)])
    high = np.sum(power[max(nf // 3, 1):])
    sr = high / max(low, 1e-12)

    # Combine: high sign-change rate * high spectral ratio, penalized by autocorrelation
    return scr * sr / (1 + max(autocorr, 0))


# ── Select Target Steps ─────────────────────────────────────────────────────

def select_targets(tax_data):
    """Select the 10 most chaotic Step profiles, return sorted list."""
    profiles = tax_data["profiles"]
    steps = [p for p in profiles if p["type"] == "Step"]

    # Rank by chaos score
    scored = []
    for p in steps:
        cs = compute_chaos_score(p["dxs"])
        scored.append((cs, p))
    scored.sort(reverse=True, key=lambda x: x[0])

    targets = []
    for i, (cs, p) in enumerate(scored[:10]):
        targets.append({
            "select_rank": i,
            "orig_rank": p["rank"],
            "idx": p["idx"],
            "chaos_score": float(cs),
            "base_dx": p["base_dx"],
            "cliffiness": p["cliffiness"],
            "gradient_vector": p["gradient_vector"],
            "cliff_direction": p["cliff_direction"],
            "weights": p["weights"],
            "features": p["features"],
            "step_location": p["features"]["step_location"],
        })

    print(f"  Selected {len(targets)} Step targets by chaos score:")
    for t in targets:
        print(f"    #{t['select_rank']+1}: orig_rank={t['orig_rank']+1}, "
              f"idx={t['idx']}, chaos={t['chaos_score']:.3f}, "
              f"cliff={t['cliffiness']:.1f}, step_loc={t['step_location']:+.4f}")

    return targets


# ── Phase 1: Logarithmic Zoom Cascade ────────────────────────────────────────

def phase1_zoom_cascade(targets):
    """20-point profiles at 6 geometrically spaced scales, centered on step location."""
    n_targets = len(targets)
    n_scales = len(ZOOM_SCALES)
    total_sims = n_targets * n_scales * N_ZOOM_PTS
    print(f"\n{'='*80}")
    print(f"PHASE 1: Logarithmic Zoom Cascade "
          f"({n_targets} x {n_scales} scales x {N_ZOOM_PTS} pts = {total_sims} sims)")
    print(f"  Scales: {ZOOM_SCALES}")
    print(f"{'='*80}")

    results = []
    t0 = time.perf_counter()
    sim_count = 0

    for ti, tgt in enumerate(targets):
        cliff_dir = np.array(tgt["cliff_direction"])
        base_w = tgt["weights"]
        center_r = tgt["step_location"]

        # Multi-scale probing: sample the same neighborhood at 6 geometrically
        # decreasing radii. If DX range stays constant across scales, the
        # landscape is fractal (no smoothness floor); if it shrinks, it's smooth.
        scale_profiles = []
        for scale in ZOOM_SCALES:
            radii = np.linspace(center_r - scale, center_r + scale, N_ZOOM_PTS)
            dxs = np.empty(N_ZOOM_PTS)

            for k, r in enumerate(radii):
                pw = perturb_weights(base_w, cliff_dir, r)
                dxs[k] = simulate_dx_only(pw)
                sim_count += 1

            dx_range = float(np.max(dxs) - np.min(dxs))
            steps_arr = np.abs(np.diff(dxs))
            max_step = float(np.max(steps_arr)) if len(steps_arr) > 0 else 0
            chaos = compute_chaos_score(dxs)

            # Local derivative magnitude
            dr = radii[1] - radii[0]
            derivs = np.diff(dxs) / dr
            mean_abs_deriv = float(np.mean(np.abs(derivs)))

            scale_profiles.append({
                "scale": scale,
                "radii": radii.tolist(),
                "dxs": dxs.tolist(),
                "dx_range": round(dx_range, 6),
                "max_step": round(max_step, 6),
                "chaos_score": round(chaos, 6),
                "mean_abs_deriv": round(mean_abs_deriv, 2),
            })

        results.append({
            "select_rank": tgt["select_rank"],
            "idx": tgt["idx"],
            "center_r": center_r,
            "scale_profiles": scale_profiles,
        })

        elapsed = time.perf_counter() - t0
        rate = elapsed / sim_count
        remaining = rate * (total_sims - sim_count)
        print(f"  [{ti+1:2d}/{n_targets}] {elapsed:.1f}s elapsed, "
              f"~{remaining:.0f}s remaining", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Phase 1 complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")
    return results


# ── Phase 2: Directional Fan ─────────────────────────────────────────────────

def phase2_directional_fan(targets):
    """8 evenly-spaced directions in the perp plane, 9-point profile at r=±0.005."""
    n_targets = len(targets)
    total_sims = n_targets * N_FAN_DIRS * N_FAN_PTS
    print(f"\n{'='*80}")
    print(f"PHASE 2: Directional Fan "
          f"({n_targets} x {N_FAN_DIRS} dirs x {N_FAN_PTS} pts = {total_sims} sims)")
    print(f"{'='*80}")

    radii = np.linspace(-FAN_RADIUS, FAN_RADIUS, N_FAN_PTS)
    results = []
    t0 = time.perf_counter()
    sim_count = 0

    for ti, tgt in enumerate(targets):
        grad = np.array(tgt["gradient_vector"])
        e1, e2 = perpendicular_basis(grad)
        base_w = tgt["weights"]

        fan_profiles = []
        # Sweep 8 directions in the plane perpendicular to the gradient.
        # Tests whether chaos is isotropic or concentrated along specific axes.
        angles = np.linspace(0, np.pi, N_FAN_DIRS, endpoint=False)

        for angle in angles:
            # Linear combination of the two orthonormal basis vectors (e1, e2)
            # traces out evenly-spaced directions in the perpendicular plane.
            direction = np.cos(angle) * e1 + np.sin(angle) * e2
            dxs = np.empty(N_FAN_PTS)

            for k, r in enumerate(radii):
                pw = perturb_weights(base_w, direction, r)
                dxs[k] = simulate_dx_only(pw)
                sim_count += 1

            dx_range = float(np.max(dxs) - np.min(dxs))
            chaos = compute_chaos_score(dxs)

            fan_profiles.append({
                "angle_rad": round(float(angle), 6),
                "direction": direction.tolist(),
                "radii": radii.tolist(),
                "dxs": dxs.tolist(),
                "dx_range": round(dx_range, 6),
                "chaos_score": round(chaos, 6),
            })

        # Also probe gradient direction for comparison
        cliff_dir = np.array(tgt["cliff_direction"])
        grad_dxs = np.empty(N_FAN_PTS)
        for k, r in enumerate(radii):
            pw = perturb_weights(base_w, cliff_dir, r)
            grad_dxs[k] = simulate_dx_only(pw)
            sim_count += 1

        grad_range = float(np.max(grad_dxs) - np.min(grad_dxs))
        grad_chaos = compute_chaos_score(grad_dxs)

        results.append({
            "select_rank": tgt["select_rank"],
            "idx": tgt["idx"],
            "e1": e1.tolist(),
            "e2": e2.tolist(),
            "fan_profiles": fan_profiles,
            "gradient_profile": {
                "radii": radii.tolist(),
                "dxs": grad_dxs.tolist(),
                "dx_range": round(grad_range, 6),
                "chaos_score": round(grad_chaos, 6),
            },
        })

        elapsed = time.perf_counter() - t0
        rate = elapsed / sim_count
        remaining = rate * (total_sims - sim_count)
        print(f"  [{ti+1:2d}/{n_targets}] {elapsed:.1f}s elapsed, "
              f"~{remaining:.0f}s remaining", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Phase 2 complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")
    return results


# ── Phase 3: 2D Micro-Grid ──────────────────────────────────────────────────

def phase3_micro_grid(targets):
    """10x10 grid in gradient + perpendicular plane for top 5 targets."""
    n_targets = min(5, len(targets))
    total_sims = n_targets * GRID_N * GRID_N
    print(f"\n{'='*80}")
    print(f"PHASE 3: 2D Micro-Grid "
          f"({n_targets} x {GRID_N}x{GRID_N} = {total_sims} sims)")
    print(f"{'='*80}")

    grid_coords = np.linspace(-GRID_RADIUS, GRID_RADIUS, GRID_N)
    results = []
    t0 = time.perf_counter()
    sim_count = 0

    for ti, tgt in enumerate(targets[:n_targets]):
        grad = np.array(tgt["gradient_vector"])
        grad_norm = np.linalg.norm(grad)
        cliff_dir = np.array(tgt["cliff_direction"])

        # Perpendicular direction via Gram-Schmidt
        e1, _ = perpendicular_basis(grad)
        base_w = tgt["weights"]

        # Build a 2D grid in the plane spanned by (cliff_dir, e1).
        # rg moves along the gradient/cliff axis, rp moves perpendicular.
        dx_grid = np.empty((GRID_N, GRID_N))
        for i, rg in enumerate(grid_coords):
            for j, rp in enumerate(grid_coords):
                direction = rg * cliff_dir + rp * e1
                pw = {}
                for wi, wn in enumerate(WEIGHT_NAMES):
                    pw[wn] = base_w[wn] + direction[wi]
                dx_grid[i, j] = simulate_dx_only(pw)
                sim_count += 1

            if (i + 1) % 5 == 0:
                elapsed = time.perf_counter() - t0
                rate = elapsed / sim_count
                remaining = rate * (total_sims - sim_count)
                print(f"    Target {ti+1}, row {i+1}/{GRID_N}, "
                      f"{elapsed:.1f}s, ~{remaining:.0f}s rem", flush=True)

        # Compute local gradient magnitude (cliffiness) via finite differences.
        # Uses forward/backward differences at boundaries, central differences
        # in the interior. The magnitude |grad DX| reveals cliff edges.
        step = grid_coords[1] - grid_coords[0]
        cliff_grid = np.zeros((GRID_N, GRID_N))
        for i in range(GRID_N):
            for j in range(GRID_N):
                if i == 0:
                    gx = (dx_grid[1, j] - dx_grid[0, j]) / step
                elif i == GRID_N - 1:
                    gx = (dx_grid[-1, j] - dx_grid[-2, j]) / step
                else:
                    gx = (dx_grid[i+1, j] - dx_grid[i-1, j]) / (2 * step)
                if j == 0:
                    gy = (dx_grid[i, 1] - dx_grid[i, 0]) / step
                elif j == GRID_N - 1:
                    gy = (dx_grid[i, -1] - dx_grid[i, -2]) / step
                else:
                    gy = (dx_grid[i, j+1] - dx_grid[i, j-1]) / (2 * step)
                cliff_grid[i, j] = np.sqrt(gx**2 + gy**2)

        results.append({
            "select_rank": tgt["select_rank"],
            "idx": tgt["idx"],
            "grid_coords": grid_coords.tolist(),
            "dx_grid": dx_grid.tolist(),
            "cliff_grid": cliff_grid.tolist(),
            "cliff_direction": cliff_dir.tolist(),
            "perp_direction": e1.tolist(),
        })

        elapsed = time.perf_counter() - t0
        print(f"  [{ti+1}/{n_targets}] done, {elapsed:.1f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Phase 3 complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")
    return results


# ── Figures ──────────────────────────────────────────────────────────────────

def fig01_zoom_cascade(zoom_results, targets):
    """2x5 grid: each target gets one panel showing all 6 zoom levels overlaid."""
    n = len(zoom_results)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    cmap = plt.cm.viridis
    scale_colors = [cmap(i / (len(ZOOM_SCALES) - 1)) for i in range(len(ZOOM_SCALES))]

    for ti, zr in enumerate(zoom_results):
        row, col = ti // n_cols, ti % n_cols
        ax = axes[row][col]
        tgt = targets[ti]

        for si, sp in enumerate(zr["scale_profiles"]):
            radii = np.array(sp["radii"])
            dxs = np.array(sp["dxs"])
            # Normalize DX to [0, 1] and radii to [-1, 1] so all 6 zoom
            # levels overlay on the same axes for visual self-similarity comparison.
            dmin, dmax = np.min(dxs), np.max(dxs)
            if dmax - dmin > 1e-12:
                dxs_norm = (dxs - dmin) / (dmax - dmin)
            else:
                dxs_norm = np.zeros_like(dxs)
            center = zr["center_r"]
            scale = sp["scale"]
            r_norm = (radii - center) / scale
            ax.plot(r_norm, dxs_norm, "-", color=scale_colors[si], lw=1.2,
                    alpha=0.85, label=f"r=±{scale}")

        ax.set_title(f"#{tgt['select_rank']+1} (idx={tgt['idx']})\n"
                      f"chaos={tgt['chaos_score']:.2f}", fontsize=9)
        ax.set_xlabel("Normalized radius", fontsize=7)
        ax.set_ylabel("Normalized DX", fontsize=7)
        ax.tick_params(labelsize=6)
        if ti == 0:
            ax.legend(fontsize=6, loc="upper left")
        clean_ax(ax)

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    fig.suptitle("Logarithmic Zoom Cascade — Self-Similarity Across Scales",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "deep_fig01_zoom_cascade.png")


def fig02_fractal_dimension(zoom_results, targets):
    """1x2: log-log dx_range vs scale (left), fractal dimension summary (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: log-log for each target
    ax = axes[0]
    slopes = []
    for ti, zr in enumerate(zoom_results):
        scales = [sp["scale"] for sp in zr["scale_profiles"]]
        ranges = [sp["dx_range"] for sp in zr["scale_profiles"]]
        # Clip for log safety
        ranges = [max(r, 1e-12) for r in ranges]
        ax.plot(scales, ranges, "o-", lw=1.5, markersize=5, alpha=0.7,
                label=f"#{targets[ti]['select_rank']+1}")

        # Fit log-log slope: slope~0 means DX range is scale-invariant (fractal),
        # slope~1 means DX range shrinks proportionally with scale (smooth/differentiable).
        log_s = np.log10(scales)
        log_r = np.log10(ranges)
        slope, intercept = np.polyfit(log_s, log_r, 1)
        slopes.append(slope)

    # Reference lines
    s_arr = np.array(ZOOM_SCALES)
    ax.plot(s_arr, s_arr * 1000, "k--", lw=1, alpha=0.4, label="slope=1 (smooth)")
    ax.plot(s_arr, np.full_like(s_arr, np.median(
        [sp["dx_range"] for zr in zoom_results for sp in zr["scale_profiles"]])),
        "k:", lw=1, alpha=0.4, label="slope=0 (fractal)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Scale (radius extent)", fontsize=11)
    ax.set_ylabel("DX Range (m)", fontsize=11)
    ax.set_title("DX Range vs Probe Scale")
    ax.legend(fontsize=7, ncol=2)
    clean_ax(ax)

    # Right: fractal dimension bar chart
    ax = axes[1]
    x_pos = np.arange(len(slopes))
    colors = ["#E24A33" if s < 0.3 else "#348ABD" if s < 0.7 else "#55A868"
              for s in slopes]
    bars = ax.bar(x_pos, slopes, color=colors, edgecolor="black", lw=0.5)
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.axhline(1, color="gray", lw=1, ls=":")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"#{targets[i]['select_rank']+1}" for i in range(len(slopes))],
                       fontsize=9)
    ax.set_ylabel("Log-log slope (0=fractal, 1=smooth)", fontsize=10)
    ax.set_title("Fractal Dimension Proxy per Target")
    ax.text(0.02, 0.95, f"Mean slope: {np.mean(slopes):.3f} ± {np.std(slopes):.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    clean_ax(ax)

    fig.suptitle("Fractal Scaling Analysis — Is There a Smoothness Floor?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "deep_fig02_fractal_dimension.png")

    return slopes


def fig03_directional_fan(fan_results, targets):
    """2x5 grid: polar-style visualization of DX range and chaos by angle."""
    n = len(fan_results)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ti, fr in enumerate(fan_results):
        row, col = ti // n_cols, ti % n_cols
        ax = axes[row][col]
        tgt = targets[ti]

        # Plot each direction's profile
        cmap = plt.cm.hsv
        for fi, fp in enumerate(fr["fan_profiles"]):
            color = cmap(fi / N_FAN_DIRS)
            radii = np.array(fp["radii"])
            dxs = np.array(fp["dxs"])
            angle_deg = float(fp["angle_rad"]) * 180 / np.pi
            ax.plot(radii, dxs, "-", color=color, lw=1, alpha=0.7,
                    label=f"{angle_deg:.0f}°")

        # Gradient direction in black
        gp = fr["gradient_profile"]
        ax.plot(gp["radii"], gp["dxs"], "k-", lw=2, alpha=0.9, label="grad")

        ax.set_title(f"#{tgt['select_rank']+1} (idx={tgt['idx']})", fontsize=9)
        ax.tick_params(labelsize=6)
        if ti == 0:
            ax.legend(fontsize=5, ncol=2, loc="upper left")
        if col == 0:
            ax.set_ylabel("DX (m)", fontsize=7)
        ax.set_xlabel("r", fontsize=7)
        clean_ax(ax)

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    fig.suptitle("Directional Fan — 8 Perpendicular Directions + Gradient (black)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "deep_fig03_directional_fan.png")


def fig04_isotropy(fan_results, targets):
    """1x2: isotropy analysis — range by angle (left), chaos by angle (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: DX range by angle for each target
    ax = axes[0]
    for ti, fr in enumerate(fan_results):
        angles = [fp["angle_rad"] for fp in fr["fan_profiles"]]
        ranges = [fp["dx_range"] for fp in fr["fan_profiles"]]
        grad_range = fr["gradient_profile"]["dx_range"]
        ax.plot(np.degrees(angles), ranges, "o-", lw=1.2, markersize=4, alpha=0.7,
                label=f"#{targets[ti]['select_rank']+1}")
        ax.axhline(grad_range, color=ax.get_lines()[-1].get_color(),
                    lw=0.5, ls=":", alpha=0.5)

    ax.set_xlabel("Angle in perpendicular plane (°)", fontsize=11)
    ax.set_ylabel("DX Range (m)", fontsize=11)
    ax.set_title("DX Range by Direction\n(dashed = gradient direction)")
    ax.legend(fontsize=7, ncol=2)
    clean_ax(ax)

    # Right: isotropy ratio (std/mean of ranges across angles).
    # Low ratio = chaos is uniform in all perpendicular directions (isotropic).
    # grad_vs_perp compares gradient-direction roughness to mean perpendicular roughness;
    # ratio >> 1 means the cliff is concentrated along the gradient axis.
    ax = axes[1]
    isotropy_ratios = []
    grad_vs_perp = []
    for ti, fr in enumerate(fan_results):
        ranges = np.array([fp["dx_range"] for fp in fr["fan_profiles"]])
        mean_r = np.mean(ranges)
        std_r = np.std(ranges)
        iso = std_r / max(mean_r, 1e-12)
        isotropy_ratios.append(iso)

        grad_range = fr["gradient_profile"]["dx_range"]
        gvp = grad_range / max(mean_r, 1e-12)
        grad_vs_perp.append(gvp)

    x_pos = np.arange(len(isotropy_ratios))
    width = 0.35
    ax.bar(x_pos - width / 2, isotropy_ratios, width, color="#348ABD",
           label="Isotropy (std/mean of perp ranges)", edgecolor="black", lw=0.5)
    ax.bar(x_pos + width / 2, grad_vs_perp, width, color="#E24A33",
           label="Grad range / mean perp range", edgecolor="black", lw=0.5)
    ax.axhline(1, color="gray", lw=1, ls=":")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"#{targets[i]['select_rank']+1}" for i in range(len(targets))],
                       fontsize=9)
    ax.set_ylabel("Ratio", fontsize=11)
    ax.set_title("Isotropy & Gradient Dominance")
    ax.legend(fontsize=9)
    clean_ax(ax)

    mean_iso = np.mean(isotropy_ratios)
    mean_gvp = np.mean(grad_vs_perp)
    ax.text(0.02, 0.95,
            f"Mean isotropy: {mean_iso:.3f}\nMean grad/perp: {mean_gvp:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Angular Structure of Type 3 Chaos",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "deep_fig04_isotropy.png")

    return isotropy_ratios, grad_vs_perp


def fig05_micro_grids(grid_results, targets):
    """2x5: DX (top) and cliffiness (bottom) micro-grids for top 5 targets."""
    n = len(grid_results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for gi, gr in enumerate(grid_results):
        coords = np.array(gr["grid_coords"])
        dx_g = np.array(gr["dx_grid"])
        cliff_g = np.array(gr["cliff_grid"])
        tgt_rank = gr["select_rank"]

        # Top: DX
        ax = axes[0][gi]
        vmax = max(np.percentile(np.abs(dx_g), 95), 1)
        im = ax.imshow(dx_g.T, origin="lower", aspect="equal",
                       extent=[coords[0], coords[-1], coords[0], coords[-1]],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label="DX (m)")
        ax.set_xlabel("Gradient dir (r)", fontsize=8)
        ax.set_ylabel("Perpendicular (r)", fontsize=8)
        ax.set_title(f"#{tgt_rank+1} DX", fontsize=10)
        ax.tick_params(labelsize=7)

        # Bottom: Cliffiness
        ax = axes[1][gi]
        im = ax.imshow(cliff_g.T, origin="lower", aspect="equal",
                       extent=[coords[0], coords[-1], coords[0], coords[-1]],
                       cmap="inferno")
        plt.colorbar(im, ax=ax, shrink=0.8, label="|grad DX|")
        ax.set_xlabel("Gradient dir (r)", fontsize=8)
        ax.set_ylabel("Perpendicular (r)", fontsize=8)
        ax.set_title(f"#{tgt_rank+1} Local Cliffiness", fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle("2D Micro-Grids (r = ±0.005) — Fitness Landscape Texture",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "deep_fig05_micro_grids.png")


def fig06_smoothness_verdict(zoom_results, fan_results, grid_results,
                              targets, slopes, isotropy_ratios, grad_vs_perp):
    """Summary figure: the verdict on smoothness, isotropy, and Wolfram class."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))

    # (0,0): Mean derivative magnitude vs scale
    ax = axes[0][0]
    for ti, zr in enumerate(zoom_results):
        scales = [sp["scale"] for sp in zr["scale_profiles"]]
        derivs = [sp["mean_abs_deriv"] for sp in zr["scale_profiles"]]
        ax.plot(scales, derivs, "o-", lw=1.2, markersize=4, alpha=0.7,
                label=f"#{targets[ti]['select_rank']+1}")

    # Reference: constant deriv = smooth, increasing = fractal
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Mean |dDX/dr|")
    ax.set_title("Gradient Magnitude vs Scale\n(flat = smooth, rising = fractal)")
    ax.legend(fontsize=6, ncol=2)
    clean_ax(ax)

    # (0,1): Chaos score vs scale
    ax = axes[0][1]
    for ti, zr in enumerate(zoom_results):
        scales = [sp["scale"] for sp in zr["scale_profiles"]]
        chaos_scores = [sp["chaos_score"] for sp in zr["scale_profiles"]]
        ax.plot(scales, chaos_scores, "o-", lw=1.2, markersize=4, alpha=0.7,
                label=f"#{targets[ti]['select_rank']+1}")
    ax.set_xscale("log")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Chaos Score")
    ax.set_title("Type 3 Chaos Score vs Scale\n(persistence = true Type 3)")
    ax.legend(fontsize=6, ncol=2)
    clean_ax(ax)

    # (0,2): DX range vs scale — the key fractal plot
    ax = axes[0][2]
    all_scales = []
    all_ranges = []
    for zr in zoom_results:
        for sp in zr["scale_profiles"]:
            all_scales.append(sp["scale"])
            all_ranges.append(sp["dx_range"])
    ax.scatter(all_scales, all_ranges, s=20, alpha=0.5, c="#4C72B0")
    # Bin means
    for scale in ZOOM_SCALES:
        vals = [r for s, r in zip(all_scales, all_ranges) if s == scale]
        ax.scatter([scale], [np.mean(vals)], s=100, c="#E24A33", zorder=5,
                   edgecolors="black", marker="D")
    # Fit
    log_s = np.log10(ZOOM_SCALES)
    mean_per_scale = [np.mean([r for s, r in zip(all_scales, all_ranges) if s == sc])
                      for sc in ZOOM_SCALES]
    log_r = np.log10(np.clip(mean_per_scale, 1e-12, None))
    slope_all, intercept = np.polyfit(log_s, log_r, 1)
    fit_x = np.logspace(np.log10(ZOOM_SCALES[-1]), np.log10(ZOOM_SCALES[0]), 50)
    fit_y = 10 ** (slope_all * np.log10(fit_x) + intercept)
    ax.plot(fit_x, fit_y, "k--", lw=2, label=f"fit slope={slope_all:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Scale")
    ax.set_ylabel("DX Range (m)")
    ax.set_title("Population Fractal Scaling")
    ax.legend(fontsize=9)
    clean_ax(ax)

    # (1,0): Histogram of fractal slopes
    ax = axes[1][0]
    ax.hist(slopes, bins=8, color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="#E24A33", lw=2, ls="--", label="Pure fractal (slope=0)")
    ax.axvline(1, color="#55A868", lw=2, ls="--", label="Smooth (slope=1)")
    ax.axvline(np.mean(slopes), color="black", lw=2, label=f"Mean={np.mean(slopes):.3f}")
    ax.set_xlabel("Log-log slope")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Fractal Slopes")
    ax.legend(fontsize=8)
    clean_ax(ax)

    # (1,1): Isotropy summary
    ax = axes[1][1]
    ax.scatter(isotropy_ratios, grad_vs_perp, s=80, c="#E24A33",
               edgecolors="black", zorder=5)
    for i, (iso, gvp) in enumerate(zip(isotropy_ratios, grad_vs_perp)):
        ax.annotate(f"#{targets[i]['select_rank']+1}", (iso, gvp),
                    fontsize=8, ha="left", va="bottom")
    ax.axhline(1, color="gray", lw=1, ls=":")
    ax.axvline(0.3, color="gray", lw=1, ls=":", alpha=0.5)
    ax.set_xlabel("Isotropy (std/mean of perp ranges)")
    ax.set_ylabel("Grad range / mean perp range")
    ax.set_title("Isotropy vs Gradient Dominance")
    ax.text(0.02, 0.95, "Low isotropy + grad~1 = isotropic chaos\n"
            "High isotropy + grad>>1 = directional cliff",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
    clean_ax(ax)

    # (1,2): Verdict text
    ax = axes[1][2]
    ax.axis("off")

    mean_slope = np.mean(slopes)
    mean_iso = np.mean(isotropy_ratios)
    mean_gvp = np.mean(grad_vs_perp)

    # Smoothness verdict: classify based on fractal slope.
    # slope < 0.2 = fractal (no smooth floor), 0.2-0.5 = partial, > 0.5 = smooth.
    if mean_slope < 0.2:
        smooth_verdict = "NO smoothness floor detected"
        smooth_detail = (f"Mean slope = {mean_slope:.3f} (near 0)\n"
                         f"DX range barely decays across 2.5 decades\n"
                         f"Landscape is fractal down to r = {ZOOM_SCALES[-1]}")
    elif mean_slope < 0.5:
        smooth_verdict = "WEAK smoothness emerges"
        smooth_detail = (f"Mean slope = {mean_slope:.3f}\n"
                         f"Some decay but structure persists\n"
                         f"Partial fractal behavior")
    else:
        smooth_verdict = "Smoothness floor found"
        smooth_detail = (f"Mean slope = {mean_slope:.3f} (near 1)\n"
                         f"DX range decays proportionally with scale\n"
                         f"Landscape becomes differentiable")

    # Isotropy verdict: low std/mean ratio + gradient ratio near 1 = isotropic.
    # Gradient ratio >> 1 means cliff structure is directionally concentrated.
    if mean_iso < 0.4 and abs(mean_gvp - 1) < 0.5:
        iso_verdict = "ISOTROPIC chaos"
        iso_detail = "Comparable variation in all directions\nTrue Type 3: no preferred axis"
    elif mean_gvp > 1.5:
        iso_verdict = "ANISOTROPIC — gradient dominant"
        iso_detail = "Gradient direction is rougher\nType 4 tendencies: directional structure"
    else:
        iso_verdict = "MIXED isotropy"
        iso_detail = "Some directional preference\nBorderline Type 3/4"

    # Wolfram class estimation from combined fractal slope and isotropy.
    # Type 3 = chaotic (scale-invariant, isotropic): no local optimizer works.
    # Type 3/4 boundary = chaotic but anisotropic: directional search may help.
    # Type 2->3 transition = some smoothness at fine scales: local gradient
    # descent becomes viable, suggesting multi-scale optimization strategy.
    if mean_slope < 0.3 and mean_iso < 0.5:
        wolfram = "TYPE 3 (Chaotic)"
        wolfram_detail = ("Scale-invariant + isotropic = pure chaos\n"
                          "No gradient descent viable at any scale\n"
                          "Optimization requires global search")
    elif mean_slope < 0.3:
        wolfram = "TYPE 3/4 boundary"
        wolfram_detail = ("Scale-invariant but anisotropic\n"
                          "Directional structure within chaos\n"
                          "Exploitable by directional search")
    else:
        wolfram = "TYPE 2→3 transition"
        wolfram_detail = ("Some smoothness at fine scales\n"
                          "Gradient descent may work locally\n"
                          "Multi-scale optimization recommended")

    verdict_text = (
        f"SMOOTHNESS VERDICT\n"
        f"{'─' * 40}\n"
        f"{smooth_verdict}\n{smooth_detail}\n\n"
        f"ISOTROPY VERDICT\n"
        f"{'─' * 40}\n"
        f"{iso_verdict}\n{iso_detail}\n\n"
        f"WOLFRAM CLASSIFICATION\n"
        f"{'─' * 40}\n"
        f"{wolfram}\n{wolfram_detail}"
    )
    ax.text(0.05, 0.95, verdict_text, transform=ax.transAxes,
            fontsize=11, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#333",
                      alpha=0.9))

    fig.suptitle("Deep Resolution Verdict — Step Zone Structure",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "deep_fig06_smoothness_verdict.png")

    return {
        "mean_fractal_slope": float(mean_slope),
        "mean_isotropy": float(mean_iso),
        "mean_grad_vs_perp": float(mean_gvp),
        "smoothness_verdict": smooth_verdict,
        "isotropy_verdict": iso_verdict,
        "wolfram_class": wolfram,
    }


# ── Console Output ──────────────────────────────────────────────────────────

def print_analysis(zoom_results, fan_results, grid_results, targets, verdicts):
    """Print a formatted console summary of all three phases and final verdicts."""
    print(f"\n{'='*80}")
    print("DEEP RESOLUTION — RESULTS")
    print(f"{'='*80}")

    # Phase 1 summary
    print(f"\n  PHASE 1: FRACTAL SCALING")
    print(f"    {'Target':>7} {'Slope':>7}", end="")
    for s in ZOOM_SCALES:
        print(f" {'r='+str(s):>12}", end="")
    print()
    print("    " + "-" * (16 + 13 * len(ZOOM_SCALES)))
    for ti, zr in enumerate(zoom_results):
        scales = [sp["scale"] for sp in zr["scale_profiles"]]
        ranges = [sp["dx_range"] for sp in zr["scale_profiles"]]
        ranges_c = [max(r, 1e-12) for r in ranges]
        slope = np.polyfit(np.log10(scales), np.log10(ranges_c), 1)[0]
        print(f"    #{targets[ti]['select_rank']+1:5d} {slope:7.3f}", end="")
        for sp in zr["scale_profiles"]:
            print(f" {sp['dx_range']:12.3f}", end="")
        print()

    # Derivative scaling
    print(f"\n    MEAN |dDX/dr| BY SCALE:")
    for s in ZOOM_SCALES:
        derivs = []
        for zr in zoom_results:
            for sp in zr["scale_profiles"]:
                if sp["scale"] == s:
                    derivs.append(sp["mean_abs_deriv"])
        print(f"      r=±{s:<10} mean|deriv| = {np.mean(derivs):>12.1f}")

    # Phase 2 summary
    print(f"\n  PHASE 2: DIRECTIONAL ISOTROPY")
    for ti, fr in enumerate(fan_results):
        ranges = [fp["dx_range"] for fp in fr["fan_profiles"]]
        grad_range = fr["gradient_profile"]["dx_range"]
        mean_r = np.mean(ranges)
        std_r = np.std(ranges)
        print(f"    #{targets[ti]['select_rank']+1}: "
              f"perp mean={mean_r:.2f}m std={std_r:.2f}m "
              f"grad={grad_range:.2f}m  ratio={grad_range/max(mean_r,1e-12):.2f}")

    # Phase 3 summary
    print(f"\n  PHASE 3: 2D MICRO-GRIDS")
    for gi, gr in enumerate(grid_results):
        dx_g = np.array(gr["dx_grid"])
        cliff_g = np.array(gr["cliff_grid"])
        print(f"    #{gr['select_rank']+1}: "
              f"DX range=[{dx_g.min():+.1f}, {dx_g.max():+.1f}]  "
              f"cliff mean={cliff_g.mean():.0f} max={cliff_g.max():.0f}")

    # Verdicts
    print(f"\n  {'='*60}")
    print(f"  VERDICTS")
    print(f"  {'='*60}")
    print(f"    Fractal slope:  {verdicts['mean_fractal_slope']:.3f}  → {verdicts['smoothness_verdict']}")
    print(f"    Isotropy:       {verdicts['mean_isotropy']:.3f}  → {verdicts['isotropy_verdict']}")
    print(f"    Grad/perp:      {verdicts['mean_grad_vs_perp']:.3f}")
    print(f"    Wolfram class:  {verdicts['wolfram_class']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run all three deep-probing phases, generate figures, and save results."""
    t_start = time.perf_counter()
    np.random.seed(123)

    # Back up brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    # Load taxonomy data
    print(f"Loading {TAX_JSON} ...")
    with open(TAX_JSON) as f:
        tax_data = json.load(f)
    print(f"  Loaded {len(tax_data['profiles'])} profiles")

    # Select targets
    print(f"\n{'='*80}")
    print("TARGET SELECTION — 10 Most Chaotic Steps")
    print(f"{'='*80}")
    targets = select_targets(tax_data)

    # Budget: Phase1 (10 targets x 6 scales x 20 pts) + Phase2 (10 x (8 dirs + grad) x 9 pts) + Phase3 (5 x 10x10)
    budget = 10 * 6 * N_ZOOM_PTS + 10 * (N_FAN_DIRS + 1) * N_FAN_PTS + 5 * GRID_N * GRID_N
    print(f"\n  Total simulation budget: ~{budget} sims")

    # Phase 1
    zoom_results = phase1_zoom_cascade(targets)

    # Phase 2
    fan_results = phase2_directional_fan(targets)

    # Phase 3
    grid_results = phase3_micro_grid(targets)

    # Restore brain.nndf
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # Figures
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")
    fig01_zoom_cascade(zoom_results, targets)
    slopes = fig02_fractal_dimension(zoom_results, targets)
    fig03_directional_fan(fan_results, targets)
    isotropy_ratios, grad_vs_perp = fig04_isotropy(fan_results, targets)
    fig05_micro_grids(grid_results, targets)
    verdicts = fig06_smoothness_verdict(zoom_results, fan_results, grid_results,
                                        targets, slopes, isotropy_ratios, grad_vs_perp)

    # Console
    print_analysis(zoom_results, fan_results, grid_results, targets, verdicts)

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "zoom_scales": ZOOM_SCALES,
            "n_zoom_pts": N_ZOOM_PTS,
            "n_fan_dirs": N_FAN_DIRS,
            "n_fan_pts": N_FAN_PTS,
            "fan_radius": FAN_RADIUS,
            "grid_n": GRID_N,
            "grid_radius": GRID_RADIUS,
        },
        "targets": targets,
        "zoom_results": zoom_results,
        "fan_results": fan_results,
        "grid_results": grid_results,
        "verdicts": verdicts,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
