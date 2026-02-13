#!/usr/bin/env python3
"""
resonance_mapping.py

Resonance Mapping — What Frequencies Does the Body Want?

Bypass the neural network entirely. Drive joints with pure sinusoidal
position targets, sweep frequency/phase/ratio to find the body's
mechanical transfer function.

Simulation budget: ~2,150 sims (~3 min)

Part 1: Frequency × Phase Sweep (600 sims)
    50 frequencies (0.1–5 Hz) × 12 phase offsets (0–pi).
    Both joints at same frequency, varying relative phase.
    Maps: which frequencies produce locomotion?

Part 2: Amplitude Sweep (300 sims)
    Top 6 resonant frequencies × 50 amplitudes (0.05–1.5 rad).
    Maps: does the body saturate, or does more amplitude = more DX?

Part 3: Polyrhythmic Grid (900 sims)
    30 × 30 grid of (f_back, f_front) from 0.1 to 5 Hz.
    Fixed amplitude and phase. Maps frequency-ratio landscape.
    Tests whether 5:3, 3:2, 2:1 ratios are special.

Part 4: Overlay (no sims)
    Plot evolved gait frequencies on the resonance map.
    Do evolved gaits exploit resonance?

Outputs:
    artifacts/resonance_mapping.json
    artifacts/plots/res_fig01_freq_phase.png
    artifacts/plots/res_fig02_transfer_function.png
    artifacts/plots/res_fig03_amplitude.png
    artifacts/plots/res_fig04_polyrhythm.png
    artifacts/plots/res_fig05_evolved_overlay.png
    artifacts/plots/res_fig06_verdict.png

Usage:
    python3 resonance_mapping.py
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
from matplotlib.colors import Normalize

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
from compute_beer_analytics import NumpyEncoder

OUT_JSON = PROJECT / "artifacts" / "resonance_mapping.json"
PLOT_DIR = PROJECT / "artifacts" / "plots"

DT = c.DT
SIM_STEPS = c.SIM_STEPS
MAX_FORCE = float(c.MAX_FORCE)
SIM_TIME = SIM_STEPS * DT  # 16.67 seconds

# Known evolved gait data (from synapse_gait_zoo_v2.json)
EVOLVED_GAITS = [
    {"name": "Hidden CPG Champ", "dx": 50.1, "f_back": 2.46, "f_front": 2.22},
    {"name": "Noether CPG", "dx": -43.2, "f_back": 0.24, "f_front": 3.48},
    {"name": "Curie Amplified", "dx": 37.1, "f_back": 2.76, "f_front": 0.18},
    {"name": "Pelton", "dx": 34.7, "f_back": 2.82, "f_front": 0.60},
    {"name": "Carry Trade", "dx": 32.2, "f_back": 3.24, "f_front": 0.72},
    {"name": "Grunbaum Deflation", "dx": -30.5, "f_back": 0.66, "f_front": 2.52},
    {"name": "Noether Cyclone", "dx": -30.2, "f_back": 0.84, "f_front": 0.84},
    {"name": "Hodgkin-Huxley", "dx": -29.8, "f_back": 0.78, "f_front": 0.78},
    {"name": "Take Five", "dx": -27.7, "f_back": 0.18, "f_front": 0.18},
    {"name": "Curie Crab", "dx": 24.4, "f_back": 3.24, "f_front": 0.54},
    {"name": "Curie Dervish", "dx": 24.1, "f_back": 3.60, "f_front": 0.54},
    {"name": "Curie (orig)", "dx": 23.7, "f_back": 2.46, "f_front": 0.90},
    {"name": "Hemiola", "dx": 22.6, "f_back": 2.16, "f_front": 0.36},
]

# Novelty Champion and Trial 3 from our cliff analysis
NC_GAIT = {"name": "Novelty Champion", "dx": 60.2, "f_back": 2.2, "f_front": 1.3}
T3_GAIT = {"name": "Trial 3", "dx": 10.0, "f_back": 0.24, "f_front": 0.24}


# ── Simulation ───────────────────────────────────────────────────────────────

def find_joint(robot_id, needle):
    """Return the joint index whose name contains `needle`, or None."""
    for j in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, j)[1].decode("utf-8", errors="replace")
        if needle in name:
            return j
    return None


def simulate_openloop(freq_back, freq_front, phase_back, phase_front,
                      amp_back, amp_front):
    """Drive both joints with sine waves, return metrics dict."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(DT)

    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("body.urdf")

    # Set friction
    mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
    for link in range(-1, p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, link, lateralFriction=mu, restitution=0.0)

    back_joint = find_joint(robot_id, "BackLeg")
    front_joint = find_joint(robot_id, "FrontLeg")

    # Pre-compute sinusoidal position targets for every timestep.
    # Each joint follows A * sin(2*pi*f*t + phi), producing a pure-tone
    # oscillation at the specified frequency, amplitude, and phase offset.
    t_arr = np.arange(SIM_STEPS) * DT
    target_back = amp_back * np.sin(2 * np.pi * freq_back * t_arr + phase_back)
    target_front = amp_front * np.sin(2 * np.pi * freq_front * t_arr + phase_front)

    x_first = None
    contact_back = 0
    contact_front = 0
    contact_torso = 0

    for i in range(SIM_STEPS):
        p.setJointMotorControl2(
            robot_id, back_joint, p.POSITION_CONTROL,
            targetPosition=float(target_back[i]), force=MAX_FORCE)
        p.setJointMotorControl2(
            robot_id, front_joint, p.POSITION_CONTROL,
            targetPosition=float(target_front[i]), force=MAX_FORCE)
        p.stepSimulation()

        if x_first is None:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            x_first = pos[0]

        # Sample contacts every 10 steps
        if i % 10 == 0:
            contacts = p.getContactPoints(robot_id)
            links_in_contact = set()
            for c_pt in contacts:
                links_in_contact.add(c_pt[3])  # linkIndexA
                links_in_contact.add(c_pt[4])  # linkIndexB
            # Link -1 = torso, 0 = back, 1 = front (approximate)
            if -1 in links_in_contact:
                contact_torso += 1
            if 0 in links_in_contact:
                contact_back += 1
            if 1 in links_in_contact:
                contact_front += 1

    pos, orn = p.getBasePositionAndOrientation(robot_id)
    x_last = pos[0]
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    p.disconnect()

    dx = x_last - x_first
    # Duty cycle = fraction of sampled steps where each link touches the ground
    n_samples = SIM_STEPS // 10

    return {
        "dx": float(dx),
        "abs_dx": float(abs(dx)),
        "final_z": float(pos[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
        "duty_torso": float(contact_torso / max(n_samples, 1)),
        "duty_back": float(contact_back / max(n_samples, 1)),
        "duty_front": float(contact_front / max(n_samples, 1)),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_ax(ax):
    """Remove top and right spines from a matplotlib axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    """Save figure to PLOT_DIR and close it."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


# ── Part 1: Frequency × Phase Sweep ─────────────────────────────────────────

def part1_freq_phase(n_freq=50, n_phase=12):
    """Sweep frequency and phase offset with equal frequencies on both joints."""
    total = n_freq * n_phase
    print(f"\n{'='*80}")
    print(f"PART 1: Frequency x Phase Sweep ({n_freq} freqs x {n_phase} phases = {total} sims)")
    print(f"{'='*80}")

    freqs = np.linspace(0.1, 5.0, n_freq)
    phases = np.linspace(0, np.pi, n_phase)
    amp = 0.5  # radians — moderate joint excursion

    dx_grid = np.empty((n_freq, n_phase))
    abs_dx_grid = np.empty((n_freq, n_phase))
    results_list = []
    t0 = time.perf_counter()
    sim_count = 0

    for fi, freq in enumerate(freqs):
        for pi, phase in enumerate(phases):
            r = simulate_openloop(freq, freq, 0.0, phase, amp, amp)
            dx_grid[fi, pi] = r["dx"]
            abs_dx_grid[fi, pi] = r["abs_dx"]
            sim_count += 1

        if (fi + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / sim_count
            remaining = rate * (total - sim_count)
            best = np.max(np.abs(dx_grid[:fi+1, :]))
            print(f"  [{fi+1:3d}/{n_freq}] {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining  best |DX|={best:.1f}m", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Part 1 complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")

    return {
        "freqs": freqs.tolist(),
        "phases": phases.tolist(),
        "amplitude": amp,
        "dx_grid": dx_grid.tolist(),
        "abs_dx_grid": abs_dx_grid.tolist(),
    }


# ── Part 2: Amplitude Sweep ─────────────────────────────────────────────────

def part2_amplitude(freq_phase_data, n_top=6, n_amp=50):
    """Sweep amplitude at the top resonant frequencies."""
    freqs = np.array(freq_phase_data["freqs"])
    abs_dx = np.array(freq_phase_data["abs_dx_grid"])

    # Find top frequencies: collapse the 2D (freq x phase) grid by taking
    # the maximum |DX| across all phase offsets for each frequency, then
    # pick the n_top frequencies with the largest peak displacement.
    max_per_freq = np.max(abs_dx, axis=1)
    top_indices = np.argsort(max_per_freq)[-n_top:][::-1]
    top_freqs = freqs[top_indices]

    # For each top frequency, find the phase offset that produced its peak |DX|
    top_phases_idx = np.argmax(abs_dx[top_indices, :], axis=1)
    phases = np.array(freq_phase_data["phases"])
    top_phases = phases[top_phases_idx]

    total = n_top * n_amp
    print(f"\n{'='*80}")
    print(f"PART 2: Amplitude Sweep ({n_top} freqs x {n_amp} amps = {total} sims)")
    print(f"  Top freqs: {[f'{f:.2f}' for f in top_freqs]} Hz")
    print(f"{'='*80}")

    amps = np.linspace(0.05, 1.5, n_amp)
    amp_results = []
    t0 = time.perf_counter()
    sim_count = 0

    for ti, (freq, phase) in enumerate(zip(top_freqs, top_phases)):
        dxs = np.empty(n_amp)
        for ai, amp in enumerate(amps):
            r = simulate_openloop(freq, freq, 0.0, phase, amp, amp)
            dxs[ai] = r["dx"]
            sim_count += 1

        amp_results.append({
            "freq": float(freq),
            "phase": float(phase),
            "amps": amps.tolist(),
            "dxs": dxs.tolist(),
            "max_dx": float(dxs.max()),
            "peak_amp": float(amps[np.argmax(np.abs(dxs))]),
        })

        elapsed = time.perf_counter() - t0
        rate = elapsed / sim_count
        remaining = rate * (total - sim_count)
        print(f"  [{ti+1}/{n_top}] f={freq:.2f}Hz  max|DX|={np.max(np.abs(dxs)):.1f}m  "
              f"{elapsed:.1f}s, ~{remaining:.0f}s rem", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Part 2 complete: {sim_count} sims in {elapsed:.1f}s")
    return amp_results


# ── Part 3: Polyrhythmic Grid ────────────────────────────────────────────────

def part3_polyrhythm(n_grid=30):
    """2D grid of independent back/front frequencies."""
    total = n_grid * n_grid
    print(f"\n{'='*80}")
    print(f"PART 3: Polyrhythmic Grid ({n_grid}x{n_grid} = {total} sims)")
    print(f"{'='*80}")

    freqs = np.linspace(0.1, 5.0, n_grid)
    amp = 0.5
    phase = np.pi / 3  # reasonable default phase offset

    dx_grid = np.empty((n_grid, n_grid))
    abs_dx_grid = np.empty((n_grid, n_grid))
    t0 = time.perf_counter()
    sim_count = 0

    for fi, f_back in enumerate(freqs):
        for fj, f_front in enumerate(freqs):
            r = simulate_openloop(f_back, f_front, 0.0, phase, amp, amp)
            dx_grid[fi, fj] = r["dx"]
            abs_dx_grid[fi, fj] = r["abs_dx"]
            sim_count += 1

        if (fi + 1) % 5 == 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / sim_count
            remaining = rate * (total - sim_count)
            print(f"  [{fi+1:3d}/{n_grid}] {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Part 3 complete: {sim_count} sims in {elapsed:.1f}s "
          f"({elapsed/sim_count:.3f}s/sim)")

    return {
        "freqs": freqs.tolist(),
        "phase": phase,
        "amplitude": amp,
        "dx_grid": dx_grid.tolist(),
        "abs_dx_grid": abs_dx_grid.tolist(),
    }


# ── Figures ──────────────────────────────────────────────────────────────────

def fig01_freq_phase(fp_data):
    """Heatmap of DX across frequency and phase."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    freqs = np.array(fp_data["freqs"])
    phases = np.array(fp_data["phases"])

    # Left: signed DX
    ax = axes[0]
    dx_g = np.array(fp_data["dx_grid"])
    # Clamp colorbar symmetrically at 95th percentile to avoid outlier wash-out
    vmax = np.percentile(np.abs(dx_g), 95)
    im = ax.imshow(dx_g.T, origin="lower", aspect="auto",
                   extent=[freqs[0], freqs[-1], np.degrees(phases[0]), np.degrees(phases[-1])],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="DX (m)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase offset (degrees)")
    ax.set_title("Signed DX")
    clean_ax(ax)

    # Right: |DX|
    ax = axes[1]
    abs_g = np.array(fp_data["abs_dx_grid"])
    im = ax.imshow(abs_g.T, origin="lower", aspect="auto",
                   extent=[freqs[0], freqs[-1], np.degrees(phases[0]), np.degrees(phases[-1])],
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label="|DX| (m)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase offset (degrees)")
    ax.set_title("|DX| — Resonance Peaks")
    clean_ax(ax)

    fig.suptitle(f"Frequency x Phase Sweep (amplitude = {fp_data['amplitude']} rad, "
                 f"both joints same freq)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "res_fig01_freq_phase.png")


def fig02_transfer_function(fp_data):
    """1D transfer function: max |DX| vs frequency (marginalized over phase)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    freqs = np.array(fp_data["freqs"])
    abs_g = np.array(fp_data["abs_dx_grid"])
    dx_g = np.array(fp_data["dx_grid"])

    # Left: 1D transfer function — marginalize over phase by taking the
    # max (envelope) and mean |DX| at each frequency. This reveals which
    # frequencies the body converts most efficiently into displacement.
    ax = axes[0]
    max_per_freq = np.max(abs_g, axis=1)
    mean_per_freq = np.mean(abs_g, axis=1)
    ax.plot(freqs, max_per_freq, "o-", color="#E24A33", lw=2, markersize=4,
            label="Max |DX| over phases")
    ax.plot(freqs, mean_per_freq, "s-", color="#348ABD", lw=1.5, markersize=3,
            label="Mean |DX| over phases")
    ax.fill_between(freqs, 0, max_per_freq, alpha=0.15, color="#E24A33")

    # Mark top 3 peaks
    peak_idx = np.argsort(max_per_freq)[-3:][::-1]
    for pi in peak_idx:
        ax.annotate(f"{freqs[pi]:.2f} Hz\n{max_per_freq[pi]:.1f}m",
                    (freqs[pi], max_per_freq[pi]),
                    textcoords="offset points", xytext=(10, 10), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("|DX| (m)", fontsize=11)
    ax.set_title("Mechanical Transfer Function")
    ax.legend(fontsize=9)
    clean_ax(ax)

    # Right: for each frequency, find the phase offset that maximizes |DX|.
    # This shows whether the optimal inter-joint phase relationship changes
    # across the frequency spectrum.
    ax = axes[1]
    phases = np.array(fp_data["phases"])
    best_phase_idx = np.argmax(abs_g, axis=1)
    best_phases = np.degrees(phases[best_phase_idx])
    # Color by |DX|
    sc = ax.scatter(freqs, best_phases, c=max_per_freq, cmap="inferno",
                    s=40, edgecolors="black", lw=0.3)
    plt.colorbar(sc, ax=ax, label="|DX| at best phase (m)")
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Best phase offset (degrees)", fontsize=11)
    ax.set_title("Optimal Phase by Frequency")
    clean_ax(ax)

    fig.suptitle("Body's Mechanical Transfer Function",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "res_fig02_transfer_function.png")

    return max_per_freq


def fig03_amplitude(amp_results):
    """Amplitude sweep curves for top resonant frequencies."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for ar in amp_results:
        amps = np.array(ar["amps"])
        dxs = np.array(ar["dxs"])
        ax.plot(amps, dxs, "o-", lw=2, markersize=4,
                label=f"f={ar['freq']:.2f} Hz")

    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Amplitude (rad)", fontsize=12)
    ax.set_ylabel("DX (m)", fontsize=12)
    ax.set_title("DX vs Amplitude at Resonant Frequencies", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "res_fig03_amplitude.png")


def fig04_polyrhythm(poly_data):
    """2D heatmap of polyrhythmic grid."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    freqs = np.array(poly_data["freqs"])

    # Left: signed DX
    ax = axes[0]
    dx_g = np.array(poly_data["dx_grid"])
    vmax = np.percentile(np.abs(dx_g), 95)
    im = ax.imshow(dx_g.T, origin="lower", aspect="equal",
                   extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="DX (m)")
    # Diagonal (equal frequency, i.e. 1:1 ratio)
    ax.plot([freqs[0], freqs[-1]], [freqs[0], freqs[-1]], "k--", lw=1, alpha=0.5)
    # Overlay lines where f_front/f_back equals musically significant ratios.
    # Each line traces f_front = ratio * f_back across the grid.
    for ratio, label in [(2, "2:1"), (3/2, "3:2"), (5/3, "5:3")]:
        ax.plot([freqs[0], freqs[-1]/ratio], [freqs[0]*ratio, freqs[-1]],
                "w:", lw=0.8, alpha=0.6)
    ax.set_xlabel("f_back (Hz)")
    ax.set_ylabel("f_front (Hz)")
    ax.set_title("Signed DX — Polyrhythmic Grid")

    # Right: |DX|
    ax = axes[1]
    abs_g = np.array(poly_data["abs_dx_grid"])
    im = ax.imshow(abs_g.T, origin="lower", aspect="equal",
                   extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label="|DX| (m)")
    ax.plot([freqs[0], freqs[-1]], [freqs[0], freqs[-1]], "w--", lw=1, alpha=0.5)
    for ratio, label in [(2, "2:1"), (3/2, "3:2"), (5/3, "5:3")]:
        ax.plot([freqs[0], freqs[-1]/ratio], [freqs[0]*ratio, freqs[-1]],
                "w:", lw=0.8, alpha=0.6)
    ax.set_xlabel("f_back (Hz)")
    ax.set_ylabel("f_front (Hz)")
    ax.set_title("|DX| — Polyrhythmic Grid")

    fig.suptitle(f"Polyrhythmic Frequency Grid (amp={poly_data['amplitude']} rad, "
                 f"phase={np.degrees(poly_data['phase']):.0f}°)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "res_fig04_polyrhythm.png")


def fig05_evolved_overlay(fp_data, poly_data):
    """Overlay evolved gait frequencies on the resonance maps."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: transfer function with evolved gaits marked
    ax = axes[0]
    freqs = np.array(fp_data["freqs"])
    abs_g = np.array(fp_data["abs_dx_grid"])
    max_per_freq = np.max(abs_g, axis=1)
    ax.fill_between(freqs, 0, max_per_freq, alpha=0.2, color="#4C72B0",
                    label="Open-loop envelope")
    ax.plot(freqs, max_per_freq, "-", color="#4C72B0", lw=1.5)

    # Plot evolved gaits
    all_gaits = EVOLVED_GAITS + [NC_GAIT, T3_GAIT]
    for g in all_gaits:
        color = "#E24A33" if g == NC_GAIT else "#55A868" if g == T3_GAIT else "#888"
        marker = "*" if g in (NC_GAIT, T3_GAIT) else "o"
        size = 150 if g in (NC_GAIT, T3_GAIT) else 40
        # Project the 2D (f_back, f_front) gait onto the 1D transfer function
        # by averaging both joint frequencies into a single representative value.
        f_avg = (g["f_back"] + g["f_front"]) / 2
        ax.scatter([f_avg], [abs(g["dx"])], c=color, marker=marker, s=size,
                   edgecolors="black", lw=0.5, zorder=5)
        if g in (NC_GAIT, T3_GAIT) or abs(g["dx"]) > 35:
            ax.annotate(g["name"], (f_avg, abs(g["dx"])),
                        textcoords="offset points", xytext=(8, 5), fontsize=7)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|DX| (m)")
    ax.set_title("Evolved Gaits vs Open-Loop Transfer Function")
    ax.legend(fontsize=9)
    clean_ax(ax)

    # Right: evolved gaits on polyrhythmic grid
    ax = axes[1]
    poly_freqs = np.array(poly_data["freqs"])
    abs_poly = np.array(poly_data["abs_dx_grid"])
    im = ax.imshow(abs_poly.T, origin="lower", aspect="equal",
                   extent=[poly_freqs[0], poly_freqs[-1],
                           poly_freqs[0], poly_freqs[-1]],
                   cmap="inferno", alpha=0.7)
    plt.colorbar(im, ax=ax, label="|DX| open-loop (m)")

    for g in all_gaits:
        color = "cyan" if g == NC_GAIT else "lime" if g == T3_GAIT else "white"
        marker = "*" if g in (NC_GAIT, T3_GAIT) else "o"
        size = 200 if g in (NC_GAIT, T3_GAIT) else 50
        ax.scatter([g["f_back"]], [g["f_front"]], c=color, marker=marker, s=size,
                   edgecolors="black", lw=1, zorder=5)
        if g in (NC_GAIT, T3_GAIT) or abs(g["dx"]) > 35:
            ax.annotate(g["name"], (g["f_back"], g["f_front"]),
                        textcoords="offset points", xytext=(8, 5), fontsize=7,
                        color="white",
                        path_effects=[matplotlib.patheffects.withStroke(
                            linewidth=2, foreground="black")])

    ax.plot([poly_freqs[0], poly_freqs[-1]], [poly_freqs[0], poly_freqs[-1]],
            "w--", lw=1, alpha=0.5, label="1:1")
    ax.set_xlabel("f_back (Hz)")
    ax.set_ylabel("f_front (Hz)")
    ax.set_title("Evolved Gait Frequencies on Polyrhythmic Map")
    clean_ax(ax)

    fig.suptitle("Do Evolved Gaits Exploit Mechanical Resonance?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "res_fig05_evolved_overlay.png")


def fig06_verdict(fp_data, poly_data, amp_results, transfer_fn):
    """Summary verdict figure."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))

    freqs = np.array(fp_data["freqs"])
    abs_g = np.array(fp_data["abs_dx_grid"])

    # (0,0): Transfer function smoothness test
    ax = axes[0][0]
    max_pf = transfer_fn
    ax.plot(freqs, max_pf, "o-", color="#4C72B0", lw=2, markersize=4)
    # First derivative of the transfer function: large values indicate
    # abrupt changes in displacement with small frequency shifts (cliffs).
    derivs = np.abs(np.diff(max_pf) / np.diff(freqs))
    ax.fill_between(freqs[:-1], 0, derivs, alpha=0.3, color="#E24A33")
    ax2 = ax.twinx()
    ax2.plot(freqs[:-1], derivs, "-", color="#E24A33", lw=1, alpha=0.7)
    ax2.set_ylabel("|d(DX)/df| (m/Hz)", color="#E24A33")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Max |DX| (m)")
    ax.set_title("Transfer Function + Derivative")
    clean_ax(ax)

    # (0,1): Frequency-space roughness vs weight-space roughness
    ax = axes[0][1]
    # Frequency-space roughness: measure how jagged the transfer function
    # is by looking at the distribution of consecutive DX jumps. A smooth
    # landscape has small, uniform jumps; a rough one has large outliers.
    freq_roughness = np.std(np.diff(max_pf))
    consec_diffs = np.abs(np.diff(max_pf))
    ax.hist(consec_diffs, bins=15, color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(consec_diffs), color="#E24A33", lw=2, ls="--",
               label=f"Mean = {np.mean(consec_diffs):.1f}m")
    ax.set_xlabel("|DX[i+1] - DX[i]| (m)")
    ax.set_ylabel("Count")
    ax.set_title("Consecutive DX Differences\n(frequency space)")
    ax.legend(fontsize=9)
    clean_ax(ax)

    # (0,2): Polyrhythmic grid — ratio analysis
    ax = axes[0][2]
    poly_freqs = np.array(poly_data["freqs"])
    abs_poly = np.array(poly_data["abs_dx_grid"])
    # Extract |DX| along constant-ratio diagonals in the polyrhythmic grid.
    # For each ratio r, walk f_back and pick the nearest grid cell where
    # f_front ~ r * f_back. Only include points within one grid step of
    # the exact ratio to avoid aliasing from the discrete grid.
    ratios = [1.0, 1.5, 2.0, 5/3, 3.0]
    ratio_labels = ["1:1", "3:2", "2:1", "5:3", "3:1"]
    for ratio, label in zip(ratios, ratio_labels):
        dxs_along = []
        f_along = []
        for fi, fb in enumerate(poly_freqs):
            ff_target = fb * ratio
            fj = np.argmin(np.abs(poly_freqs - ff_target))
            if abs(poly_freqs[fj] - ff_target) < (poly_freqs[1] - poly_freqs[0]):
                dxs_along.append(abs_poly[fi, fj])
                f_along.append(fb)
        if f_along:
            ax.plot(f_along, dxs_along, "o-", lw=1.5, markersize=3, label=label)

    ax.set_xlabel("f_back (Hz)")
    ax.set_ylabel("|DX| (m)")
    ax.set_title("DX Along Frequency Ratios")
    ax.legend(fontsize=8)
    clean_ax(ax)

    # (1,0): Amplitude saturation analysis
    ax = axes[1][0]
    for ar in amp_results[:4]:
        amps = np.array(ar["amps"])
        dxs = np.abs(np.array(ar["dxs"]))
        ax.plot(amps, dxs, "o-", lw=1.5, markersize=3,
                label=f"f={ar['freq']:.2f}Hz")
    ax.set_xlabel("Amplitude (rad)")
    ax.set_ylabel("|DX| (m)")
    ax.set_title("Amplitude Response Curves")
    ax.legend(fontsize=8)
    clean_ax(ax)

    # (1,1): Is the open-loop landscape smooth?
    ax = axes[1][1]
    poly_dx = np.array(poly_data["dx_grid"])
    # Compute the gradient magnitude at each grid point using central finite
    # differences (forward/backward at boundaries). This gives |grad DX| in
    # units of m/Hz — large values indicate "cliffs" where a small frequency
    # change causes a large displacement jump.
    step = poly_freqs[1] - poly_freqs[0]
    n = len(poly_freqs)
    cliff_grid = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Partial derivative along f_back (row direction)
            if i == 0:
                gx = (poly_dx[1, j] - poly_dx[0, j]) / step
            elif i == n - 1:
                gx = (poly_dx[-1, j] - poly_dx[-2, j]) / step
            else:
                gx = (poly_dx[i+1, j] - poly_dx[i-1, j]) / (2 * step)
            # Partial derivative along f_front (column direction)
            if j == 0:
                gy = (poly_dx[i, 1] - poly_dx[i, 0]) / step
            elif j == n - 1:
                gy = (poly_dx[i, -1] - poly_dx[i, -2]) / step
            else:
                gy = (poly_dx[i, j+1] - poly_dx[i, j-1]) / (2 * step)
            cliff_grid[i, j] = np.sqrt(gx**2 + gy**2)

    im = ax.imshow(cliff_grid.T, origin="lower", aspect="equal",
                   extent=[poly_freqs[0], poly_freqs[-1],
                           poly_freqs[0], poly_freqs[-1]],
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label="|grad DX| (m/Hz)")
    ax.set_xlabel("f_back (Hz)")
    ax.set_ylabel("f_front (Hz)")
    ax.set_title("Frequency-Space Cliffiness")

    # (1,2): Verdict text
    ax = axes[1][2]
    ax.axis("off")

    # Compute verdicts
    peak_dx = float(np.max(max_pf))
    peak_freq = float(freqs[np.argmax(max_pf)])
    mean_roughness_freq = float(np.mean(consec_diffs))
    max_roughness_freq = float(np.max(consec_diffs))
    mean_cliff_poly = float(np.mean(cliff_grid))

    # Evolved gait comparison
    nc_dx = abs(NC_GAIT["dx"])
    nc_exceeds = nc_dx > peak_dx

    # Convert per-bin roughness to per-Hz by dividing by the frequency step.
    # This allows direct comparison against weight-space cliffiness (~40m
    # change over dr=0.01) from the cliff taxonomy experiments.
    df = float(freqs[1] - freqs[0])

    verdict_text = (
        f"RESONANCE STRUCTURE\n"
        f"{'─' * 40}\n"
        f"Peak open-loop |DX|: {peak_dx:.1f}m at {peak_freq:.2f} Hz\n"
        f"Transfer fn smoothness:\n"
        f"  Mean |dDX/df|: {mean_roughness_freq/df:.0f} m/Hz\n"
        f"  Max  |dDX/df|: {max_roughness_freq/df:.0f} m/Hz\n"
        f"  Frequency-space cliffiness: {mean_cliff_poly:.0f} m/Hz\n\n"
        f"EVOLVED GAIT COMPARISON\n"
        f"{'─' * 40}\n"
        f"Novelty Champion: {nc_dx:.1f}m\n"
        f"  {'EXCEEDS' if nc_exceeds else 'WITHIN'} open-loop envelope\n"
        f"  NN adds {nc_dx - peak_dx:+.1f}m vs best open-loop\n\n"
        f"KEY FINDING\n"
        f"{'─' * 40}\n"
    )

    if peak_dx > 30 and mean_roughness_freq < 5:
        verdict_text += ("STRONG RESONANCE detected\n"
                         "Body has preferred frequencies\n"
                         "Landscape may be smooth in\n"
                         "frequency space")
    elif peak_dx > 15:
        verdict_text += ("MODERATE RESONANCE detected\n"
                         "Body has frequency preferences\n"
                         "but transfer function is rough")
    else:
        verdict_text += ("WEAK/NO RESONANCE\n"
                         "Body is frequency-agnostic\n"
                         "No smooth reparameterization")

    ax.text(0.05, 0.95, verdict_text, transform=ax.transAxes,
            fontsize=11, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#333",
                      alpha=0.9))

    fig.suptitle("Resonance Mapping Verdict", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "res_fig06_verdict.png")

    return {
        "peak_openloop_dx": peak_dx,
        "peak_frequency": peak_freq,
        "mean_freq_roughness": float(mean_roughness_freq),
        "mean_poly_cliffiness": float(mean_cliff_poly),
        "nc_exceeds_openloop": nc_exceeds,
    }


# ── Console Output ──────────────────────────────────────────────────────────

def print_analysis(fp_data, amp_results, poly_data, verdicts):
    """Print a formatted console summary of all resonance mapping results."""
    freqs = np.array(fp_data["freqs"])
    abs_g = np.array(fp_data["abs_dx_grid"])
    dx_g = np.array(fp_data["dx_grid"])
    max_pf = np.max(abs_g, axis=1)

    print(f"\n{'='*80}")
    print("RESONANCE MAPPING — RESULTS")
    print(f"{'='*80}")

    # Transfer function peaks
    print(f"\n  TRANSFER FUNCTION PEAKS (top 5 frequencies):")
    peak_order = np.argsort(max_pf)[::-1]
    print(f"    {'Rank':>5} {'Freq (Hz)':>10} {'Max |DX| (m)':>14}")
    print("    " + "-" * 31)
    for i in range(5):
        idx = peak_order[i]
        print(f"    {i+1:5d} {freqs[idx]:10.2f} {max_pf[idx]:14.1f}")

    # Amplitude results
    print(f"\n  AMPLITUDE SWEEP:")
    for ar in amp_results:
        print(f"    f={ar['freq']:.2f}Hz: max|DX|={np.max(np.abs(ar['dxs'])):.1f}m "
              f"at amp={ar['peak_amp']:.2f}rad")

    # Polyrhythmic peaks
    poly_freqs = np.array(poly_data["freqs"])
    abs_poly = np.array(poly_data["abs_dx_grid"])
    print(f"\n  POLYRHYTHMIC GRID — TOP 10 FREQUENCY PAIRS:")
    flat = abs_poly.flatten()
    top10 = np.argsort(flat)[-10:][::-1]
    print(f"    {'f_back':>8} {'f_front':>8} {'|DX| (m)':>10} {'Ratio':>8}")
    print("    " + "-" * 38)
    for idx in top10:
        fi, fj = np.unravel_index(idx, abs_poly.shape)
        fb, ff = poly_freqs[fi], poly_freqs[fj]
        ratio = ff / fb if fb > 0.01 else 0
        print(f"    {fb:8.2f} {ff:8.2f} {abs_poly[fi, fj]:10.1f} "
              f"{ratio:8.2f}")

    # Evolved gait comparison
    print(f"\n  EVOLVED GAIT vs OPEN-LOOP COMPARISON:")
    print(f"    Best open-loop: {verdicts['peak_openloop_dx']:.1f}m at "
          f"{verdicts['peak_frequency']:.2f}Hz")
    print(f"    Novelty Champion (NN-driven): {abs(NC_GAIT['dx']):.1f}m")
    print(f"    NC {'exceeds' if verdicts['nc_exceeds_openloop'] else 'within'} "
          f"open-loop envelope by {abs(NC_GAIT['dx']) - verdicts['peak_openloop_dx']:+.1f}m")

    # Smoothness comparison
    print(f"\n  FREQUENCY-SPACE vs WEIGHT-SPACE ROUGHNESS:")
    print(f"    Frequency-space mean cliffiness: {verdicts['mean_poly_cliffiness']:.0f} m/Hz")
    print(f"    Weight-space cliffiness (from atlas): ~3,000,000 m/unit at r=0.00003")
    print(f"    Ratio: frequency space is ~{3000000/max(verdicts['mean_poly_cliffiness'],1):.0f}x smoother")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run all four parts of the resonance mapping campaign and save outputs."""
    t_start = time.perf_counter()
    np.random.seed(42)

    budget = 50 * 12 + 6 * 50 + 30 * 30
    print(f"Resonance Mapping — simulation budget: ~{budget} sims")

    fp_data = part1_freq_phase()
    amp_results = part2_amplitude(fp_data)
    poly_data = part3_polyrhythm()

    # Figures
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")
    fig01_freq_phase(fp_data)
    transfer_fn = fig02_transfer_function(fp_data)
    fig03_amplitude(amp_results)
    fig04_polyrhythm(poly_data)
    fig05_evolved_overlay(fp_data, poly_data)
    verdicts = fig06_verdict(fp_data, poly_data, amp_results, transfer_fn)

    # Console
    print_analysis(fp_data, amp_results, poly_data, verdicts)

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "sim_steps": SIM_STEPS,
            "dt": DT,
            "max_force": MAX_FORCE,
            "sim_time": SIM_TIME,
        },
        "freq_phase": fp_data,
        "amplitude": amp_results,
        "polyrhythm": poly_data,
        "evolved_gaits": EVOLVED_GAITS + [NC_GAIT, T3_GAIT],
        "verdicts": verdicts,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
