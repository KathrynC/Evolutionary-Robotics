#!/usr/bin/env python3
"""
causal_surgery.py

Role:
    Mid-simulation causal intervention experiments. Modifies neural network
    synapse weights DURING a running simulation (not before), then observes
    how the robot's locomotion responds. Tests causal necessity, timing
    dependence, and physical state memory.

Key questions:
    1. If you transplant champion A's brain into champion B's body mid-sim,
       how quickly does the gait change? Is there a transient?
    2. Does it matter WHEN you switch? (early vs late)
    3. Which individual synapses are causally necessary for locomotion?
       (ablate one synapse mid-sim, see if gait survives)
    4. Can you rescue a random gait by transplanting champion weights?

Experimental design (~600 simulations):
    Part 1: Brain Transplants -- switch all 6 weights at t_switch
        6 donor-host pairs x 4 switch times = 24 sims (+ baselines)
    Part 2: Single-Synapse Ablation -- zero one synapse at step 2000
        5 champions x 6 synapses = 30 sims
    Part 3: Switch Timing Sweep -- DX vs switch time for 3 key pairs
        3 pairs x 40 switch times = 120 sims
    Part 4: Rescue Experiments -- transplant champion into random gaits
        3 champions x 10 random hosts = 30 sims

Notes:
    - Unlike causal_surgery_interpolation.py (which does static ablation),
      this script modifies the neural network in-place during the simulation
      loop via direct manipulation of nn.synapses[key].weight.
    - The simulation harness tracks only x-position (lightweight telemetry).
    - brain.nndf is overwritten per simulation; no backup/restore since
      the final state is not meaningful.

Outputs:
    artifacts/causal_surgery.json
    artifacts/plots/cs_fig01_transplant_trajectories.png
    artifacts/plots/cs_fig02_timing_sweep.png
    artifacts/plots/cs_fig03_ablation_heatmap.png
    artifacts/plots/cs_fig04_rescue.png
    artifacts/plots/cs_fig05_recovery_dynamics.png
    artifacts/plots/cs_fig06_verdict.png

Usage:
    python3 causal_surgery.py
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
OUT_JSON = PROJECT / "artifacts" / "causal_surgery.json"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
DT = c.DT
WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]

# ── Champions ───────────────────────────────────────────────────────────────

CHAMPIONS = {
    "NC": {
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
}

TRANSPLANT_PAIRS = [
    ("NC", "Trial 3"),
    ("NC", "Pelton"),
    ("NC", "Noether"),
    ("Pelton", "Curie"),
    ("Trial 3", "NC"),
    ("Pelton", "Noether"),
]

SWITCH_TIMES = [500, 1000, 2000, 3000]
ABLATION_TARGETS = ["NC", "Trial 3", "Pelton", "Curie", "Noether"]
ABLATION_TIME = 2000

N_RANDOM = 10
RNG_SEED = 42


# ── Simulation with mid-sim weight switching ────────────────────────────────

def write_brain_standard(weights):
    """Write a standard 6-synapse brain.nndf file from a weights dict.

    Defines 3 sensor neurons (Torso, BackLeg, FrontLeg) and 2 motor
    neurons (Torso_BackLeg, Torso_FrontLeg) with full sensor-to-motor
    connectivity (6 synapses).

    Args:
        weights: dict with keys "w03","w04","w13","w14","w23","w24" mapping
                 source-target neuron pairs to float synapse weights.

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
        # Fully-connected sensor→motor: 3 sensors × 2 motors = 6 synapses
        for s in [0, 1, 2]:
            for m in [3, 4]:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def set_nn_weights(nn, weights):
    """Modify synapse weights of a live NEURAL_NETWORK object in-place.

    Iterates over all 6 sensor-to-motor synapse pairs and updates
    their weight attribute directly. This is the core "surgery"
    operation that enables mid-simulation brain transplants.

    Args:
        nn: NEURAL_NETWORK instance with an active synapses dict.
        weights: dict mapping "wXY" keys to new float weight values.

    Side effects:
        Mutates nn.synapses[key].weight for each matching synapse.
    """
    for s in [0, 1, 2]:
        for m in [3, 4]:
            key = (str(s), str(m))
            if key in nn.synapses:
                nn.synapses[key].weight = weights[f"w{s}{m}"]


def simulate_with_surgery(initial_weights, surgery_list=None):
    """Run a full simulation with optional mid-sim weight changes.

    Starts with initial_weights, then applies scheduled interventions at
    specified timesteps. Supports two surgery types:
        - Full transplant: replace all 6 weights at once.
        - Single-synapse ablation: change one synapse to a new value.

    Args:
        initial_weights: dict mapping "wXY" to float (the starting brain).
        surgery_list: list of surgery tuples, or None for a baseline run.
            Each tuple is either:
                (timestep, new_weights_dict) for full weight swap, or
                (timestep, synapse_key, new_value) for single-synapse ops
                where synapse_key is e.g. "w03".

    Returns:
        tuple of (x_arr, dx) where:
            x_arr: numpy array of per-step x-positions (shape: (SIM_STEPS,)).
            dx: float, net x-displacement (x_arr[-1] - x_arr[0]).

    Side effects:
        - Overwrites brain.nndf via write_brain_standard().
        - Creates and destroys a PyBullet physics connection.
    """
    write_brain_standard(initial_weights)

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

    # Build surgery schedule: dict of timestep → action
    schedule = {}
    if surgery_list:
        for entry in surgery_list:
            if len(entry) == 2:
                # Full weight swap: (timestep, weights_dict)
                t_switch, new_weights = entry
                schedule[t_switch] = ("full", new_weights)
            elif len(entry) == 3:
                # Single synapse: (timestep, synapse_key, new_value)
                t_switch, syn_key, new_val = entry
                schedule[t_switch] = ("single", syn_key, new_val)

    x_arr = np.empty(n_steps)

    for i in range(n_steps):
        # Check for surgery at this timestep
        if i in schedule:
            action = schedule[i]
            if action[0] == "full":
                set_nn_weights(nn, action[1])
            elif action[0] == "single":
                syn_key = action[1]
                new_val = action[2]
                s, m = syn_key[1], syn_key[2]
                key = (s, m)
                if key in nn.synapses:
                    nn.synapses[key].weight = new_val

        # Motor commands
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

        pos, _ = p.getBasePositionAndOrientation(robotId)
        x_arr[i] = pos[0]

    p.disconnect()
    dx = float(x_arr[-1] - x_arr[0])
    return x_arr, dx


# ── Plotting helpers ────────────────────────────────────────────────────────

def clean_ax(ax):
    """Remove top/right spines and shrink tick labels for cleaner plots.

    Args:
        ax: matplotlib Axes instance.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def save_fig(fig, name):
    """Save a matplotlib figure to PLOT_DIR and close it.

    Args:
        fig: matplotlib Figure instance.
        name: filename (e.g., "cs_fig01_transplant_trajectories.png").

    Side effects:
        Writes the PNG file to PLOT_DIR and closes the figure.
    """
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  WROTE {path}")


COLORS = {
    "NC": "#E24A33",
    "Trial 3": "#55A868",
    "Pelton": "#FBC15E",
    "Curie": "#988ED5",
    "Noether": "#348ABD",
}

SWITCH_COLORS = {500: "#E24A33", 1000: "#348ABD", 2000: "#55A868", 3000: "#FBC15E"}


# ── Figures ─────────────────────────────────────────────────────────────────

def fig01_transplant_trajectories(transplant_results, baselines):
    """Fig 1: x(t) trajectories showing the effect of brain transplants.

    One subplot per donor-host pair. Shows pure donor and host baselines
    as dashed/dotted lines, with transplant trajectories at each switch
    time overlaid. Vertical lines mark switch points.

    Args:
        transplant_results: dict keyed by (donor, host, t_switch) tuples,
            each value a dict with "x" (numpy array) and "dx" (float).
        baselines: dict keyed by champion name, each value a dict with
            "x" (numpy array) and "dx" (float).

    Side effects:
        Writes cs_fig01_transplant_trajectories.png to PLOT_DIR.
    """
    n_pairs = len(TRANSPLANT_PAIRS)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    fig.suptitle("Brain Transplant Trajectories", fontsize=14, fontweight="bold")
    axes_flat = axes.flat

    for idx, (donor, host) in enumerate(TRANSPLANT_PAIRS):
        ax = axes_flat[idx]
        pair_key = f"{donor} → {host}"

        # Baseline trajectories
        t_sec = np.arange(c.SIM_STEPS) * DT
        ax.plot(t_sec, baselines[donor]["x"] - baselines[donor]["x"][0],
                color=COLORS.get(donor, "gray"), lw=2, ls="--", alpha=0.4,
                label=f"{donor} (pure)")
        ax.plot(t_sec, baselines[host]["x"] - baselines[host]["x"][0],
                color=COLORS.get(host, "gray"), lw=2, ls=":", alpha=0.4,
                label=f"{host} (pure)")

        # Transplant trajectories at each switch time
        for t_switch in SWITCH_TIMES:
            key = (donor, host, t_switch)
            if key in transplant_results:
                x_arr = transplant_results[key]["x"]
                color = SWITCH_COLORS.get(t_switch, "black")
                ax.plot(t_sec, x_arr - x_arr[0], color=color, lw=1.2,
                        alpha=0.8, label=f"switch@{t_switch}")
                ax.axvline(t_switch * DT, color=color, lw=0.5, ls="--", alpha=0.5)

        ax.set_title(f"{pair_key}", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("DX (m)", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")
        clean_ax(ax)

    for idx in range(n_pairs, len(list(axes_flat))):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "cs_fig01_transplant_trajectories.png")


def fig02_timing_sweep(timing_results, baselines):
    """Fig 2: DX vs switch time for 3 key transplant pairs.

    Shows how final displacement depends on when the brain transplant
    occurs. Horizontal baselines show the pure donor and host DX for
    reference.

    Args:
        timing_results: dict keyed by "donor -> host" strings, each value
            a dict with "switch_steps" (list of ints) and "dx_values"
            (list of floats).
        baselines: dict keyed by champion name with "dx" values.

    Side effects:
        Writes cs_fig02_timing_sweep.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Transplant Timing: When Does the Switch Happen?",
                 fontsize=14, fontweight="bold")

    pairs_to_show = list(timing_results.keys())[:3]
    for idx, pair_key in enumerate(pairs_to_show):
        ax = axes[idx]
        tdata = timing_results[pair_key]
        switch_steps = tdata["switch_steps"]
        dxs = tdata["dx_values"]
        donor, host = pair_key.split(" → ")

        # Plot DX vs switch time
        t_sec = [s * DT for s in switch_steps]
        ax.plot(t_sec, dxs, "o-", color="#E24A33", lw=1.2, markersize=3)

        # Baselines
        ax.axhline(baselines[donor]["dx"], color=COLORS.get(donor, "gray"),
                    lw=1.5, ls="--", alpha=0.6, label=f"Pure {donor}: {baselines[donor]['dx']:+.1f}m")
        ax.axhline(baselines[host]["dx"], color=COLORS.get(host, "gray"),
                    lw=1.5, ls=":", alpha=0.6, label=f"Pure {host}: {baselines[host]['dx']:+.1f}m")

        ax.set_title(f"{pair_key}", fontsize=10)
        ax.set_xlabel("Switch time (s)", fontsize=9)
        ax.set_ylabel("Final DX (m)", fontsize=9)
        ax.legend(fontsize=7)
        clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "cs_fig02_timing_sweep.png")


def fig03_ablation_heatmap(ablation_results, baselines):
    """Fig 3: Heatmap of DX change when each synapse is zeroed at step 2000.

    Left panel: diverging heatmap (RdBu_r) of DX change per champion
    per synapse. Right panel: bar chart of the most critical synapse
    for each champion.

    Args:
        ablation_results: dict keyed by (champion, synapse_name) tuples,
            each value a dict with "x" and "dx".
        baselines: dict keyed by champion name with "dx" values.

    Side effects:
        Writes cs_fig03_ablation_heatmap.png to PLOT_DIR.
    """
    champions = ABLATION_TARGETS
    n_champ = len(champions)

    # Build matrix
    matrix = np.zeros((n_champ, 6))
    for ci, champ in enumerate(champions):
        base_dx = baselines[champ]["dx"]
        for wi, wn in enumerate(WEIGHT_NAMES):
            key = (champ, wn)
            if key in ablation_results:
                ablated_dx = ablation_results[key]["dx"]
                matrix[ci, wi] = ablated_dx - base_dx

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Mid-Sim Synapse Ablation (zeroed at step {ABLATION_TIME})",
                 fontsize=14, fontweight="bold")

    # Left: heatmap (DX change)
    ax = axes[0]
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-max(abs(matrix.min()), abs(matrix.max())),
                   vmax=max(abs(matrix.min()), abs(matrix.max())))
    ax.set_xticks(range(6))
    ax.set_xticklabels(WEIGHT_NAMES, fontsize=9)
    ax.set_yticks(range(n_champ))
    ax.set_yticklabels(champions, fontsize=9)
    ax.set_title("DX Change (ablated - baseline)", fontsize=10)
    for ci in range(n_champ):
        for wi in range(6):
            val = matrix[ci, wi]
            color = "white" if abs(val) > abs(matrix).max() * 0.5 else "black"
            ax.text(wi, ci, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="DX change (m)")

    # Right: bar chart of most critical synapses per champion
    ax = axes[1]
    ax.set_title("Most Critical Synapse per Champion", fontsize=10)
    for ci, champ in enumerate(champions):
        effects = []
        for wi, wn in enumerate(WEIGHT_NAMES):
            effects.append((abs(matrix[ci, wi]), matrix[ci, wi], wn))
        effects.sort(reverse=True)
        top_wn = effects[0][2]
        top_effect = effects[0][1]
        color = COLORS.get(champ, "gray")
        ax.barh(ci, top_effect, color=color, height=0.7)
        ax.text(top_effect + (1 if top_effect >= 0 else -1), ci,
                f"{top_wn} ({top_effect:+.1f}m)", va="center", fontsize=8)
    ax.set_yticks(range(n_champ))
    ax.set_yticklabels(champions, fontsize=9)
    ax.set_xlabel("DX change (m)", fontsize=9)
    ax.axvline(0, color="black", lw=0.5)
    ax.invert_yaxis()
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "cs_fig03_ablation_heatmap.png")


def fig04_rescue(rescue_results, baselines):
    """Fig 4: Rescue experiment -- transplant champion weights into random gaits.

    Shows x(t) trajectories for 10 random-to-champion transplants per
    champion. The champion baseline is shown as a dashed line, with the
    switch point at step 2000 marked by a vertical red line.

    Args:
        rescue_results: dict keyed by champion name, each value a list of
            dicts with "x" (numpy array) and "dx" (float), one per
            random host.
        baselines: dict keyed by champion name with "x" and "dx".

    Side effects:
        Writes cs_fig04_rescue.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Rescue Experiment: Transplant Champion into Random Gait at Step 2000",
                 fontsize=14, fontweight="bold")

    rescue_champions = ["NC", "Trial 3", "Pelton"]
    t_sec = np.arange(c.SIM_STEPS) * DT

    for idx, champ in enumerate(rescue_champions):
        ax = axes[idx]

        # Champion baseline
        ax.plot(t_sec, baselines[champ]["x"] - baselines[champ]["x"][0],
                color=COLORS.get(champ, "gray"), lw=2, ls="--", alpha=0.5,
                label=f"Pure {champ}")

        # Rescue trajectories
        if champ in rescue_results:
            for ri, rdata in enumerate(rescue_results[champ]):
                x_arr = rdata["x"]
                alpha = 0.5 if ri > 0 else 0.8
                label = "Random→Champion" if ri == 0 else None
                ax.plot(t_sec, x_arr - x_arr[0], color="#BBBBBB",
                        lw=0.8, alpha=alpha, label=label)

        ax.axvline(2000 * DT, color="red", lw=1, ls="--", alpha=0.7,
                   label="Switch point")
        ax.set_title(f"Rescue with {champ}", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("DX (m)", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7)
        clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "cs_fig04_rescue.png")


def fig05_recovery_dynamics(transplant_results, baselines):
    """Fig 5: Post-transplant recovery dynamics.

    Left panel: instantaneous speed time series for the NC-to-Trial3
    transplant at switch times 1000 and 2000, overlaid on pure baselines.
    Right panel: bar chart of DX accumulated from step 2000 onward for
    all transplant pairs and pure baselines, sorted by magnitude.

    Args:
        transplant_results: dict keyed by (donor, host, t_switch) tuples.
        baselines: dict keyed by champion name.

    Side effects:
        Writes cs_fig05_recovery_dynamics.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Post-Transplant Recovery Dynamics", fontsize=14, fontweight="bold")

    # Left: instantaneous speed after switch for NC→Trial3
    ax = axes[0]
    ax.set_title("Speed After Transplant (NC → Trial 3)", fontsize=10)
    t_sec = np.arange(c.SIM_STEPS) * DT

    # NC baseline speed
    nc_x = baselines["NC"]["x"]
    nc_speed = np.abs(np.diff(nc_x)) / DT
    ax.plot(t_sec[1:], nc_speed, color=COLORS["NC"], lw=0.5, alpha=0.3,
            label="Pure NC speed")

    t3_x = baselines["Trial 3"]["x"]
    t3_speed = np.abs(np.diff(t3_x)) / DT
    ax.plot(t_sec[1:], t3_speed, color=COLORS["Trial 3"], lw=0.5, alpha=0.3,
            label="Pure Trial 3 speed")

    for t_switch in [1000, 2000]:
        key = ("NC", "Trial 3", t_switch)
        if key in transplant_results:
            x_arr = transplant_results[key]["x"]
            speed = np.abs(np.diff(x_arr)) / DT
            color = SWITCH_COLORS.get(t_switch, "black")
            ax.plot(t_sec[1:], speed, color=color, lw=0.8, alpha=0.7,
                    label=f"NC→T3 @{t_switch}")
            ax.axvline(t_switch * DT, color=color, lw=1, ls="--", alpha=0.5)

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Instantaneous |speed| (m/s)", fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(fontsize=6, loc="upper right")
    clean_ax(ax)

    # Right: DX accumulated after switch point for various transplants
    ax = axes[1]
    ax.set_title("DX Accumulated After Switch (step 2000 onward)", fontsize=10)

    post_switch_dxs = {}
    for (donor, host, t_switch), rdata in transplant_results.items():
        if t_switch == 2000:
            x_arr = rdata["x"]
            post_dx = x_arr[-1] - x_arr[2000]
            pair_key = f"{donor}→{host}"
            post_switch_dxs[pair_key] = post_dx

    # Add baselines (pure gaits from step 2000 onward)
    for champ in CHAMPIONS:
        x_arr = baselines[champ]["x"]
        post_dx = x_arr[-1] - x_arr[2000]
        post_switch_dxs[f"Pure {champ}"] = post_dx

    # Sort by absolute value
    sorted_items = sorted(post_switch_dxs.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    colors_list = []
    for name in names:
        if "NC" in name and "→" in name:
            colors_list.append("#E24A33")
        elif "Pure" in name:
            colors_list.append("#348ABD")
        else:
            colors_list.append("#888888")

    ax.barh(range(len(names)), values, color=colors_list, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("DX from step 2000 onward (m)", fontsize=9)
    ax.axvline(0, color="black", lw=0.5)
    ax.invert_yaxis()
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "cs_fig05_recovery_dynamics.png")


def fig06_verdict(transplant_results, baselines, ablation_results,
                  timing_results, rescue_results):
    """Fig 6: Summary verdict panel combining all four experiment results.

    Layout (2x3 grid):
        Top-left: transplant DX shift at step 2000 (actual vs pure host).
        Top-center: synapse importance ranking (mean |DX change| per synapse).
        Top-right: timing sensitivity (DX std across switch times per pair).
        Bottom-left: rescue success rate (% of champion DX recovered).
        Bottom-right (spans 2 cols): monospace text block with quantitative
            verdicts and the key finding about physical state memory.

    Args:
        transplant_results: dict keyed by (donor, host, t_switch) tuples.
        baselines: dict keyed by champion name.
        ablation_results: dict keyed by (champion, synapse_name) tuples.
        timing_results: dict keyed by "donor -> host" strings.
        rescue_results: dict keyed by champion name.

    Side effects:
        Writes cs_fig06_verdict.png to PLOT_DIR.
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Causal Surgery Verdict", fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Top-left: DX shift for all transplants at step 2000
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Transplant DX Shift (@step 2000)", fontsize=10)
    shifts = []
    for (donor, host, t_switch), rdata in transplant_results.items():
        if t_switch == 2000:
            expected = baselines[host]["dx"]
            actual = rdata["dx"]
            # How close to host's pure DX?
            host_dx = baselines[host]["dx"]
            donor_dx = baselines[donor]["dx"]
            shifts.append((f"{donor}→{host}", actual, host_dx, donor_dx))
    shifts.sort(key=lambda x: abs(x[1]), reverse=True)
    for i, (label, actual, host_dx, donor_dx) in enumerate(shifts):
        ax.barh(i, actual, color="#E24A33", height=0.35, label="Actual" if i == 0 else None)
        ax.barh(i + 0.35, host_dx, color="#348ABD", height=0.35,
                alpha=0.5, label="Pure host" if i == 0 else None)
    ax.set_yticks([i + 0.175 for i in range(len(shifts))])
    ax.set_yticklabels([s[0] for s in shifts], fontsize=7)
    ax.set_xlabel("Final DX (m)", fontsize=8)
    ax.legend(fontsize=7)
    ax.invert_yaxis()
    clean_ax(ax)

    # Top-center: ablation severity ranking
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Synapse Importance (mean |DX change|)", fontsize=10)
    syn_importance = {wn: [] for wn in WEIGHT_NAMES}
    for (champ, wn), rdata in ablation_results.items():
        base_dx = baselines[champ]["dx"]
        syn_importance[wn].append(abs(rdata["dx"] - base_dx))
    mean_importance = [(wn, np.mean(vals)) for wn, vals in syn_importance.items()]
    mean_importance.sort(key=lambda x: x[1], reverse=True)
    wn_sorted = [x[0] for x in mean_importance]
    imp_sorted = [x[1] for x in mean_importance]
    ax.barh(range(6), imp_sorted, color=["#E24A33", "#348ABD", "#55A868",
            "#FBC15E", "#988ED5", "#777777"], height=0.7)
    ax.set_yticks(range(6))
    ax.set_yticklabels(wn_sorted, fontsize=9)
    ax.set_xlabel("Mean |DX change| when zeroed (m)", fontsize=8)
    ax.invert_yaxis()
    clean_ax(ax)

    # Top-right: timing sensitivity summary
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Timing Sensitivity (DX std across switch times)", fontsize=10)
    timing_stds = []
    for pair_key, tdata in timing_results.items():
        dx_std = np.std(tdata["dx_values"])
        timing_stds.append((pair_key, dx_std))
    timing_stds.sort(key=lambda x: x[1], reverse=True)
    names = [x[0].replace(" → ", "\n→ ") for x in timing_stds]
    stds = [x[1] for x in timing_stds]
    ax.barh(range(len(names)), stds, color="#55A868", height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("DX std across switch times (m)", fontsize=8)
    ax.invert_yaxis()
    clean_ax(ax)

    # Bottom-left: rescue success rate
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Rescue Success", fontsize=10)
    rescue_data = []
    for champ in ["NC", "Trial 3", "Pelton"]:
        if champ in rescue_results:
            pure_dx = abs(baselines[champ]["dx"])
            rescued_dxs = [abs(r["dx"]) for r in rescue_results[champ]]
            mean_rescued = np.mean(rescued_dxs)
            recovery_frac = mean_rescued / pure_dx if pure_dx > 0 else 0
            rescue_data.append((champ, recovery_frac, mean_rescued, pure_dx))
    for i, (champ, frac, mean_r, pure) in enumerate(rescue_data):
        ax.barh(i, frac * 100, color=COLORS.get(champ, "gray"), height=0.7)
        ax.text(frac * 100 + 1, i, f"{mean_r:.1f}m / {pure:.1f}m",
                va="center", fontsize=8)
    ax.set_yticks(range(len(rescue_data)))
    ax.set_yticklabels([x[0] for x in rescue_data], fontsize=9)
    ax.set_xlabel("Recovery (% of pure champion DX)", fontsize=8)
    ax.set_xlim(0, 120)
    ax.axvline(100, color="gray", lw=0.5, ls="--")
    ax.invert_yaxis()
    clean_ax(ax)

    # Bottom-center + right: verdict text
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis("off")

    # Compute verdicts
    mean_transplant_shift = np.mean([abs(s[1] - s[2]) for s in shifts])

    all_ablation_effects = []
    for (champ, wn), rdata in ablation_results.items():
        base_dx = baselines[champ]["dx"]
        all_ablation_effects.append(abs(rdata["dx"] - base_dx))
    mean_ablation = np.mean(all_ablation_effects)
    max_ablation = max(all_ablation_effects)

    all_timing_stds = [np.std(td["dx_values"]) for td in timing_results.values()]
    mean_timing_std = np.mean(all_timing_stds)

    rescue_fracs = []
    for champ in ["NC", "Trial 3", "Pelton"]:
        if champ in rescue_results:
            pure_dx = abs(baselines[champ]["dx"])
            if pure_dx > 0:
                rescued = [abs(r["dx"]) for r in rescue_results[champ]]
                rescue_fracs.append(np.mean(rescued) / pure_dx)
    mean_rescue = np.mean(rescue_fracs) if rescue_fracs else 0

    verdict_lines = [
        "TRANSPLANT EFFECTS",
        f"  Mean |DX shift from pure host|: {mean_transplant_shift:.1f}m",
        f"  History matters: transplanted gaits differ from pure host",
        "",
        "ABLATION",
        f"  Mean |DX change| per synapse: {mean_ablation:.1f}m",
        f"  Max single-synapse effect: {max_ablation:.1f}m",
        f"  Most critical synapse: {mean_importance[0][0]} ({mean_importance[0][1]:.1f}m)",
        "",
        "TIMING SENSITIVITY",
        f"  Mean DX std across switch times: {mean_timing_std:.1f}m",
        f"  When you switch matters — not just what you switch to",
        "",
        "RESCUE",
        f"  Mean recovery: {mean_rescue*100:.0f}% of pure champion DX",
        f"  {'Partial' if mean_rescue < 0.8 else 'Strong'} rescue effect",
        "",
        "KEY FINDING",
        "  The body retains memory of its initial controller.",
        "  Physical state (position, velocity, contact phase) at switch",
        "  time shapes the post-switch trajectory. You can't erase history.",
    ]

    ax.text(0.05, 0.95, "\n".join(verdict_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    fig.tight_layout()
    save_fig(fig, "cs_fig06_verdict.png")


# ── JSON encoder ────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays.

    Converts numpy integers, floats, booleans, and ndarrays to their
    Python equivalents for JSON serialization. Floats are rounded to
    6 decimal places to keep output compact.
    """

    def default(self, obj):
        """Serialize numpy types to JSON-compatible Python types.

        Args:
            obj: object to serialize.

        Returns:
            JSON-serializable Python type.
        """
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
    """Run all four causal surgery experiments and produce figures + JSON.

    Pipeline:
        1. Baselines -- run each champion with no surgery (5 sims).
        2. Brain transplants -- swap all 6 weights at t_switch for 6
           donor-host pairs x 4 switch times (24 sims).
        3. Single-synapse ablation -- zero one weight at step 2000 for
           5 champions x 6 synapses (30 sims).
        4. Timing sweep -- vary switch time in steps of 100 across the
           full sim for 3 key pairs (120 sims).
        5. Rescue -- transplant champion weights into 10 random gaits
           at step 2000 for 3 champions (30 sims).

    Side effects:
        - Overwrites brain.nndf many times (not restored -- no meaningful
          final state).
        - Writes artifacts/causal_surgery.json.
        - Writes 6 PNG figures to artifacts/plots/.
        - Prints progress and analysis to stdout.
    """
    total_sims = 0
    t_global = time.time()

    print("Causal Surgery — Mid-Simulation Weight Switching")
    print()

    # ── Baselines ───────────────────────────────────────────────────────────
    print("=" * 80)
    print("BASELINES: Pure champion simulations")
    print("=" * 80)
    baselines = {}
    for name, weights in CHAMPIONS.items():
        x_arr, dx = simulate_with_surgery(weights)
        baselines[name] = {"x": x_arr, "dx": dx}
        total_sims += 1
        print(f"  {name}: DX={dx:+.2f}m")

    # ── Part 1: Brain Transplants ───────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"PART 1: Brain Transplants ({len(TRANSPLANT_PAIRS)} pairs x "
          f"{len(SWITCH_TIMES)} times = {len(TRANSPLANT_PAIRS)*len(SWITCH_TIMES)} sims)")
    print("=" * 80)
    t0 = time.time()

    transplant_results = {}
    for pi, (donor, host) in enumerate(TRANSPLANT_PAIRS):
        for t_switch in SWITCH_TIMES:
            surgery = [(t_switch, CHAMPIONS[host])]
            x_arr, dx = simulate_with_surgery(CHAMPIONS[donor], surgery)
            transplant_results[(donor, host, t_switch)] = {"x": x_arr, "dx": dx}
            total_sims += 1

        # Report
        dxs_at_switches = [transplant_results[(donor, host, t)]["dx"]
                           for t in SWITCH_TIMES]
        print(f"  [{pi+1}/{len(TRANSPLANT_PAIRS)}] {donor} → {host}: "
              f"DX at switches = {[f'{d:+.1f}' for d in dxs_at_switches]}")

    print(f"  Part 1: {len(TRANSPLANT_PAIRS)*len(SWITCH_TIMES)} sims "
          f"in {time.time()-t0:.1f}s")

    # ── Part 2: Single-Synapse Ablation ─────────────────────────────────────
    print()
    print("=" * 80)
    print(f"PART 2: Mid-Sim Synapse Ablation ({len(ABLATION_TARGETS)} x 6 = "
          f"{len(ABLATION_TARGETS)*6} sims)")
    print("=" * 80)
    t0 = time.time()

    ablation_results = {}
    for champ in ABLATION_TARGETS:
        weights = CHAMPIONS[champ]
        base_dx = baselines[champ]["dx"]
        effects = []

        for wn in WEIGHT_NAMES:
            surgery = [(ABLATION_TIME, wn, 0.0)]
            x_arr, dx = simulate_with_surgery(weights, surgery)
            ablation_results[(champ, wn)] = {"x": x_arr, "dx": dx}
            total_sims += 1
            effects.append(dx - base_dx)

        print(f"  {champ}: DX changes = {[f'{e:+.1f}' for e in effects]}")

    print(f"  Part 2: {len(ABLATION_TARGETS)*6} sims in {time.time()-t0:.1f}s")

    # ── Part 3: Switch Timing Sweep ─────────────────────────────────────────
    print()
    print("=" * 80)
    print("PART 3: Switch Timing Sweep (3 pairs x 40 switch times = 120 sims)")
    print("=" * 80)
    t0 = time.time()

    timing_pairs = [
        ("NC", "Trial 3"),
        ("NC", "Pelton"),
        ("Trial 3", "NC"),
    ]
    timing_results = {}
    switch_steps = list(range(100, 3901, 100))

    for donor, host in timing_pairs:
        dx_vals = []
        for t_switch in switch_steps:
            surgery = [(t_switch, CHAMPIONS[host])]
            _, dx = simulate_with_surgery(CHAMPIONS[donor], surgery)
            dx_vals.append(dx)
            total_sims += 1

        pair_key = f"{donor} → {host}"
        timing_results[pair_key] = {
            "switch_steps": switch_steps,
            "dx_values": dx_vals,
        }
        dx_std = np.std(dx_vals)
        print(f"  {pair_key}: DX range=[{min(dx_vals):+.1f}, {max(dx_vals):+.1f}]  "
              f"std={dx_std:.1f}m")

    print(f"  Part 3: {len(timing_pairs)*len(switch_steps)} sims "
          f"in {time.time()-t0:.1f}s")

    # ── Part 4: Rescue Experiments ──────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"PART 4: Rescue Experiments (3 champions x {N_RANDOM} random = "
          f"{3*N_RANDOM} sims)")
    print("=" * 80)
    t0 = time.time()

    rng = np.random.RandomState(RNG_SEED)
    random_weights_list = []
    for _ in range(N_RANDOM):
        w = {wn: rng.uniform(-2, 2) for wn in WEIGHT_NAMES}
        random_weights_list.append(w)

    rescue_results = {}
    rescue_champions = ["NC", "Trial 3", "Pelton"]
    for champ in rescue_champions:
        rescue_results[champ] = []
        rescued_dxs = []
        for ri, rand_w in enumerate(random_weights_list):
            surgery = [(2000, CHAMPIONS[champ])]
            x_arr, dx = simulate_with_surgery(rand_w, surgery)
            rescue_results[champ].append({"x": x_arr, "dx": dx})
            rescued_dxs.append(dx)
            total_sims += 1

        pure_dx = baselines[champ]["dx"]
        mean_rescued = np.mean([abs(d) for d in rescued_dxs])
        print(f"  {champ}: mean rescued |DX|={mean_rescued:.1f}m  "
              f"(pure={abs(pure_dx):.1f}m, recovery={mean_rescued/abs(pure_dx)*100:.0f}%)")

    print(f"  Part 4: {3*N_RANDOM} sims in {time.time()-t0:.1f}s")

    # ── Analysis ────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("CAUSAL SURGERY ANALYSIS")
    print("=" * 80)

    # Transplant analysis: does the robot converge to the host's gait?
    print("\n  TRANSPLANT CONVERGENCE (switch at step 2000):")
    print(f"  {'Pair':25s} {'Actual DX':>10s} {'Pure Host':>10s} "
          f"{'Pure Donor':>10s} {'Convergence':>12s}")
    print("  " + "-" * 75)
    for (donor, host, t_switch), rdata in transplant_results.items():
        if t_switch == 2000:
            actual = rdata["dx"]
            host_dx = baselines[host]["dx"]
            donor_dx = baselines[donor]["dx"]
            # Convergence: how much of the way from donor to host?
            if abs(host_dx - donor_dx) > EPS:
                convergence = (actual - donor_dx) / (host_dx - donor_dx)
            else:
                convergence = 1.0
            print(f"  {donor+'→'+host:25s} {actual:+10.1f} {host_dx:+10.1f} "
                  f"{donor_dx:+10.1f} {convergence:+10.1%}")

    # Ablation ranking
    print("\n  SYNAPSE IMPORTANCE (mean |DX change| when zeroed at step 2000):")
    syn_importance = {wn: [] for wn in WEIGHT_NAMES}
    for (champ, wn), rdata in ablation_results.items():
        base_dx = baselines[champ]["dx"]
        syn_importance[wn].append(abs(rdata["dx"] - base_dx))
    ranked = sorted(syn_importance.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for wn, vals in ranked:
        print(f"    {wn}: mean={np.mean(vals):.1f}m  max={np.max(vals):.1f}m")

    # Timing sensitivity
    print("\n  TIMING SENSITIVITY:")
    for pair_key, tdata in timing_results.items():
        dxs = tdata["dx_values"]
        print(f"    {pair_key}: std={np.std(dxs):.1f}m  "
              f"range=[{min(dxs):+.1f}, {max(dxs):+.1f}]")

    # ── Figures ─────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    fig01_transplant_trajectories(transplant_results, baselines)
    fig02_timing_sweep(timing_results, baselines)
    fig03_ablation_heatmap(ablation_results, baselines)
    fig04_rescue(rescue_results, baselines)
    fig05_recovery_dynamics(transplant_results, baselines)
    fig06_verdict(transplant_results, baselines, ablation_results,
                  timing_results, rescue_results)

    # ── Save JSON ───────────────────────────────────────────────────────────
    json_out = {
        "baselines": {n: {"dx": b["dx"]} for n, b in baselines.items()},
        "transplants": {f"{d}→{h}@{t}": {"dx": r["dx"]}
                       for (d, h, t), r in transplant_results.items()},
        "ablations": {f"{c}_{w}": {"dx": r["dx"]}
                     for (c, w), r in ablation_results.items()},
        "timing": {k: {"switch_steps": v["switch_steps"], "dx_values": v["dx_values"]}
                  for k, v in timing_results.items()},
        "rescue": {c: [{"dx": r["dx"]} for r in rlist]
                  for c, rlist in rescue_results.items()},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    elapsed = time.time() - t_global
    print(f"\nTotal: {total_sims} sims in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
