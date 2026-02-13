#!/usr/bin/env python3
"""
timestep_atlas.py

Role:
    Timestep Atlas -- Adding DT as a 7th Gaitspace Dimension.

    Sweeps all 116 zoo gaits across 7 DT values in two modes to separate
    physics-resolution artifacts from controller-sampling-rate artifacts.

Modes:
    Coupled mode:
        NN updates every physics step (current behavior). At finer DTs the NN
        runs more frequently, so both physics resolution and control rate change
        together.
    Decoupled mode:
        NN updates at fixed control_dt=1/240 (4000 times total). Physics
        substeps at finer DT between NN updates, with motor commands held
        constant during substeps. This isolates the physics-resolution effect.

Simulation budget:
    116 gaits x (7 coupled + 3 decoupled) = 1,160 sims (~25-40 min)

Metrics per (gait, dt, mode):
    retention_abs  -- abs(DX(dt)) / max(abs(DX_baseline), 0.5)
    sign_flip      -- sign(DX(dt)) != sign(DX_baseline)
    collapse_dt    -- first DT finer than baseline where retention_abs < 0.5
    best_dt        -- DT producing max abs(DX)
    monotonic      -- |DX| decreases as DT shrinks, within 5% tolerance
    robustness     -- mean of min(retention_abs, 1.0) across all DTs (capped)

Notes:
    Comparing coupled vs decoupled robustness reveals the source of timestep
    sensitivity: if decoupled holds but coupled fails, the gait is sensitive to
    controller sampling rate; if both fail, the solver resolution is the issue.

    Checkpoint/resume support allows interruption and restart of the long sweep
    via the --resume flag and a JSONL progress file.

Outputs:
    artifacts/timestep_atlas.json              -- full results + summary
    artifacts/timestep_atlas_progress.jsonl    -- checkpoint file
    artifacts/plots/ts_fig01_heatmap_dx.png
    artifacts/plots/ts_fig02_heatmap_retention.png
    artifacts/plots/ts_fig03_robustness.png
    artifacts/plots/ts_fig04_champion_curves.png
    artifacts/plots/ts_fig05_collapse.png

Usage:
    python3 timestep_atlas.py              # full run (~30 min)
    python3 timestep_atlas.py --resume     # restart after interruption
"""

import argparse
import json
import math
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
from matplotlib.colors import TwoSlopeNorm

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from compute_beer_analytics import NumpyEncoder

# ── Constants ────────────────────────────────────────────────────────────────

CONTROL_DT = 1 / 240           # fixed NN update rate for decoupled mode
CONTROL_STEPS = c.SIM_STEPS    # 4000 NN updates (same wall-clock as baseline)

# DT sweep values: coarser-than-baseline through 4x finer
DT_VALUES = [1/120, 1/180, 1/240, 1/360, 1/480, 1/720, 1/960]
DT_LABELS = ["1/120", "1/180", "1/240", "1/360", "1/480", "1/720", "1/960"]
DT_HZ     = [120, 180, 240, 360, 480, 720, 960]

# Decoupled mode only for integer multiples of control_dt
# k = round(control_dt / dt) must be integer >= 2
DECOUPLED_DTS = [1/480, 1/720, 1/960]   # k=2, k=3, k=4

BASELINE_DT = 1 / 240
BASELINE_IDX = 2  # index of 1/240 in DT_VALUES

# Architecture color mapping for plots
ARCH_COLORS = {
    "standard_6": "#4C72B0",
    "crosswired_10": "#C44E52",
    "crosswired_8": "#DD8452",
    "hidden": "#55A868",
}

ZOO_PATH = PROJECT / "synapse_gait_zoo.json"
OUT_JSON = PROJECT / "artifacts" / "timestep_atlas.json"
PROGRESS_PATH = PROJECT / "artifacts" / "timestep_atlas_progress.jsonl"
PLOT_DIR = PROJECT / "artifacts" / "plots"


# ── Brain Writers (from generate_telemetry.py) ──────────────────────────────

def write_brain_crosswired(w03, w13, w23, w04, w14, w24,
                           w34=0.0, w43=0.0, w33=0.0, w44=0.0):
    """Write brain.nndf for a standard or crosswired topology.

    Creates an NNDF file with 3 sensor neurons (Torso, BackLeg, FrontLeg)
    and 2 motor neurons (Torso_BackLeg, Torso_FrontLeg). The 6 sensor-to-motor
    weights are always included; the 4 motor-to-motor weights (crosswired) are
    only written when non-zero.

    Args:
        w03: Sensor 0 (Torso) -> Motor 3 (BackLeg) weight.
        w13: Sensor 1 (BackLeg) -> Motor 3 (BackLeg) weight.
        w23: Sensor 2 (FrontLeg) -> Motor 3 (BackLeg) weight.
        w04: Sensor 0 (Torso) -> Motor 4 (FrontLeg) weight.
        w14: Sensor 1 (BackLeg) -> Motor 4 (FrontLeg) weight.
        w24: Sensor 2 (FrontLeg) -> Motor 4 (FrontLeg) weight.
        w34: Motor 3 -> Motor 4 cross-connection weight.
        w43: Motor 4 -> Motor 3 cross-connection weight.
        w33: Motor 3 self-feedback weight.
        w44: Motor 4 self-feedback weight.

    Side effects:
        Overwrites PROJECT/brain.nndf.
    """
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for src, tgt, w in [("0","3",w03), ("1","3",w13), ("2","3",w23),
                             ("0","4",w04), ("1","4",w14), ("2","4",w24),
                             ("3","4",w34), ("4","3",w43), ("3","3",w33), ("4","4",w44)]:
            if w != 0.0:
                f.write(f'    <synapse sourceNeuronName = "{src}" targetNeuronName = "{tgt}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def write_brain_full(neurons, synapses):
    """Write brain.nndf for an arbitrary topology (including hidden neurons).

    Supports any combination of sensor, motor, and hidden neurons with
    arbitrary synapse connectivity, as used by the 'hidden' architecture
    gaits in the zoo.

    Args:
        neurons: List of dicts, each with keys 'id', 'type' ('sensor'/'motor'/'hidden'),
            and 'ref' (linkName for sensors, jointName for motors; absent for hidden).
        synapses: List of dicts, each with keys 'src' (source neuron id),
            'tgt' (target neuron id), and 'w' (weight). Zero-weight synapses
            are skipped.

    Side effects:
        Overwrites PROJECT/brain.nndf.
    """
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        for neuron in neurons:
            nid, ntype, ref = neuron["id"], neuron["type"], neuron.get("ref")
            if ntype == "sensor":
                f.write(f'    <neuron name = "{nid}" type = "sensor" linkName = "{ref}" />\n')
            elif ntype == "motor":
                f.write(f'    <neuron name = "{nid}" type = "motor"  jointName = "{ref}" />\n')
            else:
                f.write(f'    <neuron name = "{nid}" type = "hidden" />\n')
        for syn in synapses:
            w = syn["w"]
            if w != 0.0:
                f.write(f'    <synapse sourceNeuronName = "{syn["src"]}" targetNeuronName = "{syn["tgt"]}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def write_brain_for_gait(gait_data):
    """Dispatch to the correct brain writer based on the gait's architecture field.

    Args:
        gait_data: Dict from synapse_gait_zoo.json containing at minimum an
            'architecture' key. For 'hidden' architecture, must include 'neurons'
            and 'synapses' lists. For all others (standard_6, crosswired_10,
            crosswired_8), must include a 'weights' dict with wXY keys.

    Side effects:
        Overwrites PROJECT/brain.nndf via the appropriate writer function.
    """
    arch = gait_data.get("architecture", "standard_6")
    if arch == "hidden":
        write_brain_full(gait_data["neurons"], gait_data["synapses"])
    else:
        w = gait_data["weights"]
        write_brain_crosswired(
            w.get("w03", 0.0), w.get("w13", 0.0), w.get("w23", 0.0),
            w.get("w04", 0.0), w.get("w14", 0.0), w.get("w24", 0.0),
            w.get("w34", 0.0), w.get("w43", 0.0), w.get("w33", 0.0), w.get("w44", 0.0),
        )


# ── Simulation Harness ──────────────────────────────────────────────────────

def simulate_coupled(gait_data, dt):
    """Run a simulation in coupled mode where the NN updates every physics step.

    The total simulated wall-clock time is kept constant at
    CONTROL_STEPS * CONTROL_DT (~16.67s), so finer DTs produce more physics
    steps and correspondingly more NN updates. At DT=1/480 the NN updates
    8000 times (2x baseline).

    Args:
        gait_data: Gait definition dict from the zoo (weights, architecture, etc.).
        dt: Physics timestep in seconds (e.g., 1/480).

    Returns:
        dx: Float displacement along the x-axis in meters.

    Side effects:
        Writes brain.nndf to disk.
        Creates and destroys a PyBullet DIRECT-mode physics session.
    """
    write_brain_for_gait(gait_data)

    sim_duration = CONTROL_STEPS * CONTROL_DT  # same wall-clock as baseline
    n_steps = round(sim_duration / dt)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(dt)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
    for link in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, link, lateralFriction=mu, restitution=0.0)

    nn = NEURAL_NETWORK(str(PROJECT / "brain.nndf"))
    max_force = float(getattr(c, "MAX_FORCE", 150.0))

    start_x = p.getBasePositionAndOrientation(robotId)[0][0]

    for i in range(n_steps):
        # Act
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

    end_x = p.getBasePositionAndOrientation(robotId)[0][0]
    p.disconnect()
    return end_x - start_x


def simulate_decoupled(gait_data, dt):
    """Run a simulation in decoupled mode to isolate physics resolution effects.

    The NN updates at a fixed rate of control_dt=1/240 (CONTROL_STEPS=4000
    total updates), regardless of the physics timestep. Between each NN update,
    k = round(control_dt / dt) physics substeps are taken with motor commands
    held constant. This separates physics-resolution sensitivity from
    controller-sampling-rate sensitivity.

    Args:
        gait_data: Gait definition dict from the zoo (weights, architecture, etc.).
        dt: Physics timestep in seconds. Must satisfy k = round(CONTROL_DT / dt) >= 2.

    Returns:
        dx: Float displacement along the x-axis in meters.

    Side effects:
        Writes brain.nndf to disk.
        Creates and destroys a PyBullet DIRECT-mode physics session.
    """
    write_brain_for_gait(gait_data)

    k = round(CONTROL_DT / dt)
    assert k >= 2, f"Decoupled mode requires k >= 2, got k={k} for dt={dt}"

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(dt)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
    for link in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, link, lateralFriction=mu, restitution=0.0)

    nn = NEURAL_NETWORK(str(PROJECT / "brain.nndf"))
    max_force = float(getattr(c, "MAX_FORCE", 150.0))

    start_x = p.getBasePositionAndOrientation(robotId)[0][0]

    for control_step in range(CONTROL_STEPS):
        # Apply motor commands from NN
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

        # Physics substeps (motor commands held constant)
        for _ in range(k):
            p.stepSimulation()

        # NN update (sensor reads + recompute)
        nn.Update()

    end_x = p.getBasePositionAndOrientation(robotId)[0][0]
    p.disconnect()
    return end_x - start_x


# ── Checkpointing ───────────────────────────────────────────────────────────

def load_progress():
    """Load completed simulation entries from the JSONL checkpoint file.

    Each line in the checkpoint is a JSON object keyed by (gait, dt_hz, mode).
    Used by --resume to skip already-completed simulations.

    Returns:
        Dict mapping (gait_name, dt_hz, mode) tuples to their result dicts.
        Empty dict if no checkpoint file exists.
    """
    done = {}
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = (entry["gait"], entry["dt_hz"], entry["mode"])
                done[key] = entry
    return done


def append_progress(entry):
    """Append one completed simulation result to the JSONL checkpoint file.

    Args:
        entry: Result dict containing 'gait', 'dt_hz', 'mode', 'dx', etc.

    Side effects:
        Creates the artifacts directory if it does not exist.
        Appends one JSON line to PROGRESS_PATH.
    """
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "a") as f:
        f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")


# ── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(gaits, resume=False):
    """Run the full DT sweep across all gaits in both coupled and decoupled modes.

    Iterates through every gait in the zoo, running coupled-mode simulations
    at all 7 DT values and decoupled-mode simulations at the 3 applicable
    DT values. Progress is checkpointed after each simulation so the sweep
    can be resumed after interruption.

    Args:
        gaits: List of (gait_name, gait_data, cat_name) tuples from the zoo.
        resume: If True, load previously completed results from the checkpoint
            file and skip those simulations.

    Returns:
        List of result dicts, one per (gait, dt, mode) combination. Each dict
        contains 'gait', 'category', 'architecture', 'dt', 'dt_hz', 'dt_label',
        'mode', 'dx', and 'elapsed_s'.

    Side effects:
        Writes brain.nndf repeatedly (once per simulation).
        Appends to the JSONL checkpoint file.
        Prints progress to stdout every 50 simulations.
    """
    done = load_progress() if resume else {}
    if resume and done:
        print(f"  Resuming: {len(done)} results already completed")

    results = list(done.values())
    total_sims = len(gaits) * (len(DT_VALUES) + len(DECOUPLED_DTS))
    completed = len(done)

    t_start = time.perf_counter()

    for gait_idx, (gait_name, gait_data, cat_name) in enumerate(gaits):
        arch = gait_data.get("architecture", "standard_6")

        # Coupled mode: all 7 DTs
        for dt_idx, (dt, dt_hz, dt_label) in enumerate(zip(DT_VALUES, DT_HZ, DT_LABELS)):
            key = (gait_name, dt_hz, "coupled")
            if key in done:
                continue

            t0 = time.perf_counter()
            dx = simulate_coupled(gait_data, dt)
            elapsed = time.perf_counter() - t0
            completed += 1

            entry = {
                "gait": gait_name,
                "category": cat_name,
                "architecture": arch,
                "dt": float(dt),
                "dt_hz": dt_hz,
                "dt_label": dt_label,
                "mode": "coupled",
                "dx": float(dx),
                "elapsed_s": round(elapsed, 3),
            }
            results.append(entry)
            append_progress(entry)

            if completed % 50 == 0 or completed == total_sims:
                total_elapsed = time.perf_counter() - t_start
                rate = total_elapsed / completed if completed > 0 else 0
                remaining = rate * (total_sims - completed)
                print(f"  [{completed:4d}/{total_sims}] {gait_name:30s} dt={dt_label:6s} "
                      f"mode=coupled   DX={dx:+7.2f}  "
                      f"({total_elapsed:.0f}s elapsed, ~{remaining:.0f}s left)",
                      flush=True)

        # Decoupled mode: only DTs that are integer multiples of control_dt
        for dt in DECOUPLED_DTS:
            dt_hz = round(1 / dt)
            dt_label = f"1/{dt_hz}"
            key = (gait_name, dt_hz, "decoupled")
            if key in done:
                continue

            t0 = time.perf_counter()
            dx = simulate_decoupled(gait_data, dt)
            elapsed = time.perf_counter() - t0
            completed += 1

            entry = {
                "gait": gait_name,
                "category": cat_name,
                "architecture": arch,
                "dt": float(dt),
                "dt_hz": dt_hz,
                "dt_label": dt_label,
                "mode": "decoupled",
                "dx": float(dx),
                "elapsed_s": round(elapsed, 3),
            }
            results.append(entry)
            append_progress(entry)

            if completed % 50 == 0 or completed == total_sims:
                total_elapsed = time.perf_counter() - t_start
                rate = total_elapsed / completed if completed > 0 else 0
                remaining = rate * (total_sims - completed)
                print(f"  [{completed:4d}/{total_sims}] {gait_name:30s} dt={dt_label:6s} "
                      f"mode=decoupled DX={dx:+7.2f}  "
                      f"({total_elapsed:.0f}s elapsed, ~{remaining:.0f}s left)",
                      flush=True)

    total_elapsed = time.perf_counter() - t_start
    print(f"\n  Sweep complete: {completed} sims in {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)")
    return results


# ── Metric Computation ───────────────────────────────────────────────────────

def compute_metrics(gaits, results):
    """Compute per-gait timestep-sensitivity metrics from sweep results.

    Analyzes each gait's displacement across all tested DTs relative to the
    baseline (1/240 coupled) to quantify robustness, detect collapses, and
    compare coupled vs decoupled behavior.

    Args:
        gaits: List of (gait_name, gait_data, cat_name) tuples from the zoo.
        results: List of result dicts from run_sweep(), each containing
            'gait', 'dt_hz', 'mode', 'dx', etc.

    Returns:
        Dict keyed by gait_name, where each value is a metrics dict containing:
            baseline_dx         -- DX at 1/240 coupled (the reference).
            architecture        -- Network topology label.
            category            -- Zoo category name.
            coupled_dxs         -- {dt_hz_str: dx} for all 7 coupled DTs.
            decoupled_dxs       -- {dt_hz_str: dx} for the 3 decoupled DTs.
            retention_abs       -- {dt_hz_str: float} ratio of |DX| to baseline.
            decoupled_retention -- {dt_hz_str: float} retention for decoupled mode.
            sign_flips          -- {dt_hz_str: bool} whether displacement reversed.
            collapse_dt         -- First DT (Hz) finer than baseline with retention < 0.5,
                                   or None if no collapse.
            best_dt             -- DT (Hz) producing max |DX| in coupled mode.
            monotonic           -- True if |DX| decreases as DT shrinks (5% tolerance).
            robustness          -- Mean of min(retention, 1.0) across coupled DTs.
            decoupled_robustness -- Same metric for decoupled DTs only.
            is_zero_baseline    -- True if |baseline_dx| < 0.5m.
    """
    # Index results by (gait, dt_hz, mode)
    by_key = {}
    for r in results:
        by_key[(r["gait"], r["dt_hz"], r["mode"])] = r

    metrics = {}
    for gait_name, gait_data, cat_name in gaits:
        arch = gait_data.get("architecture", "standard_6")

        # Baseline DX at 1/240 coupled
        baseline_entry = by_key.get((gait_name, 240, "coupled"))
        if baseline_entry is None:
            continue
        baseline_dx = baseline_entry["dx"]

        # Coupled DX values
        coupled_dxs = {}
        for dt_hz in DT_HZ:
            entry = by_key.get((gait_name, dt_hz, "coupled"))
            if entry:
                coupled_dxs[dt_hz] = entry["dx"]

        # Decoupled DX values
        decoupled_dxs = {}
        for dt in DECOUPLED_DTS:
            dt_hz = round(1 / dt)
            entry = by_key.get((gait_name, dt_hz, "decoupled"))
            if entry:
                decoupled_dxs[dt_hz] = entry["dx"]

        # Retention and sign flips (coupled mode)
        # Floor the denominator at 0.5 to avoid division-by-zero for near-stationary gaits
        denom = max(abs(baseline_dx), 0.5)
        is_zero_baseline = abs(baseline_dx) < 0.5

        retention_abs = {}
        sign_flips = {}
        for dt_hz, dx in coupled_dxs.items():
            if is_zero_baseline:
                # Zero-motion baseline: retention = 1.0 if also near-zero
                if abs(dx) < 0.5:
                    retention_abs[dt_hz] = 1.0
                else:
                    retention_abs[dt_hz] = abs(dx) / 0.5  # "unstable_zero" — flag as high
                sign_flips[dt_hz] = False
            else:
                retention_abs[dt_hz] = abs(dx) / denom
                sign_flips[dt_hz] = (np.sign(dx) != np.sign(baseline_dx))

        # Collapse DT: first DT finer than baseline where retention < 0.5
        collapse_dt = None
        for dt_hz in DT_HZ:
            if dt_hz <= 240:
                continue  # only look at finer-than-baseline
            ret = retention_abs.get(dt_hz)
            if ret is not None and ret < 0.5:
                collapse_dt = dt_hz
                break

        # Best DT: DT producing max |DX|
        best_dt = max(coupled_dxs, key=lambda hz: abs(coupled_dxs[hz]))

        # Monotonic: |DX| decreases as DT shrinks (finer than baseline)
        # Allow 5% tolerance: a small bump doesn't fail
        finer_dts = [hz for hz in sorted(DT_HZ) if hz > 240]
        monotonic = True
        prev_abs_dx = abs(baseline_dx)
        for dt_hz in finer_dts:
            dx = coupled_dxs.get(dt_hz)
            if dx is None:
                continue
            curr_abs_dx = abs(dx)
            # Allow 5% bump above previous
            if curr_abs_dx > prev_abs_dx * 1.05:
                monotonic = False
                break
            # Track running min so a recovery after a dip doesn't mask a later drop
            prev_abs_dx = min(prev_abs_dx, curr_abs_dx)

        # Robustness: mean of min(retention, 1.0) across all coupled DTs
        ret_values = [min(retention_abs.get(hz, 1.0), 1.0) for hz in DT_HZ]
        robustness = float(np.mean(ret_values))

        # Decoupled retention
        decoupled_retention = {}
        for dt_hz, dx in decoupled_dxs.items():
            if is_zero_baseline:
                decoupled_retention[dt_hz] = 1.0 if abs(dx) < 0.5 else abs(dx) / 0.5
            else:
                decoupled_retention[dt_hz] = abs(dx) / denom

        # Decoupled robustness (only 3 DTs)
        dec_ret_values = [min(decoupled_retention.get(hz, 1.0), 1.0)
                          for hz in [480, 720, 960]]
        decoupled_robustness = float(np.mean(dec_ret_values))

        metrics[gait_name] = {
            "gait": gait_name,
            "category": cat_name,
            "architecture": arch,
            "baseline_dx": float(baseline_dx),
            "coupled_dxs": {str(k): float(v) for k, v in coupled_dxs.items()},
            "decoupled_dxs": {str(k): float(v) for k, v in decoupled_dxs.items()},
            "retention_abs": {str(k): round(float(v), 4) for k, v in retention_abs.items()},
            "decoupled_retention": {str(k): round(float(v), 4) for k, v in decoupled_retention.items()},
            "sign_flips": {str(k): bool(v) for k, v in sign_flips.items()},
            "collapse_dt": collapse_dt,
            "best_dt": best_dt,
            "monotonic": bool(monotonic),
            "robustness": round(robustness, 4),
            "decoupled_robustness": round(decoupled_robustness, 4),
            "is_zero_baseline": bool(is_zero_baseline),
        }

    return metrics


# ── Plotting Helpers ─────────────────────────────────────────────────────────

def clean_ax(ax):
    """Remove top and right spines from a matplotlib axes for a cleaner look.

    Args:
        ax: matplotlib Axes object to clean.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    """Save a matplotlib figure to the plots directory and close it.

    Args:
        fig: matplotlib Figure to save.
        name: Filename (e.g., 'ts_fig01_heatmap_dx.png').

    Side effects:
        Creates PLOT_DIR if it does not exist.
        Writes a PNG file at 150 DPI.
        Closes the figure to free memory.
        Prints the output path to stdout.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


# ── Figure 1: DX Heatmap (Coupled) ──────────────────────────────────────────

def fig01_heatmap_dx(metrics, gaits):
    """Generate a 116x7 heatmap of DX values in coupled mode.

    Rows are gaits sorted by |baseline DX| descending, columns are the 7 DT
    values. A left-side color strip shows the architecture of each gait.
    Uses a diverging RdBu_r colormap centered at 0, clipped at the 95th
    percentile of |DX| for readability.

    Args:
        metrics: Per-gait metrics dict from compute_metrics().
        gaits: List of (gait_name, gait_data, cat_name) tuples.

    Side effects:
        Saves ts_fig01_heatmap_dx.png to PLOT_DIR.
    """
    # Sort gaits by |baseline DX| descending
    gait_names = [g[0] for g in gaits if g[0] in metrics]
    gait_names.sort(key=lambda n: abs(metrics[n]["baseline_dx"]), reverse=True)

    n_gaits = len(gait_names)
    n_dts = len(DT_HZ)

    # Build matrix
    dx_matrix = np.full((n_gaits, n_dts), np.nan)
    archs = []
    for i, gn in enumerate(gait_names):
        m = metrics[gn]
        archs.append(m["architecture"])
        for j, hz in enumerate(DT_HZ):
            val = m["coupled_dxs"].get(str(hz))
            if val is not None:
                dx_matrix[i, j] = val

    # Color limits: symmetric around 0, clipped at 95th percentile
    vmax = np.nanpercentile(np.abs(dx_matrix), 95)
    vmax = max(vmax, 1.0)

    fig, (ax_strip, ax_main) = plt.subplots(
        1, 2, figsize=(12, max(8, n_gaits * 0.12)),
        gridspec_kw={"width_ratios": [0.3, 10]}, sharey=True)

    # Architecture color strip
    arch_colors = np.array([ARCH_COLORS.get(a, "#888888") for a in archs])
    for i, ac in enumerate(arch_colors):
        ax_strip.barh(i, 1, color=ac, height=1.0, align="center")
    ax_strip.set_xlim(0, 1)
    ax_strip.set_yticks(range(n_gaits))
    ax_strip.set_yticklabels(gait_names, fontsize=4)
    ax_strip.set_xticks([])
    ax_strip.set_title("Arch", fontsize=8)
    ax_strip.invert_yaxis()

    # Main heatmap
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_main.imshow(dx_matrix, aspect="auto", cmap="RdBu_r", norm=norm,
                        interpolation="nearest")
    ax_main.set_xticks(range(n_dts))
    ax_main.set_xticklabels(DT_LABELS, fontsize=8)
    ax_main.set_xlabel("Timestep (DT)")
    ax_main.set_title("DX by Timestep (Coupled Mode)")
    ax_main.set_yticks([])

    cbar = plt.colorbar(im, ax=ax_main, shrink=0.6, label="DX (m)")

    # Architecture legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=a)
                       for a, c in ARCH_COLORS.items()]
    ax_strip.legend(handles=legend_elements, loc="lower left", fontsize=5,
                    framealpha=0.8)

    fig.suptitle("Timestep Atlas: DX Heatmap (116 gaits x 7 DTs, coupled mode)",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, "ts_fig01_heatmap_dx.png")


# ── Figure 2: Retention Heatmap ─────────────────────────────────────────────

def fig02_heatmap_retention(metrics, gaits):
    """Generate a 116x7 heatmap of retention values with sign-flip markers.

    Similar layout to fig01 but shows retention_abs (capped at 2.0 for display)
    using an RdYlGn colormap (green = retained, red = collapsed). Cells where
    the displacement sign flipped relative to baseline are marked with a
    multiplication sign.

    Args:
        metrics: Per-gait metrics dict from compute_metrics().
        gaits: List of (gait_name, gait_data, cat_name) tuples.

    Side effects:
        Saves ts_fig02_heatmap_retention.png to PLOT_DIR.
    """
    gait_names = [g[0] for g in gaits if g[0] in metrics]
    gait_names.sort(key=lambda n: abs(metrics[n]["baseline_dx"]), reverse=True)

    n_gaits = len(gait_names)
    n_dts = len(DT_HZ)

    ret_matrix = np.full((n_gaits, n_dts), np.nan)
    flip_matrix = np.full((n_gaits, n_dts), False)
    archs = []

    for i, gn in enumerate(gait_names):
        m = metrics[gn]
        archs.append(m["architecture"])
        for j, hz in enumerate(DT_HZ):
            val = m["retention_abs"].get(str(hz))
            if val is not None:
                ret_matrix[i, j] = min(val, 2.0)  # cap display at 2.0
            fl = m["sign_flips"].get(str(hz))
            if fl:
                flip_matrix[i, j] = True

    fig, (ax_strip, ax_main) = plt.subplots(
        1, 2, figsize=(12, max(8, n_gaits * 0.12)),
        gridspec_kw={"width_ratios": [0.3, 10]}, sharey=True)

    # Architecture strip
    for i, a in enumerate(archs):
        ax_strip.barh(i, 1, color=ARCH_COLORS.get(a, "#888888"),
                      height=1.0, align="center")
    ax_strip.set_xlim(0, 1)
    ax_strip.set_yticks(range(n_gaits))
    ax_strip.set_yticklabels(gait_names, fontsize=4)
    ax_strip.set_xticks([])
    ax_strip.set_title("Arch", fontsize=8)
    ax_strip.invert_yaxis()

    # Retention heatmap
    im = ax_main.imshow(ret_matrix, aspect="auto", cmap="RdYlGn",
                        vmin=0, vmax=1.5, interpolation="nearest")
    ax_main.set_xticks(range(n_dts))
    ax_main.set_xticklabels(DT_LABELS, fontsize=8)
    ax_main.set_xlabel("Timestep (DT)")
    ax_main.set_title("Retention (abs) by Timestep")
    ax_main.set_yticks([])

    # Mark sign flips with 'x'
    for i in range(n_gaits):
        for j in range(n_dts):
            if flip_matrix[i, j]:
                ax_main.text(j, i, "\u00d7", ha="center", va="center",
                             fontsize=5, color="black", fontweight="bold")

    plt.colorbar(im, ax=ax_main, shrink=0.6, label="retention_abs")

    fig.suptitle("Timestep Atlas: Retention Heatmap (\u00d7 = sign flip)",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, "ts_fig02_heatmap_retention.png")


# ── Figure 3: Robustness ────────────────────────────────────────────────────

def fig03_robustness(metrics):
    """Generate a two-panel robustness analysis figure.

    Left panel: Histogram of coupled robustness scores, stacked by architecture,
    showing how each topology distributes across the 0-1 robustness range.

    Right panel: Scatter plot of robustness vs |baseline DX|, colored by
    architecture, to reveal whether faster gaits tend to be more or less
    timestep-stable. A horizontal reference line at robustness=0.5 marks the
    collapse threshold.

    Args:
        metrics: Per-gait metrics dict from compute_metrics().

    Side effects:
        Saves ts_fig03_robustness.png to PLOT_DIR.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    all_m = list(metrics.values())

    # Left: histogram by architecture
    for arch, color in ARCH_COLORS.items():
        vals = [m["robustness"] for m in all_m if m["architecture"] == arch]
        if vals:
            ax1.hist(vals, bins=20, range=(0, 1), alpha=0.6, color=color,
                     label=f"{arch} (n={len(vals)})", edgecolor="white")
    ax1.set_xlabel("Robustness (mean capped retention)")
    ax1.set_ylabel("Count")
    ax1.set_title("Robustness Distribution by Architecture")
    ax1.legend(fontsize=8)
    clean_ax(ax1)

    # Right: robustness vs |baseline DX|
    robustness = np.array([m["robustness"] for m in all_m])
    abs_baseline = np.array([abs(m["baseline_dx"]) for m in all_m])
    arch_list = [m["architecture"] for m in all_m]

    for arch, color in ARCH_COLORS.items():
        mask = [a == arch for a in arch_list]
        if any(mask):
            ax2.scatter(abs_baseline[mask], robustness[mask],
                        c=color, s=20, alpha=0.7, label=arch, edgecolors="none")
    ax2.set_xlabel("|Baseline DX| (m)")
    ax2.set_ylabel("Robustness")
    ax2.set_title("Robustness vs Baseline |DX|")
    ax2.legend(fontsize=8)
    ax2.axhline(0.5, color="gray", ls=":", lw=0.8)
    clean_ax(ax2)

    fig.suptitle("Timestep Robustness Analysis", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "ts_fig03_robustness.png")


# ── Figure 4: Champion Curves ───────────────────────────────────────────────

def fig04_champion_curves(metrics):
    """Generate a 2x4 grid of DX-vs-DT curves for the top 8 gaits.

    Selects the 8 gaits with the largest |baseline DX| and plots their
    displacement across all tested DTs. Each subplot overlays coupled (solid
    blue circles) and decoupled (dashed red squares) curves on shared axes,
    with a vertical reference line at the 240 Hz baseline. Subplot titles
    show the gait name, baseline DX, and robustness score.

    Args:
        metrics: Per-gait metrics dict from compute_metrics().

    Side effects:
        Saves ts_fig04_champion_curves.png to PLOT_DIR.
    """
    sorted_gaits = sorted(metrics.values(),
                          key=lambda m: abs(m["baseline_dx"]), reverse=True)
    top8 = sorted_gaits[:8]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes_flat = axes.flatten()

    for idx, m in enumerate(top8):
        ax = axes_flat[idx]
        gn = m["gait"]

        # Coupled
        coupled_hz = sorted([int(k) for k in m["coupled_dxs"]])
        coupled_dx = [m["coupled_dxs"][str(hz)] for hz in coupled_hz]
        ax.plot(coupled_hz, coupled_dx, "o-", color="#4C72B0", lw=1.5,
                markersize=5, label="Coupled")

        # Decoupled
        dec_hz = sorted([int(k) for k in m["decoupled_dxs"]])
        dec_dx = [m["decoupled_dxs"][str(hz)] for hz in dec_hz]
        if dec_dx:
            ax.plot(dec_hz, dec_dx, "s--", color="#C44E52", lw=1.5,
                    markersize=5, label="Decoupled")

        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.axvline(240, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.set_title(f"{gn}\nbase={m['baseline_dx']:+.1f}m  rob={m['robustness']:.2f}",
                     fontsize=8)
        ax.set_xlabel("DT (Hz)", fontsize=7)
        ax.set_ylabel("DX (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6)
        clean_ax(ax)

    fig.suptitle("Top 8 Gaits: DX vs Timestep (Coupled + Decoupled)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "ts_fig04_champion_curves.png")


# ── Figure 5: Collapse Analysis ─────────────────────────────────────────────

def fig05_collapse(metrics):
    """Generate a two-panel collapse analysis figure.

    Left panel: Histogram showing at which DT (Hz) gaits first collapse
    (retention < 0.5). Also reports how many gaits never collapse.

    Right panel: Scatter plot of coupled vs decoupled robustness, colored
    by architecture. Points above the y=x diagonal indicate gaits that are
    more robust in decoupled mode (suggesting controller-sampling-rate
    sensitivity rather than solver-resolution sensitivity).

    Args:
        metrics: Per-gait metrics dict from compute_metrics().

    Side effects:
        Saves ts_fig05_collapse.png to PLOT_DIR.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    all_m = list(metrics.values())

    # Left: collapse onset histogram
    collapse_vals = [m["collapse_dt"] for m in all_m if m["collapse_dt"] is not None]
    no_collapse = sum(1 for m in all_m if m["collapse_dt"] is None)

    if collapse_vals:
        bins = sorted(set(collapse_vals))
        ax1.hist(collapse_vals, bins=len(bins), color="#C44E52", alpha=0.7,
                 edgecolor="white")
    ax1.set_xlabel("Collapse Onset DT (Hz)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Collapse Onset ({len(collapse_vals)} collapse, "
                  f"{no_collapse} never collapse)")
    clean_ax(ax1)

    # Right: coupled vs decoupled robustness
    coupled_rob = []
    decoupled_rob = []
    arch_list = []
    for m in all_m:
        coupled_rob.append(m["robustness"])
        decoupled_rob.append(m["decoupled_robustness"])
        arch_list.append(m["architecture"])

    coupled_rob = np.array(coupled_rob)
    decoupled_rob = np.array(decoupled_rob)

    for arch, color in ARCH_COLORS.items():
        mask = [a == arch for a in arch_list]
        if any(mask):
            ax2.scatter(coupled_rob[mask], decoupled_rob[mask],
                        c=color, s=20, alpha=0.7, label=arch, edgecolors="none")

    # y=x line
    lim = [0, max(1.0, max(coupled_rob.max(), decoupled_rob.max()) * 1.05)]
    ax2.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("Coupled Robustness")
    ax2.set_ylabel("Decoupled Robustness")
    ax2.set_title("Coupled vs Decoupled Robustness")
    ax2.legend(fontsize=8)
    clean_ax(ax2)

    fig.suptitle("Timestep Collapse & Mode Comparison", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "ts_fig05_collapse.png")


# ── Console Summary ──────────────────────────────────────────────────────────

def print_summary(metrics):
    """Print comprehensive summary tables of timestep atlas results to console.

    Outputs six sections:
        1. Coupled robustness distribution (mean, median, P10, P90).
        2. Decoupled robustness distribution.
        3. Robustness breakdown by architecture.
        4. Behavior stats (collapse rate, sign-flip rate, monotonicity).
        5. Top 10 most robust gaits.
        6. Bottom 10 least robust gaits.
        7. Mode interpretation (solver vs sampling-rate artifact counts).

    Args:
        metrics: Per-gait metrics dict from compute_metrics().

    Side effects:
        Prints formatted text to stdout.
    """
    all_m = list(metrics.values())
    robustness = np.array([m["robustness"] for m in all_m])
    dec_robustness = np.array([m["decoupled_robustness"] for m in all_m])
    abs_baseline = np.array([abs(m["baseline_dx"]) for m in all_m])

    print(f"\n{'='*80}")
    print("TIMESTEP ATLAS — SUMMARY")
    print(f"{'='*80}")

    # Overall robustness stats
    print(f"\n  COUPLED ROBUSTNESS:")
    print(f"    Mean:   {np.mean(robustness):.3f}")
    print(f"    Median: {np.median(robustness):.3f}")
    print(f"    P10:    {np.percentile(robustness, 10):.3f}")
    print(f"    P90:    {np.percentile(robustness, 90):.3f}")

    print(f"\n  DECOUPLED ROBUSTNESS:")
    print(f"    Mean:   {np.mean(dec_robustness):.3f}")
    print(f"    Median: {np.median(dec_robustness):.3f}")
    print(f"    P10:    {np.percentile(dec_robustness, 10):.3f}")
    print(f"    P90:    {np.percentile(dec_robustness, 90):.3f}")

    # Architecture comparison
    print(f"\n  ROBUSTNESS BY ARCHITECTURE:")
    print(f"    {'Architecture':<18} {'N':>4} {'Coupled':>10} {'Decoupled':>10} {'Monotonic%':>10}")
    print("    " + "-" * 56)
    for arch in ARCH_COLORS:
        am = [m for m in all_m if m["architecture"] == arch]
        if not am:
            continue
        rob = np.mean([m["robustness"] for m in am])
        dec = np.mean([m["decoupled_robustness"] for m in am])
        mono = np.mean([m["monotonic"] for m in am]) * 100
        print(f"    {arch:<18} {len(am):4d} {rob:10.3f} {dec:10.3f} {mono:9.1f}%")

    # Collapse stats
    collapse_count = sum(1 for m in all_m if m["collapse_dt"] is not None)
    sign_flip_count = sum(1 for m in all_m
                          if any(m["sign_flips"].get(str(hz), False)
                                 for hz in DT_HZ if hz > 240))
    monotonic_count = sum(1 for m in all_m if m["monotonic"])

    print(f"\n  BEHAVIOR STATS:")
    print(f"    Gaits with collapse:    {collapse_count}/{len(all_m)} "
          f"({collapse_count/len(all_m)*100:.1f}%)")
    print(f"    Gaits with sign flip:   {sign_flip_count}/{len(all_m)} "
          f"({sign_flip_count/len(all_m)*100:.1f}%)")
    print(f"    Monotonic decrease:     {monotonic_count}/{len(all_m)} "
          f"({monotonic_count/len(all_m)*100:.1f}%)")

    # Top 10 most robust
    sorted_by_rob = sorted(all_m, key=lambda m: m["robustness"], reverse=True)
    print(f"\n  TOP 10 MOST ROBUST:")
    print(f"    {'Gait':<35} {'Arch':<16} {'Base DX':>8} {'Rob':>6} {'DecRob':>7} {'Mono':>5}")
    print("    " + "-" * 80)
    for m in sorted_by_rob[:10]:
        mono_s = "yes" if m["monotonic"] else "no"
        print(f"    {m['gait']:<35} {m['architecture']:<16} "
              f"{m['baseline_dx']:+8.1f} {m['robustness']:6.3f} "
              f"{m['decoupled_robustness']:7.3f} {mono_s:>5}")

    # Bottom 10 (least robust)
    print(f"\n  BOTTOM 10 LEAST ROBUST:")
    print(f"    {'Gait':<35} {'Arch':<16} {'Base DX':>8} {'Rob':>6} {'DecRob':>7} {'Collapse':>8}")
    print("    " + "-" * 86)
    for m in sorted_by_rob[-10:]:
        col = str(m["collapse_dt"]) if m["collapse_dt"] else "never"
        print(f"    {m['gait']:<35} {m['architecture']:<16} "
              f"{m['baseline_dx']:+8.1f} {m['robustness']:6.3f} "
              f"{m['decoupled_robustness']:7.3f} {col:>8}")

    # Coupled vs decoupled interpretation
    # Gaits where decoupled holds but coupled fails -> sampling rate artifact
    # Gaits where both fail -> solver resolution artifact
    coupled_fail = [m for m in all_m if m["robustness"] < 0.5]
    dec_fail_too = [m for m in coupled_fail if m["decoupled_robustness"] < 0.5]
    dec_holds = [m for m in coupled_fail
                 if m["decoupled_robustness"] >= 0.5]

    print(f"\n  MODE INTERPRETATION:")
    print(f"    Coupled robustness < 0.5:   {len(coupled_fail)} gaits")
    print(f"      Decoupled also < 0.5:     {len(dec_fail_too)} (solver artifact)")
    print(f"      Decoupled holds >= 0.5:   {len(dec_holds)} (sampling rate artifact)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Entry point: parse arguments, run the full DT sweep, compute metrics, and generate outputs.

    Orchestrates the complete timestep atlas pipeline:
        1. Back up brain.nndf (restored after sweep).
        2. Load all 116 gaits from the synapse zoo.
        3. Run coupled and decoupled simulations across all DT values.
        4. Compute per-gait metrics (retention, robustness, collapse, etc.).
        5. Print a console summary.
        6. Generate 5 diagnostic figures.
        7. Save the full results as JSON.

    Side effects:
        Writes artifacts/timestep_atlas.json, checkpoint JSONL, and 5 PNG figures.
        Temporarily modifies brain.nndf (backed up and restored).
    """
    ap = argparse.ArgumentParser(
        description="Timestep Atlas: sweep 116 gaits across 7 DTs in coupled/decoupled modes.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint (skip completed sims)")
    args = ap.parse_args()

    t_start = time.perf_counter()

    # Back up brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    # Load zoo
    print(f"Loading {ZOO_PATH} ...")
    zoo = json.loads(ZOO_PATH.read_text())

    gaits = []
    for cat_name, cat in zoo["categories"].items():
        for gait_name, gait_data in cat["gaits"].items():
            gaits.append((gait_name, gait_data, cat_name))
    print(f"  Loaded {len(gaits)} gaits")

    total_sims = len(gaits) * (len(DT_VALUES) + len(DECOUPLED_DTS))
    print(f"  Total simulation budget: {total_sims} sims "
          f"({len(gaits)} gaits x ({len(DT_VALUES)} coupled + {len(DECOUPLED_DTS)} decoupled))")

    # Run sweep
    print(f"\n{'='*80}")
    print("RUNNING DT SWEEP")
    print(f"{'='*80}")
    results = run_sweep(gaits, resume=args.resume)

    # Restore brain.nndf
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # Compute metrics
    print(f"\n{'='*80}")
    print("COMPUTING METRICS")
    print(f"{'='*80}")
    metrics = compute_metrics(gaits, results)
    print(f"  Computed metrics for {len(metrics)} gaits")

    # Console summary
    print_summary(metrics)

    # Figures
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")
    fig01_heatmap_dx(metrics, gaits)
    fig02_heatmap_retention(metrics, gaits)
    fig03_robustness(metrics)
    fig04_champion_curves(metrics)
    fig05_collapse(metrics)

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_gaits": len(gaits),
        "n_dts": len(DT_VALUES),
        "dt_values_hz": DT_HZ,
        "modes": ["coupled", "decoupled"],
        "decoupled_dts_hz": [round(1/dt) for dt in DECOUPLED_DTS],
        "total_sims": len(results),
        "coupled_robustness_mean": round(float(np.mean([m["robustness"] for m in metrics.values()])), 4),
        "decoupled_robustness_mean": round(float(np.mean([m["decoupled_robustness"] for m in metrics.values()])), 4),
        "collapse_count": sum(1 for m in metrics.values() if m["collapse_dt"] is not None),
        "monotonic_count": sum(1 for m in metrics.values() if m["monotonic"]),
    }

    output = {
        "meta": summary,
        "gait_metrics": metrics,
        "raw_results": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
