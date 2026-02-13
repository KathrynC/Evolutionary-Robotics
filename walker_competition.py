#!/usr/bin/env python3
"""
walker_competition.py

Role:
    Head-to-head competition of 5 gaitspace walker algorithms, each with a
    1,000-evaluation budget on the 6D synapse weight space. Compares
    optimization strategies across multiple performance dimensions and
    produces a ranked leaderboard with supporting visualizations.

Walkers:
    1. Hill Climber        -- accept perturbation if |DX| improves
    2. Ridge Walker        -- Pareto-non-dominated moves on (|DX|, efficiency)
    3. Cliff Mapper        -- probe 10 directions, walk toward steepest cliff
    4. Novelty Seeker      -- pick most behaviorally novel candidate from 5
    5. Ensemble Explorer   -- 20 parallel hill climbers with teleportation

Scoring dimensions (6 metrics, all higher-is-better):
    best_dx, best_efficiency, best_speed, pareto_size, diversity, unique_regimes

Notes:
    - Each walker gets exactly 1,000 simulation evaluations (budget-matched).
    - All walkers share the same evaluate() function (headless PyBullet, DIRECT mode).
    - The simulation harness is self-contained: writes brain.nndf, runs PyBullet,
      extracts metrics via compute_beer_analytics.
    - brain.nndf is backed up before the run and restored afterward.
    - PCA projections (behavioral space + weight space) are computed with numpy only.

Outputs:
    artifacts/walker_competition.json
    artifacts/plots/comp_fig01_leaderboard.png
    artifacts/plots/comp_fig02_best_of_n.png
    artifacts/plots/comp_fig03_pareto.png
    artifacts/plots/comp_fig04_diversity.png
    artifacts/plots/comp_fig05_trajectories.png
    artifacts/plots/comp_fig06_radar.png

Usage:
    conda activate er
    python3 walker_competition.py
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
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from compute_beer_analytics import compute_all, DT, NumpyEncoder

# ── Constants ────────────────────────────────────────────────────────────────

SENSOR_NEURONS = [0, 1, 2]
MOTOR_NEURONS = [3, 4]
WEIGHT_NAMES = [f"w{s}{m}" for s in SENSOR_NEURONS for m in MOTOR_NEURONS]
N_WEIGHTS = len(WEIGHT_NAMES)  # 6

EVAL_BUDGET = 1000
PLOT_DIR = PROJECT / "artifacts" / "plots"
OUT_JSON = PROJECT / "artifacts" / "walker_competition.json"

WALKER_NAMES = [
    "Hill Climber",
    "Ridge Walker",
    "Cliff Mapper",
    "Novelty Seeker",
    "Ensemble Explorer",
]
WALKER_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


# ── Simulation (shared from random_search_500.py pattern) ───────────────────

def write_brain(weights):
    """Write a brain.nndf file from a weight dict mapping 'wXY' to float values.

    Defines the standard 3-sensor, 2-motor topology with full connectivity
    (6 synapses). Overwrites PROJECT/brain.nndf.

    Args:
        weights: dict with keys like "w03", "w04", ... "w24" mapping
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
        for s in SENSOR_NEURONS:
            for m in MOTOR_NEURONS:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def evaluate(weights):
    """Run one headless simulation and return a compact metrics dict.

    Writes brain.nndf, runs a full PyBullet simulation in DIRECT mode,
    records per-step telemetry, and computes Beer-framework analytics.

    Args:
        weights: dict mapping synapse names ("wXY") to float values.

    Returns:
        dict with keys: dx, abs_dx, speed, efficiency, work_proxy,
        phase_lock, entropy, roll_dom.

    Side effects:
        - Overwrites brain.nndf via write_brain().
        - Creates and destroys a PyBullet physics connection.
    """
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
    roll_a = np.empty(n_steps); pitch_a = np.empty(n_steps); yaw_a = np.empty(n_steps)
    ct = np.empty(n_steps, dtype=bool)
    cb = np.empty(n_steps, dtype=bool)
    cf = np.empty(n_steps, dtype=bool)
    j0p = np.empty(n_steps); j0v = np.empty(n_steps); j0t = np.empty(n_steps)
    j1p = np.empty(n_steps); j1v = np.empty(n_steps); j1t = np.empty(n_steps)

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

        x[i] = pos[0]; y[i] = pos[1]; z[i] = pos[2]
        vx[i] = vel_lin[0]; vy[i] = vel_lin[1]; vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]; wy[i] = vel_ang[1]; wz[i] = vel_ang[2]
        roll_a[i] = rpy_vals[0]; pitch_a[i] = rpy_vals[1]; yaw_a[i] = rpy_vals[2]

        # Detect ground contact per link: cp[3] is the link index on bodyA.
        # link index -1 is the base link (Torso).
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
        "t": t_arr, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll_a, "pitch": pitch_a, "yaw": yaw_a,
        "contact_torso": ct, "contact_back": cb, "contact_front": cf,
        "j0_pos": j0p, "j0_vel": j0v, "j0_tau": j0t,
        "j1_pos": j1p, "j1_vel": j1v, "j1_tau": j1t,
    }

    analytics = compute_all(data, DT)
    o = analytics["outcome"]
    coord = analytics["coordination"]
    contact = analytics["contact"]
    ra = analytics["rotation_axis"]

    return {
        "dx": o["dx"],
        "abs_dx": abs(o["dx"]),
        "speed": o["mean_speed"],
        "efficiency": o["distance_per_work"],
        "work_proxy": o["work_proxy"],
        "phase_lock": coord["phase_lock_score"],
        "entropy": contact["contact_entropy_bits"],
        "roll_dom": ra["axis_dominance"][0],
    }


# ── Shared helpers ───────────────────────────────────────────────────────────

def random_weights():
    """Generate a random weight dict with each value sampled uniformly from [-1, 1].

    Returns:
        dict mapping each synapse name in WEIGHT_NAMES to a random float.
    """
    return {wn: random.uniform(-1, 1) for wn in WEIGHT_NAMES}


def weights_to_vec(weights):
    """Convert weight dict to a numpy array, ordered by WEIGHT_NAMES.

    Args:
        weights: dict mapping synapse names to float values.

    Returns:
        numpy array of shape (N_WEIGHTS,).
    """
    return np.array([weights[wn] for wn in WEIGHT_NAMES])


def vec_to_weights(vec):
    """Convert a numpy array back to a weight dict, keyed by WEIGHT_NAMES.

    Args:
        vec: numpy array of shape (N_WEIGHTS,).

    Returns:
        dict mapping synapse names to float values.
    """
    return {wn: float(vec[i]) for i, wn in enumerate(WEIGHT_NAMES)}


def perturb(weights, radius):
    """Perturb weights along a random 6D unit vector at given radius.

    Generates a uniformly random direction on the N_WEIGHTS-dimensional
    unit sphere and moves the weight vector by exactly `radius` in that
    direction. The result is not clamped, so weights may exceed [-1, 1].

    Args:
        weights: dict mapping synapse names to float values.
        radius: step size (Euclidean distance in weight space).

    Returns:
        New weight dict with the perturbation applied.
    """
    direction = np.random.randn(N_WEIGHTS)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:  # guard against near-zero random vector (astronomically rare)
        direction = np.ones(N_WEIGHTS)
        norm = np.linalg.norm(direction)
    direction = direction / norm

    vec = weights_to_vec(weights)
    new_vec = vec + radius * direction
    return vec_to_weights(new_vec)


def behavioral_vec(metrics):
    """Extract a 6D behavioral descriptor from a metrics dict (raw, unnormalized).

    The descriptor captures six orthogonal aspects of gait behavior:
    displacement, speed, efficiency, coordination, contact complexity,
    and rotation axis dominance.

    Args:
        metrics: dict returned by evaluate().

    Returns:
        numpy array of shape (6,): [abs_dx, speed, efficiency,
        phase_lock, entropy, roll_dom].
    """
    return np.array([
        metrics["abs_dx"],
        metrics["speed"],
        metrics["efficiency"],
        metrics["phase_lock"],
        metrics["entropy"],
        metrics["roll_dom"],
    ])


def normalize_behavioral_vecs(vecs):
    """Min-max normalize behavioral vectors to [0, 1] per dimension.

    Constant dimensions (range < 1e-12) are set to range 1.0 to avoid
    division by zero, effectively zeroing those dimensions.

    Args:
        vecs: list of behavioral vectors (each a numpy array or list).

    Returns:
        numpy array of shape (len(vecs), 6) with values in [0, 1].
    """
    arr = np.array(vecs)
    if len(arr) == 0:
        return arr
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0  # avoid division by zero for constant dimensions
    return (arr - mins) / ranges


def behavioral_distance(bv1, bv2):
    """Compute the Euclidean distance between two behavioral vectors.

    Args:
        bv1: first behavioral vector (array-like).
        bv2: second behavioral vector (array-like).

    Returns:
        float: L2 distance between the two vectors.
    """
    return np.linalg.norm(np.array(bv1) - np.array(bv2))


def novelty(bvec, archive_bvecs, k=15):
    """Compute the novelty score as mean distance to k nearest neighbors.

    Args:
        bvec: behavioral vector to score (array-like).
        archive_bvecs: list of archived behavioral vectors to compare against.
        k: number of nearest neighbors to average over. If the archive
           has fewer than k entries, all entries are used.

    Returns:
        float: mean Euclidean distance to the k nearest archive entries.
        Returns inf if the archive is empty.
    """
    if len(archive_bvecs) == 0:
        return float('inf')
    dists = [behavioral_distance(bvec, a) for a in archive_bvecs]
    dists.sort()
    k_actual = min(k, len(dists))
    return np.mean(dists[:k_actual])


def is_dominated(a_metrics, b_metrics):
    """Test Pareto dominance: True if b dominates a on (|DX|, efficiency).

    Dominance means b >= a on both objectives and b > a on at least one.

    Args:
        a_metrics: metrics dict for the candidate being tested.
        b_metrics: metrics dict for the potential dominator.

    Returns:
        bool: True if b Pareto-dominates a.
    """
    a_dx, a_eff = a_metrics["abs_dx"], a_metrics["efficiency"]
    b_dx, b_eff = b_metrics["abs_dx"], b_metrics["efficiency"]
    return (b_dx >= a_dx and b_eff >= a_eff) and (b_dx > a_dx or b_eff > a_eff)


def pareto_front(archive):
    """Extract the Pareto front from an archive of metrics dicts.

    Uses brute-force O(n^2) pairwise dominance checks. Suitable for
    archives up to ~1000 points.

    Args:
        archive: list of metrics dicts, each with "abs_dx" and "efficiency".

    Returns:
        list of non-dominated metrics dicts (the Pareto front).
    """
    front = []
    for i, a in enumerate(archive):
        dominated = False
        for j, b in enumerate(archive):
            if i != j and is_dominated(a, b):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


# ── Walker implementations ──────────────────────────────────────────────────

def run_hill_climber():
    """Walker 1: Simple hill climber maximizing |DX|.

    Strategy: greedy local search with radius-0.1 perturbations.
    Accepts a candidate only if it strictly improves |DX|.
    Budget: 1 initial + 999 steps = 1,000 evaluations.

    Returns:
        tuple of (archive, weight_history) where:
            archive: list of dicts with "weights" and "metrics" for every evaluation.
            weight_history: list of numpy weight vectors (one per evaluation).
    """
    print("\n[1/5] Hill Climber (999 steps)...")
    archive = []
    weight_history = []

    # Initial random point
    w = random_weights()
    m = evaluate(w)
    archive.append({"weights": w, "metrics": m})
    weight_history.append(weights_to_vec(w))

    # Greedy local search: each step perturbs and keeps only strict improvements
    for step in range(999):
        w_new = perturb(w, radius=0.1)
        m_new = evaluate(w_new)
        archive.append({"weights": w_new, "metrics": m_new})
        weight_history.append(weights_to_vec(w_new))

        # Accept only if strictly better -- classic hill climbing acceptance rule
        if m_new["abs_dx"] > m["abs_dx"]:
            w = w_new
            m = m_new

        if (step + 1) % 200 == 0:
            print(f"    step {step+1}/999  best |DX|={m['abs_dx']:.3f}  "
                  f"evals={len(archive)}")

    print(f"    Done. {len(archive)} evals, best |DX|={m['abs_dx']:.3f}")
    return archive, weight_history


def run_ridge_walker():
    """Walker 2: Pareto-non-dominated moves on (|DX|, efficiency).

    Strategy: multi-objective search. Each step generates 3 candidates,
    filters to those not dominated by the current point, and picks the
    one farthest in 2D objective space (encouraging Pareto front
    exploration rather than stagnation).
    Budget: 1 initial + 3 * 333 = 1,000 evaluations.

    Returns:
        tuple of (archive, weight_history) where:
            archive: list of dicts with "weights" and "metrics" for every evaluation.
            weight_history: list of numpy weight vectors (one per evaluation).
    """
    print("\n[2/5] Ridge Walker (333 steps, 3 candidates/step)...")
    archive = []
    weight_history = []

    w = random_weights()
    m = evaluate(w)
    archive.append({"weights": w, "metrics": m})
    weight_history.append(weights_to_vec(w))

    # Multi-objective search: walk along the Pareto ridge of (|DX|, efficiency)
    for step in range(333):
        # Generate 3 candidate perturbations per step (budget: 3 * 333 = 999 + 1 init = 1000)
        candidates = []
        for _ in range(3):
            w_c = perturb(w, radius=0.1)
            m_c = evaluate(w_c)
            archive.append({"weights": w_c, "metrics": m_c})
            weight_history.append(weights_to_vec(w_c))
            candidates.append((w_c, m_c))

        # Filter to non-dominated candidates (not dominated by current point)
        # A candidate passes if it is at least as good as current on both objectives
        non_dominated = [(wc, mc) for wc, mc in candidates if not is_dominated(mc, m)]

        if non_dominated:
            # Among non-dominated, pick the one farthest from current in 2D objective space.
            # This encourages exploration along the Pareto front rather than staying put.
            best_dist = -1
            best_wc, best_mc = None, None
            for wc, mc in non_dominated:
                dist = np.sqrt((mc["abs_dx"] - m["abs_dx"])**2 +
                               (mc["efficiency"] - m["efficiency"])**2)
                if dist > best_dist:
                    best_dist = dist
                    best_wc, best_mc = wc, mc
            w, m = best_wc, best_mc

        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/333  |DX|={m['abs_dx']:.3f}  "
                  f"eff={m['efficiency']:.5f}  evals={len(archive)}")

    # Use remaining budget (1000 - 1 - 333*3 = 0, exact)
    print(f"    Done. {len(archive)} evals, final |DX|={m['abs_dx']:.3f}")
    return archive, weight_history


def run_cliff_mapper():
    """Walker 3: Probe 10 directions, walk toward steepest cliff.

    Strategy: gradient-like exploration. Each step probes 10 nearby points
    (radius 0.05) and moves to whichever shows the largest absolute change
    in |DX|, regardless of sign. This deliberately seeks "cliffs" in the
    fitness landscape -- regions of high sensitivity.
    Budget: 1 initial + 10 * 99 + 9 remainder = 1,000 evaluations.

    Returns:
        tuple of (archive, weight_history) where:
            archive: list of dicts with "weights" and "metrics" for every evaluation.
            weight_history: list of numpy weight vectors (one per evaluation).
    """
    print("\n[3/5] Cliff Mapper (99 steps, 10 probes/step)...")
    archive = []
    weight_history = []

    w = random_weights()
    m = evaluate(w)
    archive.append({"weights": w, "metrics": m})
    weight_history.append(weights_to_vec(w))

    # Gradient-based exploration: probe multiple directions, follow steepest change
    for step in range(99):
        # Sample 10 nearby points to estimate the local gradient landscape
        # Uses a smaller radius (0.05) than other walkers for finer gradient resolution
        probes = []
        for _ in range(10):
            w_p = perturb(w, radius=0.05)
            m_p = evaluate(w_p)
            archive.append({"weights": w_p, "metrics": m_p})
            weight_history.append(weights_to_vec(w_p))
            # Track absolute change in |DX| -- both improvements and drops count
            delta_dx = abs(m_p["abs_dx"] - m["abs_dx"])
            probes.append((w_p, m_p, delta_dx))

        # Move to probe with largest |delta_DX| — seek disruption, not necessarily improvement.
        # This deliberately walks toward "cliffs" in the fitness landscape.
        best_probe = max(probes, key=lambda x: x[2])
        w, m = best_probe[0], best_probe[1]

        if (step + 1) % 25 == 0:
            print(f"    step {step+1}/99  |DX|={m['abs_dx']:.3f}  "
                  f"max delta={best_probe[2]:.3f}  evals={len(archive)}")

    # Remaining budget: 1000 - 1 - 99*10 = 9; spend on random perturbations
    remaining = EVAL_BUDGET - len(archive)
    for _ in range(remaining):
        w_r = perturb(w, radius=0.1)
        m_r = evaluate(w_r)
        archive.append({"weights": w_r, "metrics": m_r})
        weight_history.append(weights_to_vec(w_r))

    print(f"    Done. {len(archive)} evals, final |DX|={m['abs_dx']:.3f}")
    return archive, weight_history


def run_novelty_seeker():
    """Walker 4: Pick most behaviorally novel candidate from 5.

    Strategy: novelty-driven search. Each step generates 5 candidates
    (radius 0.2), normalizes all behavioral vectors together, and moves
    to whichever candidate has the highest k-NN novelty score -- regardless
    of its fitness. This maximizes behavioral diversity rather than
    optimizing any single metric.
    Budget: 1 initial + 5 * 199 + 4 remainder = 1,000 evaluations.

    Returns:
        tuple of (archive, weight_history) where:
            archive: list of dicts with "weights" and "metrics" for every evaluation.
            weight_history: list of numpy weight vectors (one per evaluation).
    """
    print("\n[4/5] Novelty Seeker (199 steps, 5 candidates/step)...")
    archive = []
    weight_history = []
    behavior_archive = []  # list of behavioral vectors for novelty computation

    w = random_weights()
    m = evaluate(w)
    archive.append({"weights": w, "metrics": m})
    weight_history.append(weights_to_vec(w))
    behavior_archive.append(behavioral_vec(m))

    # Novelty-driven search: prioritize behavioral diversity over fitness
    # Uses a larger perturbation radius (0.2) to encourage exploration
    for step in range(199):
        # Generate 5 candidates per step (budget: 5 * 199 = 995 + 1 init = 996)
        candidates = []
        for _ in range(5):
            w_c = perturb(w, radius=0.2)
            m_c = evaluate(w_c)
            archive.append({"weights": w_c, "metrics": m_c})
            weight_history.append(weights_to_vec(w_c))
            bv = behavioral_vec(m_c)
            candidates.append((w_c, m_c, bv))

        # Normalize archive + candidates together so distances are scale-invariant.
        # Must include candidates in normalization to avoid biased distance comparisons.
        all_bvecs = behavior_archive + [c[2] for c in candidates]
        normalized = normalize_behavioral_vecs(all_bvecs)
        n_archive = len(behavior_archive)
        norm_archive = normalized[:n_archive]

        # Pick the candidate most distant from existing archive (k-NN novelty score)
        best_nov = -1
        best_idx = 0
        for ci, (wc, mc, bv) in enumerate(candidates):
            norm_bv = normalized[n_archive + ci]
            nov = novelty(norm_bv, norm_archive, k=15)
            if nov > best_nov:
                best_nov = nov
                best_idx = ci

        # Move to the most novel candidate regardless of its fitness
        w, m = candidates[best_idx][0], candidates[best_idx][1]
        # Add all candidates to behavior archive (not just the winner)
        # to build a denser archive for better novelty estimates
        for _, _, bv in candidates:
            behavior_archive.append(bv)

        if (step + 1) % 50 == 0:
            print(f"    step {step+1}/199  |DX|={m['abs_dx']:.3f}  "
                  f"novelty={best_nov:.3f}  evals={len(archive)}")

    # Remaining: 1000 - 1 - 199*5 = 4
    remaining = EVAL_BUDGET - len(archive)
    for _ in range(remaining):
        w_r = perturb(w, radius=0.2)
        m_r = evaluate(w_r)
        archive.append({"weights": w_r, "metrics": m_r})
        weight_history.append(weights_to_vec(w_r))

    print(f"    Done. {len(archive)} evals")
    return archive, weight_history


def run_ensemble_explorer():
    """Walker 5: 20 parallel hill climbers with teleportation.

    Strategy: island-model search. 20 walkers hill-climb independently.
    Every 10 steps, walkers that have converged behaviorally (normalized
    distance < 0.3) are "teleported" -- the worse walker is reset to a
    random location to maintain population diversity.
    Budget: 20 initial + 20 * 49 = 1,000 evaluations.

    Returns:
        tuple of (archive, weight_history) where:
            archive: list of dicts with "weights" and "metrics" for every evaluation.
            weight_history: list of numpy weight vectors (one per evaluation).
    """
    print("\n[5/5] Ensemble Explorer (20 walkers x 49 steps)...")
    n_walkers = 20
    n_steps = 49
    archive = []
    weight_history = []

    # Initialize 20 walkers
    walkers_w = []
    walkers_m = []
    for _ in range(n_walkers):
        w = random_weights()
        m = evaluate(w)
        archive.append({"weights": w, "metrics": m})
        weight_history.append(weights_to_vec(w))
        walkers_w.append(w)
        walkers_m.append(m)

    # 20 init evals done. Budget: 1000 - 20 = 980 for steps. 980 / 20 = 49 steps.
    for step in range(n_steps):
        # Independent hill climbing: each walker perturbs and accepts improvements
        for wi in range(n_walkers):
            w_new = perturb(walkers_w[wi], radius=0.1)
            m_new = evaluate(w_new)
            archive.append({"weights": w_new, "metrics": m_new})
            weight_history.append(weights_to_vec(w_new))

            if m_new["abs_dx"] > walkers_m[wi]["abs_dx"]:
                walkers_w[wi] = w_new
                walkers_m[wi] = m_new

        # Every 10 steps: teleportation to prevent convergence to the same basin.
        # Walkers that have become behaviorally similar are "teleported" to random
        # positions in weight space, maintaining population diversity.
        if (step + 1) % 10 == 0:
            # Compute behavioral vectors for all walkers
            bvecs = [behavioral_vec(walkers_m[wi]) for wi in range(n_walkers)]
            norm_bvecs = normalize_behavioral_vecs(bvecs)

            # Find pairs within threshold (0.3 in normalized behavioral space)
            for wi in range(n_walkers):
                for wj in range(wi + 1, n_walkers):
                    dist = np.linalg.norm(norm_bvecs[wi] - norm_bvecs[wj])
                    if dist < 0.3:
                        # Teleport the worse one to a random location
                        worse = wi if walkers_m[wi]["abs_dx"] < walkers_m[wj]["abs_dx"] else wj
                        walkers_w[worse] = random_weights()
                        # Don't evaluate here — it'll be evaluated on next step.
                        # Zero metrics so the walker won't be incorrectly treated as high-fitness.
                        walkers_m[worse] = {"abs_dx": 0, "dx": 0, "speed": 0,
                                            "efficiency": 0, "work_proxy": 0,
                                            "phase_lock": 0, "entropy": 0, "roll_dom": 0}

        if (step + 1) % 10 == 0:
            best_dx = max(walkers_m[wi]["abs_dx"] for wi in range(n_walkers))
            print(f"    step {step+1}/{n_steps}  best walker |DX|={best_dx:.3f}  "
                  f"evals={len(archive)}")

    print(f"    Done. {len(archive)} evals")
    return archive, weight_history


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_walkers(all_archives):
    """Compute 6 competition metrics for each walker to build the leaderboard.

    Metrics computed per walker:
        best_dx: highest |DX| found in the archive.
        best_efficiency: highest efficiency among gaits with |DX| > 2m.
        best_speed: highest mean speed found.
        pareto_size: number of Pareto-non-dominated points on (|DX|, efficiency).
        diversity: mean pairwise behavioral distance (sampled, 200 pairs).
        unique_regimes: number of non-empty clusters from k-means (k=10).

    Args:
        all_archives: list of 5 archives, one per walker. Each archive is a
                      list of dicts with "weights" and "metrics".

    Returns:
        dict mapping walker name to a dict of the 6 metric values.
    """
    scores = {}

    for wi, name in enumerate(WALKER_NAMES):
        archive = all_archives[wi]
        metrics_list = [e["metrics"] for e in archive]

        # Best |DX|
        best_dx = max(m["abs_dx"] for m in metrics_list)

        # Best efficiency (among evals with |DX| > 2m) -- filter out near-stationary
        # gaits where efficiency is meaningless (tiny work -> inflated ratio)
        efficient = [m["efficiency"] for m in metrics_list if m["abs_dx"] > 2.0]
        best_eff = max(efficient) if efficient else 0.0

        # Best speed
        best_speed = max(m["speed"] for m in metrics_list)

        # Pareto front size on (|DX|, efficiency)
        pf = pareto_front(metrics_list)
        pareto_size = len(pf)

        # Diversity: mean pairwise behavioral distance (sample 200 pairs).
        # Sampling avoids O(n^2) cost for archives of 1000 points.
        bvecs = [behavioral_vec(m) for m in metrics_list]
        norm_bvecs = normalize_behavioral_vecs(bvecs)
        n_samples = min(200, len(norm_bvecs) * (len(norm_bvecs) - 1) // 2)
        pair_dists = []
        for _ in range(n_samples):
            i, j = random.sample(range(len(norm_bvecs)), 2)
            pair_dists.append(np.linalg.norm(norm_bvecs[i] - norm_bvecs[j]))
        diversity = np.mean(pair_dists) if pair_dists else 0.0

        # Unique regimes: k-means k=10 on behavioral vecs
        unique_regimes = count_unique_regimes(norm_bvecs, k=10)

        scores[name] = {
            "best_dx": best_dx,
            "best_efficiency": best_eff,
            "best_speed": best_speed,
            "pareto_size": pareto_size,
            "diversity": diversity,
            "unique_regimes": unique_regimes,
        }

    return scores


def count_unique_regimes(norm_bvecs, k=10, max_iter=50):
    """Count unique behavioral regimes via k-means clustering (numpy-only).

    Runs Lloyd's algorithm with random initialization and returns the number
    of clusters that contain at least one point. Convergence is detected
    when centroids stop moving (within floating point tolerance).

    Args:
        norm_bvecs: list or array of normalized behavioral vectors.
        k: number of clusters to initialize. Defaults to 10.
        max_iter: maximum Lloyd iterations. Defaults to 50.

    Returns:
        int: number of non-empty clusters (between 1 and k).
    """
    if len(norm_bvecs) < k:
        return len(norm_bvecs)

    data = np.array(norm_bvecs)
    n = len(data)

    # Initialize centroids randomly
    indices = random.sample(range(n), k)
    centroids = data[indices].copy()

    for _ in range(max_iter):
        # Assign points to nearest centroid
        dists = np.zeros((n, k))
        for ci in range(k):
            dists[:, ci] = np.linalg.norm(data - centroids[ci], axis=1)
        labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                new_centroids[ci] = data[mask].mean(axis=0)
            else:
                new_centroids[ci] = centroids[ci]

        # Converge when centroids stop moving (within floating point tolerance)
        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    # Count non-empty clusters -- some of the k=10 clusters may be empty
    unique_labels = len(set(labels.tolist()))
    return unique_labels


def compute_ranks(scores):
    """Rank walkers 1--5 on each metric (lower rank = better).

    All metrics are treated as higher-is-better. The overall rank is the
    sum of per-metric ranks, with ties broken by the best_dx rank
    (displacement as tiebreaker).

    Args:
        scores: dict mapping walker name to a dict of 6 metric values
                (as returned by score_walkers).

    Returns:
        dict mapping walker name to a dict containing:
            - per-metric ranks (int 1--5 for each of the 6 metrics)
            - "total": sum of the 6 ranks
            - "overall": final ranking (1 = winner)
    """
    metric_keys = ["best_dx", "best_efficiency", "best_speed",
                   "pareto_size", "diversity", "unique_regimes"]
    # All metrics: higher is better
    ranks = {name: {} for name in WALKER_NAMES}

    for mk in metric_keys:
        vals = [(name, scores[name][mk]) for name in WALKER_NAMES]
        vals.sort(key=lambda x: x[1], reverse=True)  # highest first
        for rank, (name, _) in enumerate(vals, 1):
            ranks[name][mk] = rank

    # Overall: sum of ranks (lower total = better overall performance)
    for name in WALKER_NAMES:
        ranks[name]["total"] = sum(ranks[name][mk] for mk in metric_keys)

    # Overall rank -- ties broken by best_dx rank (displacement as tiebreaker)
    totals = [(name, ranks[name]["total"]) for name in WALKER_NAMES]
    totals.sort(key=lambda x: (x[1], ranks[x[0]]["best_dx"]))
    for rank, (name, _) in enumerate(totals, 1):
        ranks[name]["overall"] = rank

    return ranks


# ── Zoo context ──────────────────────────────────────────────────────────────

def load_zoo_summary():
    """Load gait metrics from the v2 zoo JSON for context plotting.

    Extracts key metrics from each gait in synapse_gait_zoo_v2.json to
    overlay zoo data on competition plots (e.g., Pareto front).

    Returns:
        list of metric dicts, one per zoo gait, with keys: name, dx,
        abs_dx, speed, efficiency, phase_lock, entropy, roll_dom.
        Returns an empty list if the zoo file does not exist.
    """
    zoo_path = PROJECT / "synapse_gait_zoo_v2.json"
    if not zoo_path.exists():
        return []
    with open(zoo_path) as f:
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
                "abs_dx": abs(o.get("dx", 0)),
                "speed": o.get("mean_speed", 0),
                "efficiency": o.get("distance_per_work", 0),
                "phase_lock": coord.get("phase_lock_score", 0),
                "entropy": contact.get("contact_entropy_bits", 0),
                "roll_dom": ra.get("axis_dominance", [0, 0, 0])[0],
            })
    return gaits


# ── Plotting helpers ─────────────────────────────────────────────────────────

def clean_ax(ax):
    """Remove top and right spines from a matplotlib Axes for a cleaner look.

    Args:
        ax: matplotlib Axes instance.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    """Save a matplotlib figure to PLOT_DIR and close it to free memory.

    Args:
        fig: matplotlib Figure instance.
        name: filename (e.g., "comp_fig01_leaderboard.png").

    Side effects:
        Creates PLOT_DIR if it does not exist.
        Writes the PNG file and closes the figure.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


# ── Figure generation ────────────────────────────────────────────────────────

def plot_leaderboard(scores, ranks):
    """Fig 1: Render the competition leaderboard as a color-coded table figure.

    Cells are colored by rank: green (#1), blue (#2), red (#5), gold (overall winner).

    Args:
        scores: dict from score_walkers().
        ranks: dict from compute_ranks().

    Side effects:
        Writes comp_fig01_leaderboard.png to PLOT_DIR.
    """
    metric_labels = {
        "best_dx": "Best |DX|",
        "best_efficiency": "Best Efficiency",
        "best_speed": "Best Speed",
        "pareto_size": "Pareto Size",
        "diversity": "Diversity",
        "unique_regimes": "Unique Regimes",
    }
    metric_keys = list(metric_labels.keys())

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    # Table data
    col_labels = ["Walker"] + [metric_labels[k] for k in metric_keys] + ["Total", "Rank"]
    cell_text = []
    cell_colors = []

    # Sort by overall rank
    sorted_names = sorted(WALKER_NAMES, key=lambda n: ranks[n]["overall"])

    for name in sorted_names:
        row = [name]
        row_colors = ["white"]
        for mk in metric_keys:
            val = scores[name][mk]
            rank = ranks[name][mk]
            if isinstance(val, float):
                cell = f"{val:.4f} (#{rank})"
            else:
                cell = f"{val} (#{rank})"
            row.append(cell)
            # Color by rank
            if rank == 1:
                row_colors.append("#d4edda")  # green
            elif rank == 2:
                row_colors.append("#d1ecf1")  # blue
            elif rank == 5:
                row_colors.append("#f8d7da")  # red
            else:
                row_colors.append("white")
        row.append(str(ranks[name]["total"]))
        row_colors.append("white")
        row.append(f"#{ranks[name]['overall']}")
        if ranks[name]["overall"] == 1:
            row_colors.append("#ffd700")  # gold
        else:
            row_colors.append("white")
        cell_text.append(row)
        cell_colors.append(row_colors)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e0e0e0")

    winner = sorted_names[0]
    ax.set_title(f"Gaitspace Walker Competition — Winner: {winner}",
                 fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    save_fig(fig, "comp_fig01_leaderboard.png")


def plot_best_of_n(all_archives):
    """Fig 2: Best |DX| found so far vs evaluation count for all 5 walkers.

    Shows the cumulative maximum |DX| curve (best-of-N) to compare
    convergence speed and final performance across strategies.

    Args:
        all_archives: list of 5 archives (one per walker).

    Side effects:
        Writes comp_fig02_best_of_n.png to PLOT_DIR.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for wi, name in enumerate(WALKER_NAMES):
        archive = all_archives[wi]
        abs_dxs = [e["metrics"]["abs_dx"] for e in archive]
        best_so_far = np.maximum.accumulate(abs_dxs)
        ax.plot(np.arange(1, len(best_so_far) + 1), best_so_far,
                lw=1.8, color=WALKER_COLORS[wi], label=name, alpha=0.85)

    ax.set_xlabel("Evaluation count")
    ax.set_ylabel("Best |DX| found (meters)")
    ax.set_title("Best-of-N Curve: All Walkers")
    ax.legend(fontsize=9)
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "comp_fig02_best_of_n.png")


def plot_pareto(all_archives, zoo):
    """Fig 3: |DX| vs efficiency scatter for all walkers with zoo context.

    Overlays the global Pareto front (across all walkers) as a dashed line,
    and shows zoo gaits as gray background points for context.

    Args:
        all_archives: list of 5 archives (one per walker).
        zoo: list of zoo gait metric dicts from load_zoo_summary().

    Side effects:
        Writes comp_fig03_pareto.png to PLOT_DIR.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Zoo background
    if zoo:
        zoo_dx = [g["abs_dx"] for g in zoo]
        zoo_eff = [g["efficiency"] for g in zoo]
        ax.scatter(zoo_dx, zoo_eff, c="#CCCCCC", s=25, alpha=0.5,
                   label=f"Zoo ({len(zoo)})", zorder=1)

    # Each walker
    for wi, name in enumerate(WALKER_NAMES):
        archive = all_archives[wi]
        dxs = [e["metrics"]["abs_dx"] for e in archive]
        effs = [e["metrics"]["efficiency"] for e in archive]
        ax.scatter(dxs, effs, c=WALKER_COLORS[wi], s=12, alpha=0.4,
                   label=name, zorder=2 + wi, edgecolors="none")

    # Global Pareto front across all walkers
    all_metrics = []
    for archive in all_archives:
        all_metrics.extend([e["metrics"] for e in archive])
    pf = pareto_front(all_metrics)
    if pf:
        pf_sorted = sorted(pf, key=lambda m: m["abs_dx"])
        ax.plot([m["abs_dx"] for m in pf_sorted],
                [m["efficiency"] for m in pf_sorted],
                "k--", lw=1.5, alpha=0.6, label=f"Pareto front ({len(pf)} pts)")

    ax.set_xlabel("|DX| (meters)")
    ax.set_ylabel("Efficiency (distance/work)")
    ax.set_title("|DX| vs Efficiency: All Walkers in Zoo Context")
    ax.legend(fontsize=8, loc="upper right")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "comp_fig03_pareto.png")


def plot_diversity(all_archives):
    """Fig 4: PCA projection of behavioral space, colored by walker.

    Computes numpy-only PCA (eigh on covariance matrix) on all 5,000
    behavioral vectors and projects to 2D. Variance explained is shown
    on axis labels.

    Args:
        all_archives: list of 5 archives (one per walker).

    Side effects:
        Writes comp_fig04_diversity.png to PLOT_DIR.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect all behavioral vectors
    all_bvecs = []
    all_labels = []
    for wi, name in enumerate(WALKER_NAMES):
        archive = all_archives[wi]
        for e in archive:
            all_bvecs.append(behavioral_vec(e["metrics"]))
            all_labels.append(wi)

    all_bvecs = np.array(all_bvecs)
    all_labels = np.array(all_labels)

    # PCA (numpy-only, no sklearn): center, covariance, eigenvectors.
    # eigh returns eigenvalues in ascending order; reverse for descending variance.
    mean = all_bvecs.mean(axis=0)
    centered = all_bvecs - mean
    cov = np.dot(centered.T, centered) / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending so PC1 captures the most variance
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    projected = centered @ eigvecs[:, :2]

    for wi, name in enumerate(WALKER_NAMES):
        mask = all_labels == wi
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   c=WALKER_COLORS[wi], s=10, alpha=0.4, label=name,
                   edgecolors="none")

    var_explained = eigvals[idx[:2]] / eigvals.sum() * 100
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)")
    ax.set_title("Behavioral Space PCA: All Walkers")
    ax.legend(fontsize=9)
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "comp_fig04_diversity.png")


def plot_trajectories(all_weight_histories):
    """Fig 5: Weight-space PCA trajectory for each walker (5 subplots).

    Computes a shared PCA across all weight vectors from all walkers,
    then plots each walker's trajectory in 2D weight-PC space. Points
    are colored by evaluation order (viridis), with start/end markers.

    Args:
        all_weight_histories: list of 5 lists of numpy weight vectors.

    Side effects:
        Writes comp_fig05_trajectories.png to PLOT_DIR.
    """
    # Collect all weight vectors for shared PCA
    all_vecs = []
    for wh in all_weight_histories:
        all_vecs.extend(wh)
    all_vecs = np.array(all_vecs)

    mean = all_vecs.mean(axis=0)
    centered = all_vecs - mean
    cov = np.dot(centered.T, centered) / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]

    fig, axes = plt.subplots(1, 5, figsize=(19, 4))

    offset = 0
    for wi, name in enumerate(WALKER_NAMES):
        ax = axes[wi]
        wh = np.array(all_weight_histories[wi])
        proj = (wh - mean) @ eigvecs[:, :2]

        # Plot trajectory as connected line
        ax.plot(proj[:, 0], proj[:, 1], lw=0.3, alpha=0.3,
                color=WALKER_COLORS[wi])
        ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)),
                   cmap="viridis", s=3, alpha=0.5, edgecolors="none")
        ax.scatter(proj[0, 0], proj[0, 1], c="green", s=40, marker="^",
                   zorder=5, label="start")
        ax.scatter(proj[-1, 0], proj[-1, 1], c="red", s=40, marker="s",
                   zorder=5, label="end")

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("WPC1")
        if wi == 0:
            ax.set_ylabel("WPC2")
        ax.legend(fontsize=7, loc="upper right")
        clean_ax(ax)

    fig.suptitle("Weight-Space PCA Trajectories", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "comp_fig05_trajectories.png")


def plot_radar(scores):
    """Fig 6: Radar chart with 6 normalized metrics per walker.

    Each metric is min-max normalized to [0, 1] across walkers so all
    axes are comparable. The radar polygon area gives a visual sense
    of overall performance breadth.

    Args:
        scores: dict from score_walkers().

    Side effects:
        Writes comp_fig06_radar.png to PLOT_DIR.
    """
    metric_keys = ["best_dx", "best_efficiency", "best_speed",
                   "pareto_size", "diversity", "unique_regimes"]
    metric_labels = ["Best |DX|", "Best Eff.", "Best Speed",
                     "Pareto Size", "Diversity", "Unique Regimes"]

    # Normalize each metric to [0, 1] across walkers so all axes are comparable
    norm_scores = {}
    for mk in metric_keys:
        vals = [scores[name][mk] for name in WALKER_NAMES]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax - vmin > 1e-12 else 1.0
        norm_scores[mk] = {name: (scores[name][mk] - vmin) / rng for name in WALKER_NAMES}

    n_metrics = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # duplicate first angle to close the radar polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for wi, name in enumerate(WALKER_NAMES):
        values = [norm_scores[mk][name] for mk in metric_keys]
        values += values[:1]
        ax.plot(angles, values, "o-", lw=2, color=WALKER_COLORS[wi],
                label=name, alpha=0.7, markersize=5)
        ax.fill(angles, values, color=WALKER_COLORS[wi], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Walker Performance Radar", fontsize=13, pad=20)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    save_fig(fig, "comp_fig06_radar.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run all 5 walkers, score them, save JSON results, and generate comparison figures.

    Pipeline:
        1. Back up brain.nndf (restored at end).
        2. Run all 5 walker algorithms sequentially (1,000 evals each).
        3. Compute scores and ranks across 6 performance dimensions.
        4. Print leaderboard to console.
        5. Save structured results to artifacts/walker_competition.json.
        6. Generate 6 comparison figures to artifacts/plots/.

    Side effects:
        - Overwrites and restores brain.nndf.
        - Writes JSON and PNG artifacts to artifacts/.
        - Prints progress and leaderboard to stdout.
    """
    # Backup brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    print("=" * 70)
    print("GAITSPACE WALKER COMPETITION")
    print(f"Budget: {EVAL_BUDGET} evaluations per walker")
    print(f"Walkers: {len(WALKER_NAMES)}")
    print("=" * 70)

    t_total = time.perf_counter()
    all_archives = []
    all_weight_histories = []

    # Run all 5 walkers
    runners = [
        run_hill_climber,
        run_ridge_walker,
        run_cliff_mapper,
        run_novelty_seeker,
        run_ensemble_explorer,
    ]

    for runner in runners:
        t_w = time.perf_counter()
        archive, weight_history = runner()
        elapsed = time.perf_counter() - t_w
        print(f"    Time: {elapsed:.1f}s ({elapsed/len(archive):.3f}s/eval)")
        all_archives.append(archive)
        all_weight_histories.append(weight_history)

    total_elapsed = time.perf_counter() - t_total
    print(f"\nAll walkers done in {total_elapsed:.1f}s")

    # Restore brain.nndf
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # ── Scoring ──────────────────────────────────────────────────────────────

    print("\nScoring...")
    scores = score_walkers(all_archives)
    ranks = compute_ranks(scores)

    # Print leaderboard
    print("\n" + "=" * 70)
    print("LEADERBOARD")
    print("=" * 70)
    metric_labels = {
        "best_dx": "Best |DX|",
        "best_efficiency": "Best Eff.",
        "best_speed": "Best Speed",
        "pareto_size": "Pareto",
        "diversity": "Diversity",
        "unique_regimes": "Regimes",
    }
    metric_keys = list(metric_labels.keys())

    header = f"{'Walker':<22}"
    for mk in metric_keys:
        header += f" {metric_labels[mk]:>12}"
    header += f" {'Total':>6} {'Rank':>5}"
    print(header)
    print("-" * len(header))

    sorted_names = sorted(WALKER_NAMES, key=lambda n: ranks[n]["overall"])
    for name in sorted_names:
        row = f"{name:<22}"
        for mk in metric_keys:
            val = scores[name][mk]
            rk = ranks[name][mk]
            if isinstance(val, float):
                row += f" {val:>8.3f}(#{rk})"
            else:
                row += f" {val:>8d}(#{rk})"
        row += f" {ranks[name]['total']:>6d}"
        row += f"   #{ranks[name]['overall']}"
        print(row)

    winner = sorted_names[0]
    print(f"\nWINNER: {winner}")

    # ── Save JSON ────────────────────────────────────────────────────────────

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "eval_budget": EVAL_BUDGET,
            "n_walkers": len(WALKER_NAMES),
            "total_time_s": total_elapsed,
            "winner": winner,
        },
        "walkers": {},
    }

    for wi, name in enumerate(WALKER_NAMES):
        archive = all_archives[wi]
        output["walkers"][name] = {
            "n_evals": len(archive),
            "scores": scores[name],
            "ranks": ranks[name],
            "best_trial": max(archive, key=lambda e: e["metrics"]["abs_dx"]),
        }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    # ── Figures ──────────────────────────────────────────────────────────────

    print("\nGenerating figures...")
    zoo = load_zoo_summary()

    plot_leaderboard(scores, ranks)
    plot_best_of_n(all_archives)
    plot_pareto(all_archives, zoo)
    plot_diversity(all_archives)
    plot_trajectories(all_weight_histories)
    plot_radar(scores)

    print(f"\nDone. {total_elapsed:.1f}s total.")


if __name__ == "__main__":
    main()
