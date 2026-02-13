#!/usr/bin/env python3
"""
walker_competition.py

Head-to-head competition of 5 gaitspace walker algorithms, each with a
1,000-evaluation budget on the 6D synapse weight space.

Walkers:
  1. Hill Climber        — accept perturbation if |DX| improves
  2. Ridge Walker        — Pareto-non-dominated moves on (|DX|, efficiency)
  3. Cliff Mapper        — probe 10 directions, walk toward steepest cliff
  4. Novelty Seeker      — pick most behaviorally novel candidate from 5
  5. Ensemble Explorer   — 20 parallel hill climbers with teleportation

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
    """Run one headless simulation → compact metrics dict."""
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
    """Random weight dict, each in [-1, 1]."""
    return {wn: random.uniform(-1, 1) for wn in WEIGHT_NAMES}


def weights_to_vec(weights):
    """Convert weight dict to numpy array."""
    return np.array([weights[wn] for wn in WEIGHT_NAMES])


def vec_to_weights(vec):
    """Convert numpy array to weight dict."""
    return {wn: float(vec[i]) for i, wn in enumerate(WEIGHT_NAMES)}


def perturb(weights, radius):
    """Perturb weights along a random 6D unit vector at given radius."""
    direction = np.random.randn(N_WEIGHTS)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        direction = np.ones(N_WEIGHTS)
        norm = np.linalg.norm(direction)
    direction = direction / norm

    vec = weights_to_vec(weights)
    new_vec = vec + radius * direction
    return vec_to_weights(new_vec)


def behavioral_vec(metrics):
    """6D behavioral descriptor from metrics dict (raw, unnormalized)."""
    return np.array([
        metrics["abs_dx"],
        metrics["speed"],
        metrics["efficiency"],
        metrics["phase_lock"],
        metrics["entropy"],
        metrics["roll_dom"],
    ])


def normalize_behavioral_vecs(vecs):
    """Normalize behavioral vectors to [0, 1] per dimension."""
    arr = np.array(vecs)
    if len(arr) == 0:
        return arr
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0
    return (arr - mins) / ranges


def behavioral_distance(bv1, bv2):
    """Euclidean distance between two behavioral vectors."""
    return np.linalg.norm(np.array(bv1) - np.array(bv2))


def novelty(bvec, archive_bvecs, k=15):
    """Mean distance to k nearest neighbors in archive."""
    if len(archive_bvecs) == 0:
        return float('inf')
    dists = [behavioral_distance(bvec, a) for a in archive_bvecs]
    dists.sort()
    k_actual = min(k, len(dists))
    return np.mean(dists[:k_actual])


def is_dominated(a_metrics, b_metrics):
    """True if b dominates a on (|DX|, efficiency) — b >= a on both, b > a on at least one."""
    a_dx, a_eff = a_metrics["abs_dx"], a_metrics["efficiency"]
    b_dx, b_eff = b_metrics["abs_dx"], b_metrics["efficiency"]
    return (b_dx >= a_dx and b_eff >= a_eff) and (b_dx > a_dx or b_eff > a_eff)


def pareto_front(archive):
    """Return list of non-dominated points from archive (list of metrics dicts)."""
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
    """Walker 1: Simple hill climber maximizing |DX|. 1 eval/step, 999 steps."""
    print("\n[1/5] Hill Climber (999 steps)...")
    archive = []
    weight_history = []

    # Initial random point
    w = random_weights()
    m = evaluate(w)
    archive.append({"weights": w, "metrics": m})
    weight_history.append(weights_to_vec(w))

    for step in range(999):
        w_new = perturb(w, radius=0.1)
        m_new = evaluate(w_new)
        archive.append({"weights": w_new, "metrics": m_new})
        weight_history.append(weights_to_vec(w_new))

        if m_new["abs_dx"] > m["abs_dx"]:
            w = w_new
            m = m_new

        if (step + 1) % 200 == 0:
            print(f"    step {step+1}/999  best |DX|={m['abs_dx']:.3f}  "
                  f"evals={len(archive)}")

    print(f"    Done. {len(archive)} evals, best |DX|={m['abs_dx']:.3f}")
    return archive, weight_history


def run_ridge_walker():
    """Walker 2: Pareto-non-dominated moves on (|DX|, efficiency). 3 evals/step, 333 steps."""
    print("\n[2/5] Ridge Walker (333 steps, 3 candidates/step)...")
    archive = []
    weight_history = []

    w = random_weights()
    m = evaluate(w)
    archive.append({"weights": w, "metrics": m})
    weight_history.append(weights_to_vec(w))

    for step in range(333):
        candidates = []
        for _ in range(3):
            w_c = perturb(w, radius=0.1)
            m_c = evaluate(w_c)
            archive.append({"weights": w_c, "metrics": m_c})
            weight_history.append(weights_to_vec(w_c))
            candidates.append((w_c, m_c))

        # Filter to non-dominated candidates (not dominated by current point)
        non_dominated = [(wc, mc) for wc, mc in candidates if not is_dominated(mc, m)]

        if non_dominated:
            # Pick the one farthest from current in objective space
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
    """Walker 3: Probe 10 directions, walk toward steepest cliff. 10 evals/step, 99 steps."""
    print("\n[3/5] Cliff Mapper (99 steps, 10 probes/step)...")
    archive = []
    weight_history = []

    w = random_weights()
    m = evaluate(w)
    archive.append({"weights": w, "metrics": m})
    weight_history.append(weights_to_vec(w))

    for step in range(99):
        probes = []
        for _ in range(10):
            w_p = perturb(w, radius=0.05)
            m_p = evaluate(w_p)
            archive.append({"weights": w_p, "metrics": m_p})
            weight_history.append(weights_to_vec(w_p))
            delta_dx = abs(m_p["abs_dx"] - m["abs_dx"])
            probes.append((w_p, m_p, delta_dx))

        # Move to probe with largest |delta_DX| — seek disruption
        best_probe = max(probes, key=lambda x: x[2])
        w, m = best_probe[0], best_probe[1]

        if (step + 1) % 25 == 0:
            print(f"    step {step+1}/99  |DX|={m['abs_dx']:.3f}  "
                  f"max delta={best_probe[2]:.3f}  evals={len(archive)}")

    # Remaining budget: 1000 - 1 - 99*10 = 9
    remaining = EVAL_BUDGET - len(archive)
    for _ in range(remaining):
        w_r = perturb(w, radius=0.1)
        m_r = evaluate(w_r)
        archive.append({"weights": w_r, "metrics": m_r})
        weight_history.append(weights_to_vec(w_r))

    print(f"    Done. {len(archive)} evals, final |DX|={m['abs_dx']:.3f}")
    return archive, weight_history


def run_novelty_seeker():
    """Walker 4: Pick most behaviorally novel candidate from 5. 5 evals/step, 199 steps."""
    print("\n[4/5] Novelty Seeker (199 steps, 5 candidates/step)...")
    archive = []
    weight_history = []
    behavior_archive = []  # list of behavioral vectors for novelty computation

    w = random_weights()
    m = evaluate(w)
    archive.append({"weights": w, "metrics": m})
    weight_history.append(weights_to_vec(w))
    behavior_archive.append(behavioral_vec(m))

    for step in range(199):
        candidates = []
        for _ in range(5):
            w_c = perturb(w, radius=0.2)
            m_c = evaluate(w_c)
            archive.append({"weights": w_c, "metrics": m_c})
            weight_history.append(weights_to_vec(w_c))
            bv = behavioral_vec(m_c)
            candidates.append((w_c, m_c, bv))

        # Normalize archive + candidates together for fair distance computation
        all_bvecs = behavior_archive + [c[2] for c in candidates]
        normalized = normalize_behavioral_vecs(all_bvecs)
        n_archive = len(behavior_archive)
        norm_archive = normalized[:n_archive]

        # Pick most novel candidate
        best_nov = -1
        best_idx = 0
        for ci, (wc, mc, bv) in enumerate(candidates):
            norm_bv = normalized[n_archive + ci]
            nov = novelty(norm_bv, norm_archive, k=15)
            if nov > best_nov:
                best_nov = nov
                best_idx = ci

        w, m = candidates[best_idx][0], candidates[best_idx][1]
        # Add all candidates to behavior archive
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
    """Walker 5: 20 parallel hill climbers with teleportation. 1 eval/walker/step, 49 steps."""
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
        # Each walker does one hill climbing step
        for wi in range(n_walkers):
            w_new = perturb(walkers_w[wi], radius=0.1)
            m_new = evaluate(w_new)
            archive.append({"weights": w_new, "metrics": m_new})
            weight_history.append(weights_to_vec(w_new))

            if m_new["abs_dx"] > walkers_m[wi]["abs_dx"]:
                walkers_w[wi] = w_new
                walkers_m[wi] = m_new

        # Every 10 steps: teleportation — deduplicate crowded walkers
        if (step + 1) % 10 == 0:
            # Compute behavioral vectors for all walkers
            bvecs = [behavioral_vec(walkers_m[wi]) for wi in range(n_walkers)]
            norm_bvecs = normalize_behavioral_vecs(bvecs)

            # Find pairs within threshold
            for wi in range(n_walkers):
                for wj in range(wi + 1, n_walkers):
                    dist = np.linalg.norm(norm_bvecs[wi] - norm_bvecs[wj])
                    if dist < 0.3:
                        # Teleport the worse one
                        worse = wi if walkers_m[wi]["abs_dx"] < walkers_m[wj]["abs_dx"] else wj
                        walkers_w[worse] = random_weights()
                        # Don't evaluate here — it'll be evaluated on next step
                        # Reset metrics to trigger re-evaluation
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
    """Compute 6 competition metrics for each walker → leaderboard."""
    scores = {}

    for wi, name in enumerate(WALKER_NAMES):
        archive = all_archives[wi]
        metrics_list = [e["metrics"] for e in archive]

        # Best |DX|
        best_dx = max(m["abs_dx"] for m in metrics_list)

        # Best efficiency (among evals with |DX| > 2m)
        efficient = [m["efficiency"] for m in metrics_list if m["abs_dx"] > 2.0]
        best_eff = max(efficient) if efficient else 0.0

        # Best speed
        best_speed = max(m["speed"] for m in metrics_list)

        # Pareto front size on (|DX|, efficiency)
        pf = pareto_front(metrics_list)
        pareto_size = len(pf)

        # Diversity: mean pairwise behavioral distance (sample 200 pairs)
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
    """Simple k-means clustering, return count of non-empty clusters. Numpy-only."""
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

        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    # Count non-empty clusters
    unique_labels = len(set(labels.tolist()))
    return unique_labels


def compute_ranks(scores):
    """Rank walkers 1-5 on each metric. Lower rank = better. Return ranks dict + overall."""
    metric_keys = ["best_dx", "best_efficiency", "best_speed",
                   "pareto_size", "diversity", "unique_regimes"]
    # All metrics: higher is better
    ranks = {name: {} for name in WALKER_NAMES}

    for mk in metric_keys:
        vals = [(name, scores[name][mk]) for name in WALKER_NAMES]
        vals.sort(key=lambda x: x[1], reverse=True)  # highest first
        for rank, (name, _) in enumerate(vals, 1):
            ranks[name][mk] = rank

    # Overall: sum of ranks
    for name in WALKER_NAMES:
        ranks[name]["total"] = sum(ranks[name][mk] for mk in metric_keys)

    # Overall rank
    totals = [(name, ranks[name]["total"]) for name in WALKER_NAMES]
    totals.sort(key=lambda x: (x[1], ranks[x[0]]["best_dx"]))  # tie-break on best_dx rank
    for rank, (name, _) in enumerate(totals, 1):
        ranks[name]["overall"] = rank

    return ranks


# ── Zoo context ──────────────────────────────────────────────────────────────

def load_zoo_summary():
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


# ── Figure generation ────────────────────────────────────────────────────────

def plot_leaderboard(scores, ranks):
    """Fig 1: Leaderboard table as figure."""
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
    """Fig 2: Best |DX| vs evaluation count, all 5 walkers overlaid."""
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
    """Fig 3: |DX| vs efficiency scatter, all walkers + zoo context."""
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
    """Fig 4: PCA of behavioral space, all archived points by walker."""
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

    # PCA (numpy-only): center, covariance, eigenvectors
    mean = all_bvecs.mean(axis=0)
    centered = all_bvecs - mean
    cov = np.dot(centered.T, centered) / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
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
    """Fig 5: Weight-space PCA trajectory for each walker (5 subplots)."""
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
    """Fig 6: Radar chart with 6 normalized metrics per walker."""
    metric_keys = ["best_dx", "best_efficiency", "best_speed",
                   "pareto_size", "diversity", "unique_regimes"]
    metric_labels = ["Best |DX|", "Best Eff.", "Best Speed",
                     "Pareto Size", "Diversity", "Unique Regimes"]

    # Normalize each metric to [0, 1] across walkers
    norm_scores = {}
    for mk in metric_keys:
        vals = [scores[name][mk] for name in WALKER_NAMES]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax - vmin > 1e-12 else 1.0
        norm_scores[mk] = {name: (scores[name][mk] - vmin) / rng for name in WALKER_NAMES}

    n_metrics = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

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
