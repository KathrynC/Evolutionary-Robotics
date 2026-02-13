#!/usr/bin/env python3
"""
llm_seeded_evolution.py — Part B: LLM-Seeded Evolution

Tests whether starting evolution from LLM-generated weights (the smooth
submanifold) is a launchpad or a trap.

Runs Hill Climber with 500 evaluations from 4 starting conditions:
1. Revelation weights (best LLM gait: DX=29.17m)
2. Walk cluster weights (most common LLM output: DX=8.32m)
3. Stagger cluster weights (DX=4.39m)
4. Random initialization (baseline, 5 independent runs)

Each run tracks: fitness trajectory, weight trajectory, behavioral diversity.

Output: artifacts/llm_seeded_evolution_results.json
        artifacts/plots/le_fig01_fitness_trajectories.png
        artifacts/plots/le_fig02_final_comparison.png
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
import constants as c
from structured_random_common import compute_all, DT

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]
N_WEIGHTS = 6
EVAL_BUDGET = 500
N_RANDOM_RUNS = 5  # independent random baselines

OUT_PATH = PROJECT / "artifacts" / "llm_seeded_evolution_results.json"
PLOT_DIR = PROJECT / "artifacts" / "plots"

# LLM-generated seed weights (from structured random experiments)
LLM_SEEDS = {
    "Revelation": {
        "desc": "Best LLM gait (DX=29.17m, asymmetric, high displacement)",
        "weights": {"w03": -0.8, "w04": 0.6, "w13": 0.2, "w14": -0.9,
                    "w23": 0.5, "w24": -0.4},
    },
    "Walk_cluster": {
        "desc": "Most common LLM output (39 seeds → identical weights, DX=8.32m)",
        "weights": {"w03": 0.6, "w04": -0.4, "w13": 0.2, "w14": -0.8,
                    "w23": 0.5, "w24": -0.3},
    },
    "Stagger_cluster": {
        "desc": "Third most common (17 seeds, DX=4.39m)",
        "weights": {"w03": 0.6, "w04": -0.4, "w13": -0.2, "w14": 0.8,
                    "w23": 0.3, "w24": -0.5},
    },
    "Ecclesiastes": {
        "desc": "Most efficient LLM gait (eff=0.00495)",
        "weights": {"w03": 0.6, "w04": -0.5, "w13": -0.4, "w14": 0.8,
                    "w23": 0.2, "w24": -0.9},
    },
}


# ── Simulation ───────────────────────────────────────────────────────────────

def write_brain(weights):
    """Write brain.nndf with standard 6-synapse topology."""
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


def evaluate(weights):
    """Run headless sim and return metrics dict."""
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

    t_arr = np.empty(n_steps)
    x = np.empty(n_steps); y = np.empty(n_steps); z = np.empty(n_steps)
    vx = np.empty(n_steps); vy = np.empty(n_steps); vz = np.empty(n_steps)
    wx = np.empty(n_steps); wy = np.empty(n_steps); wz = np.empty(n_steps)
    roll = np.empty(n_steps); pitch = np.empty(n_steps); yaw = np.empty(n_steps)
    contact_torso = np.empty(n_steps, dtype=bool)
    contact_back = np.empty(n_steps, dtype=bool)
    contact_front = np.empty(n_steps, dtype=bool)
    j0_pos = np.empty(n_steps); j0_vel = np.empty(n_steps); j0_tau = np.empty(n_steps)
    j1_pos = np.empty(n_steps); j1_vel = np.empty(n_steps); j1_tau = np.empty(n_steps)

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

    back_link_idx = link_indices.get("BackLeg", -1)
    front_link_idx = link_indices.get("FrontLeg", -1)
    j0_idx = joint_indices.get("Torso_BackLeg", 0)
    j1_idx = joint_indices.get("Torso_FrontLeg", 1)

    for i in range(n_steps):
        for neuronName in nn.neurons:
            n_obj = nn.neurons[neuronName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes,
                                                n_obj.Get_Value(), max_force)
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

        contact_pts = p.getContactPoints(robotId)
        torso_contact = False; back_contact = False; front_contact = False
        for cp in contact_pts:
            li = cp[3]
            if li == -1:
                torso_contact = True
            elif li == back_link_idx:
                back_contact = True
            elif li == front_link_idx:
                front_contact = True
        contact_torso[i] = torso_contact
        contact_back[i] = back_contact
        contact_front[i] = front_contact

        js0 = p.getJointState(robotId, j0_idx)
        js1 = p.getJointState(robotId, j1_idx)
        j0_pos[i] = js0[0]; j0_vel[i] = js0[1]; j0_tau[i] = js0[3]
        j1_pos[i] = js1[0]; j1_vel[i] = js1[1]; j1_tau[i] = js1[3]

    p.disconnect()

    data = {
        "t": t_arr, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "contact_torso": contact_torso,
        "contact_back": contact_back,
        "contact_front": contact_front,
        "j0_pos": j0_pos, "j0_vel": j0_vel, "j0_tau": j0_tau,
        "j1_pos": j1_pos, "j1_vel": j1_vel, "j1_tau": j1_tau,
    }
    analytics = compute_all(data, DT)

    return {
        "dx": float(analytics["outcome"]["dx"]),
        "abs_dx": float(abs(analytics["outcome"]["dx"])),
        "speed": float(analytics["outcome"]["mean_speed"]),
        "efficiency": float(analytics["outcome"]["distance_per_work"]),
        "work_proxy": float(analytics["outcome"]["work_proxy"]),
        "phase_lock": float(analytics["coordination"]["phase_lock_score"]),
        "entropy": float(analytics["contact"]["contact_entropy_bits"]),
        "roll_dom": float(analytics["rotation_axis"]["axis_dominance"][0]),
    }


# ── Evolution ────────────────────────────────────────────────────────────────

def weights_to_vec(weights):
    return np.array([weights[wn] for wn in WEIGHT_NAMES])


def vec_to_weights(vec):
    return {wn: float(vec[i]) for i, wn in enumerate(WEIGHT_NAMES)}


def perturb(weights, radius):
    direction = np.random.randn(N_WEIGHTS)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        direction = np.ones(N_WEIGHTS)
        norm = np.linalg.norm(direction)
    direction /= norm
    vec = weights_to_vec(weights)
    return vec_to_weights(vec + radius * direction)


def run_hill_climber(seed_weights, budget=EVAL_BUDGET, radius=0.1):
    """Hill climber maximizing abs_dx from a given starting point."""
    w = dict(seed_weights)
    m = evaluate(w)
    best_fitness = m["abs_dx"]
    evals = 1

    fitness_history = [best_fitness]
    best_history = [best_fitness]
    weight_history = [weights_to_vec(w).tolist()]

    while evals < budget:
        candidate = perturb(w, radius)
        m_c = evaluate(candidate)
        evals += 1

        fitness_history.append(m_c["abs_dx"])
        if m_c["abs_dx"] > best_fitness:
            w = candidate
            m = m_c
            best_fitness = m_c["abs_dx"]
        best_history.append(best_fitness)
        weight_history.append(weights_to_vec(w).tolist())

    return {
        "best_fitness": float(best_fitness),
        "best_weights": w,
        "best_metrics": m,
        "fitness_history": fitness_history,
        "best_history": best_history,
        "n_evals": evals,
        "final_distance_from_start": float(np.linalg.norm(
            weights_to_vec(w) - weights_to_vec(seed_weights))),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LLM-SEEDED EVOLUTION EXPERIMENT")
    print(f"  {EVAL_BUDGET} evaluations per run, radius=0.1")
    print(f"  {len(LLM_SEEDS)} LLM seeds + {N_RANDOM_RUNS} random baselines")
    print("=" * 60)

    # Backup brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.bak"
    if brain_path.exists():
        import shutil
        shutil.copy2(brain_path, backup_path)

    results = {"runs": {}, "meta": {}}
    results["meta"] = {
        "eval_budget": EVAL_BUDGET,
        "radius": 0.1,
        "n_random_baselines": N_RANDOM_RUNS,
        "llm_seeds": {k: v["desc"] for k, v in LLM_SEEDS.items()},
    }

    total_evals = 0
    t0 = time.time()

    # Run LLM-seeded trials
    for name, seed_info in LLM_SEEDS.items():
        print(f"\n  [{name}] {seed_info['desc'][:50]}...")
        t1 = time.time()
        run = run_hill_climber(seed_info["weights"])
        dt = time.time() - t1
        total_evals += run["n_evals"]

        print(f"    Start: {abs(seed_info['weights']['w03']):.1f}... -> "
              f"DX={run['fitness_history'][0]:.2f}m")
        print(f"    Best:  DX={run['best_fitness']:.2f}m "
              f"(after {run['n_evals']} evals, {dt:.1f}s)")
        print(f"    Distance from start: {run['final_distance_from_start']:.3f}")

        results["runs"][name] = {
            "type": "llm",
            "seed_weights": seed_info["weights"],
            "seed_desc": seed_info["desc"],
            "best_fitness": run["best_fitness"],
            "best_weights": run["best_weights"],
            "best_metrics": run["best_metrics"],
            "best_history": run["best_history"],
            "n_evals": run["n_evals"],
            "time_s": dt,
            "final_distance_from_start": run["final_distance_from_start"],
        }

        # Save incrementally
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Run random baselines
    for i in range(N_RANDOM_RUNS):
        random_seed = {wn: random.uniform(-1, 1) for wn in WEIGHT_NAMES}
        name = f"Random_{i}"
        print(f"\n  [{name}] Random initialization...")
        t1 = time.time()
        run = run_hill_climber(random_seed)
        dt = time.time() - t1
        total_evals += run["n_evals"]

        print(f"    Start: DX={run['fitness_history'][0]:.2f}m")
        print(f"    Best:  DX={run['best_fitness']:.2f}m "
              f"(after {run['n_evals']} evals, {dt:.1f}s)")
        print(f"    Distance from start: {run['final_distance_from_start']:.3f}")

        results["runs"][name] = {
            "type": "random",
            "seed_weights": {k: float(v) for k, v in random_seed.items()},
            "best_fitness": run["best_fitness"],
            "best_weights": run["best_weights"],
            "best_metrics": run["best_metrics"],
            "best_history": run["best_history"],
            "n_evals": run["n_evals"],
            "time_s": dt,
            "final_distance_from_start": run["final_distance_from_start"],
        }

        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - t0
    results["meta"]["total_evals"] = total_evals
    results["meta"]["total_time_s"] = total_time

    # Restore brain.nndf
    if backup_path.exists():
        import shutil
        shutil.copy2(backup_path, brain_path)

    # ── Analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    llm_runs = {k: v for k, v in results["runs"].items() if v["type"] == "llm"}
    random_runs = {k: v for k, v in results["runs"].items() if v["type"] == "random"}

    llm_best = max(v["best_fitness"] for v in llm_runs.values())
    random_best = max(v["best_fitness"] for v in random_runs.values())
    random_mean = np.mean([v["best_fitness"] for v in random_runs.values()])
    random_std = np.std([v["best_fitness"] for v in random_runs.values()])

    print(f"\n  LLM-seeded best: {llm_best:.2f}m")
    print(f"  Random best: {random_best:.2f}m")
    print(f"  Random mean±std: {random_mean:.2f}±{random_std:.2f}m")

    # At what eval did each run reach 10m?
    print(f"\n  Evals to reach 10m (or never):")
    for name, run in results["runs"].items():
        hist = run["best_history"]
        reached = None
        for i, f in enumerate(hist):
            if f >= 10.0:
                reached = i
                break
        if reached is not None:
            print(f"    {name}: eval {reached}")
        else:
            print(f"    {name}: never (best={run['best_fitness']:.2f}m)")

    results["analysis"] = {
        "llm_best": llm_best,
        "random_best": random_best,
        "random_mean": float(random_mean),
        "random_std": float(random_std),
    }

    # ── Figures ──────────────────────────────────────────────────────
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Fig 1: Fitness trajectories
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_llm = {"Revelation": "#e41a1c", "Walk_cluster": "#377eb8",
                  "Stagger_cluster": "#4daf4a", "Ecclesiastes": "#984ea3"}
    for name, run in llm_runs.items():
        ax.plot(run["best_history"], label=f"{name} (start={run['best_history'][0]:.1f}m)",
                color=colors_llm.get(name, "black"), linewidth=2)
    for name, run in random_runs.items():
        ax.plot(run["best_history"], label=f"{name}",
                color="#aaaaaa", alpha=0.5, linewidth=1)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best abs(DX) [m]")
    ax.set_title(f"LLM-Seeded vs Random Evolution ({EVAL_BUDGET} evals)")
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(y=29.17, color="red", ls=":", alpha=0.3, label="Revelation initial")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "le_fig01_fitness_trajectories.png", dpi=300)
    plt.close(fig)
    print(f"\n  WROTE le_fig01_fitness_trajectories.png")

    # Fig 2: Final comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results["runs"].keys())
    fitnesses = [results["runs"][n]["best_fitness"] for n in names]
    colors = [colors_llm.get(n, "#aaaaaa") for n in names]
    bars = ax.bar(range(len(names)), fitnesses, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Best abs(DX) [m]")
    ax.set_title("Final Fitness: LLM-Seeded vs Random")
    for bar, f in zip(bars, fitnesses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{f:.1f}m", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "le_fig02_final_comparison.png", dpi=300)
    plt.close(fig)
    print(f"  WROTE le_fig02_final_comparison.png")

    # Final save
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  WROTE {OUT_PATH}")
    print(f"  Total: {total_evals} evaluations in {total_time:.0f}s "
          f"({total_evals/total_time:.1f} evals/s)")


if __name__ == "__main__":
    main()
