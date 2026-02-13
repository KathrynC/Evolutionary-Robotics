#!/usr/bin/env python3
"""
perturbation_probing.py — Phase 7 / A3: Direct Cliffiness Measurement

Directly measures cliffiness at all unique LLM-generated weight vectors
using the 6-direction perturbation protocol from atlas_cliffiness.py.
Compares directly measured cliffiness to atlas-interpolated values.

Saves partial results after each weight vector (interruptible).

Output: artifacts/perturbation_probing_results.json
        artifacts/plots/pp_fig01_measured_vs_interpolated.png
        artifacts/plots/pp_fig02_llm_vs_atlas_cliffiness.png
"""

import json
import sys
import os
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]
R_PROBE = 0.05
N_DIRECTIONS = 6
OUT_PATH = PROJECT / "artifacts" / "perturbation_probing_results.json"


# ── Simulation ───────────────────────────────────────────────────────────────

def write_brain_standard(weights):
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


def simulate_dx_only(weights):
    """Run headless sim and return x-displacement."""
    import pybullet as p
    import pybullet_data
    import pyrosim.pyrosim as pyrosim
    from pyrosim.neuralNetwork import NEURAL_NETWORK
    import constants as c

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


# ── Perturbation helpers ─────────────────────────────────────────────────────

def random_direction_6d():
    """Random unit vector in 6D."""
    v = np.random.randn(6)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        v = np.ones(6)
        norm = np.linalg.norm(v)
    return v / norm


def perturb_weights(base_weights, direction, radius):
    """Create perturbed weight dict."""
    w = {}
    for i, wn in enumerate(WEIGHT_NAMES):
        w[wn] = base_weights[wn] + radius * direction[i]
    return w


def probe_cliffiness(weights):
    """Probe 6 random directions and compute cliffiness + gradient."""
    base_dx = simulate_dx_only(weights)
    directions = np.array([random_direction_6d() for _ in range(N_DIRECTIONS)])
    delta_dxs = np.empty(N_DIRECTIONS)

    for k in range(N_DIRECTIONS):
        pw = perturb_weights(weights, directions[k], R_PROBE)
        delta_dxs[k] = simulate_dx_only(pw) - base_dx

    try:
        grad = np.linalg.solve(directions, delta_dxs / R_PROBE)
    except np.linalg.LinAlgError:
        grad = np.zeros(6)

    return {
        "base_dx": float(base_dx),
        "cliffiness": float(np.max(np.abs(delta_dxs))),
        "gradient_magnitude": float(np.linalg.norm(grad)),
        "gradient_vector": grad.tolist(),
        "delta_dxs": delta_dxs.tolist(),
        "directions": directions.tolist(),
    }


# ── Data loading ─────────────────────────────────────────────────────────────

def load_unique_llm_weights():
    """Load all unique LLM-generated weight vectors (not baseline)."""
    conditions = ["verbs", "theorems", "bible", "places"]
    unique = {}  # key: tuple of weight values, value: dict with metadata

    for cond in conditions:
        path = PROJECT / "artifacts" / f"structured_random_{cond}.json"
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        with open(path) as f:
            trials = json.load(f)

        for trial in trials:
            wt = trial["weights"]
            key = tuple(wt[k] for k in WEIGHT_NAMES)
            if key not in unique:
                unique[key] = {
                    "weights": {k: float(wt[k]) for k in WEIGHT_NAMES},
                    "conditions": [cond],
                    "seeds": [trial["seed"]],
                    "known_dx": float(trial["dx"]),
                }
            else:
                if cond not in unique[key]["conditions"]:
                    unique[key]["conditions"].append(cond)
                if trial["seed"] not in unique[key]["seeds"]:
                    unique[key]["seeds"].append(trial["seed"])

    return list(unique.values())


def load_interpolated_cliffiness():
    """Load atlas-interpolated cliffiness from categorical_structure_results.json."""
    cs_path = PROJECT / "artifacts" / "categorical_structure_results.json"
    if not cs_path.exists():
        return {}
    with open(cs_path) as f:
        cs = json.load(f)
    return cs.get("map_G", {})


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PERTURBATION PROBING — Direct Cliffiness at LLM Points")
    print(f"  r_probe = {R_PROBE}, {N_DIRECTIONS} directions per point")
    print("=" * 60)

    unique_weights = load_unique_llm_weights()
    print(f"  Found {len(unique_weights)} unique LLM weight vectors")

    # Load partial results if resuming
    results = {"probes": [], "meta": {}}
    done_keys = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
        for probe in results.get("probes", []):
            wt = probe["weights"]
            key = tuple(wt[k] for k in WEIGHT_NAMES)
            done_keys.add(key)
        print(f"  Resuming: {len(done_keys)} probes already done")

    results["meta"] = {
        "r_probe": R_PROBE,
        "n_directions": N_DIRECTIONS,
        "n_unique_weights": len(unique_weights),
    }

    total = len(unique_weights)
    n_done = len(done_keys)
    n_sims = 0

    for i, entry in enumerate(unique_weights):
        key = tuple(entry["weights"][k] for k in WEIGHT_NAMES)
        if key in done_keys:
            continue

        conds = ", ".join(entry["conditions"])
        seeds_str = entry["seeds"][0][:40] if entry["seeds"] else "?"
        print(f"\n  [{n_done+1}/{total}] {conds}: {seeds_str}...")

        probe = probe_cliffiness(entry["weights"])
        n_sims += N_DIRECTIONS + 1  # 1 base + 6 perturbations

        probe["weights"] = entry["weights"]
        probe["conditions"] = entry["conditions"]
        probe["seeds"] = entry["seeds"]
        probe["known_dx"] = entry["known_dx"]
        probe["dx_match"] = abs(probe["base_dx"] - entry["known_dx"]) < 0.01

        results["probes"].append(probe)
        done_keys.add(key)
        n_done += 1

        print(f"    DX={probe['base_dx']:.3f}m (known: {entry['known_dx']:.3f}m) "
              f"cliffiness={probe['cliffiness']:.4f} "
              f"grad_mag={probe['gradient_magnitude']:.4f}")

        # Save after each probe
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    results["meta"]["total_sims"] = n_sims

    # ── Analysis ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    probes = results["probes"]
    cliffiness_vals = [p["cliffiness"] for p in probes]
    grad_mags = [p["gradient_magnitude"] for p in probes]

    print(f"  Total probes: {len(probes)}")
    print(f"  DX matches: {sum(1 for p in probes if p.get('dx_match', False))}/{len(probes)}")
    print(f"  Cliffiness: mean={np.mean(cliffiness_vals):.4f}, "
          f"median={np.median(cliffiness_vals):.4f}, "
          f"max={np.max(cliffiness_vals):.4f}")
    print(f"  Gradient magnitude: mean={np.mean(grad_mags):.4f}, "
          f"median={np.median(grad_mags):.4f}")

    # Compare to atlas median (7.33 from categorical_structure_results.json)
    atlas_path = PROJECT / "artifacts" / "atlas_cliffiness.json"
    if atlas_path.exists():
        with open(atlas_path) as f:
            atlas = json.load(f)
        atlas_cliffiness = [p["cliffiness"] for p in atlas["probe_results"]]
        atlas_median = np.median(atlas_cliffiness)
        print(f"\n  Atlas cliffiness median: {atlas_median:.4f}")
        n_below = sum(1 for c in cliffiness_vals if c < atlas_median)
        print(f"  LLM probes below atlas median: {n_below}/{len(cliffiness_vals)} "
              f"({100*n_below/len(cliffiness_vals):.0f}%)")

        # Mann-Whitney U test
        all_llm = np.array(cliffiness_vals)
        all_atlas = np.array(atlas_cliffiness)
        n1, n2 = len(all_llm), len(all_atlas)
        combined = np.concatenate([all_llm, all_atlas])
        ranks = np.empty_like(combined)
        order = np.argsort(combined)
        ranks[order] = np.arange(1, len(combined) + 1)
        U1 = np.sum(ranks[:n1]) - n1 * (n1 + 1) / 2
        mu_U = n1 * n2 / 2
        sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z_score = (U1 - mu_U) / sigma_U if sigma_U > 0 else 0.0
        print(f"  Mann-Whitney U: z={z_score:.3f} "
              f"({'LLM smoother' if z_score < 0 else 'LLM rougher'})")

        results["comparison"] = {
            "atlas_median_cliffiness": float(atlas_median),
            "llm_median_cliffiness": float(np.median(cliffiness_vals)),
            "llm_mean_cliffiness": float(np.mean(cliffiness_vals)),
            "fraction_below_atlas_median": float(n_below / len(cliffiness_vals)),
            "mann_whitney_z": float(z_score),
        }

        # Per-condition breakdown
        print("\n  Per-condition measured cliffiness:")
        for cond in ["verbs", "theorems", "bible", "places"]:
            cond_cliff = [p["cliffiness"] for p in probes
                          if cond in p.get("conditions", [])]
            if cond_cliff:
                print(f"    {cond}: mean={np.mean(cond_cliff):.4f}, "
                      f"median={np.median(cond_cliff):.4f}, n={len(cond_cliff)}")

    # ── Figures ──────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = PROJECT / "artifacts" / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Fig 1: Measured vs atlas cliffiness distributions
    if atlas_path.exists():
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0, max(np.max(atlas_cliffiness), np.max(cliffiness_vals)), 40)
        ax.hist(atlas_cliffiness, bins=bins, alpha=0.5, density=True,
                label=f"Atlas 500 points (median={atlas_median:.2f})")
        ax.hist(cliffiness_vals, bins=bins, alpha=0.7, density=True,
                label=f"LLM {len(cliffiness_vals)} points (median={np.median(cliffiness_vals):.2f})")
        ax.axvline(atlas_median, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Cliffiness (max |ΔDX| at r=0.05)")
        ax.set_ylabel("Density")
        ax.set_title("Direct Cliffiness Measurement: LLM Points vs Atlas")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "pp_fig01_measured_vs_atlas.png", dpi=300)
        plt.close(fig)
        print(f"\n  WROTE {plot_dir / 'pp_fig01_measured_vs_atlas.png'}")

    # Fig 2: Per-condition box plot
    fig, ax = plt.subplots(figsize=(8, 5))
    cond_data = []
    cond_labels = []
    for cond in ["verbs", "theorems", "bible", "places"]:
        cond_cliff = [p["cliffiness"] for p in probes
                      if cond in p.get("conditions", [])]
        if cond_cliff:
            cond_data.append(cond_cliff)
            cond_labels.append(cond)
    if atlas_path.exists():
        cond_data.append(atlas_cliffiness)
        cond_labels.append("atlas\n(500 random)")
    bp = ax.boxplot(cond_data, labels=cond_labels, patch_artist=True)
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#aaaaaa"]
    for patch, color in zip(bp["boxes"], colors[:len(cond_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Cliffiness")
    ax.set_title("Directly Measured Cliffiness by Condition")
    fig.tight_layout()
    fig.savefig(plot_dir / "pp_fig02_cliffiness_by_condition.png", dpi=300)
    plt.close(fig)
    print(f"  WROTE {plot_dir / 'pp_fig02_cliffiness_by_condition.png'}")

    # Final save
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  WROTE {OUT_PATH}")


if __name__ == "__main__":
    main()
