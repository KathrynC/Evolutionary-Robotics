#!/usr/bin/env python3
"""
categorical_structure.py

Formal Categorical Structure Validation
=========================================

Empirically validates the claims from atlas_llm_evolution_theory.md that the
pipeline Sem -> Wt -> Beh has genuine categorical structure:

  F: Sem -> Wt is a functor preserving structural relationships
  G: Wt -> Beh is approximately a functor on the smooth subcategory Wt_smooth
  The LLM acts as a regularizer keeping F(Sem) in Wt_smooth

Uses existing data from 495 trials (5 conditions x ~100 trials), 500 atlas
cliffiness probes, and 50 cliff taxonomy profiles. No new simulations required.

OUTPUTS
-------
  artifacts/categorical_structure_results.json  — all computed metrics
  artifacts/plots/cs_fig01-08_*.png             — 8 matplotlib figures
  Console summary report

PHASES
------
  0. Data loading
  1. Define categories operationally (Sem, Wt, Beh)
  2. Functor F: Sem -> Wt (clustering, synonyms, morphism preservation, collapse)
  3. Map G: Wt -> Beh (atlas continuity, LLM as regularizer, local continuity)
  4. Composition G∘F (Mantel test, triptych, synonym behavioral equivalence)
  5. Sheaf structure (patch identification, LLM patch selection)
  6. Information geometry (output distributions, effective dimensionality)
  8. Output (JSON, figures, console report)

Usage:
    python3 categorical_structure.py
"""

import copy
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from compute_beer_analytics import NumpyEncoder

PLOT_DIR = PROJECT / "artifacts" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    "verbs":    PROJECT / "artifacts" / "structured_random_verbs.json",
    "theorems": PROJECT / "artifacts" / "structured_random_theorems.json",
    "bible":    PROJECT / "artifacts" / "structured_random_bible.json",
    "places":   PROJECT / "artifacts" / "structured_random_places.json",
    "baseline": PROJECT / "artifacts" / "structured_random_baseline.json",
}

ATLAS_PATH = PROJECT / "artifacts" / "atlas_cliffiness.json"
TAXONOMY_PATH = PROJECT / "artifacts" / "cliff_taxonomy.json"

COLORS = {
    "verbs":    "#E24A33",
    "theorems": "#348ABD",
    "bible":    "#988ED5",
    "places":   "#8EBA42",
    "baseline": "#777777",
}

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]
BEH_KEYS = ["dx", "speed", "efficiency", "phase_lock", "entropy", "roll_dom", "yaw_net_rad", "dy"]

# Synonym sets: groups of seeds that should map to identical weights
SYNONYM_SETS = {
    "stumble": ["stumble", "stolpern", "tropezar", "tropecar"],
    "walk":    ["zou", "geotda", "yamshi", "aruku", "peripatein", "kutembea",
                "chalna", "ambulare"],
    "crawl":   ["crawl", "pa", "kriechen", "polzti", "kutambaa", "serpere",
                "ramper", "yazhaf"],
    "jump":    ["prygat", "saltar", "tiao", "tobu", "yaqfiz", "kudna", "kuruka"],
    "stagger": ["stagger", "wobble", "sendelemek", "schwanken", "waddle",
                "chanceler", "shatat'sya", "taumeln", "yureru", "heundeullida",
                "tambalearse"],
    "sprint":  ["sprint", "leap", "rennen"],
}

# Structural property tags for a subset of seeds
STRUCTURAL_TAGS = {
    "periodic":   ["Ecclesiastes 1:6", "KAM Theorem"],
    "symmetric":  ["Noether's Theorem", "Konig's Theorem"],
    "violent":    ["Revelation 6:8"],
    "stable":     ["Ecclesiastes 1:6"],
}


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


# ── Mann-Whitney U (from structured_random_compare.py) ──────────────────────

def mann_whitney_u(x, y):
    """Mann-Whitney U test (no scipy). Returns (U, z-score)."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.empty_like(combined, dtype=float)
    order = np.argsort(combined)
    ranks[order] = np.arange(1, len(combined) + 1)
    for val in np.unique(combined):
        mask = combined == val
        ranks[mask] = ranks[mask].mean()
    r1 = ranks[:nx].sum()
    u1 = r1 - nx * (nx + 1) / 2
    mu = nx * ny / 2
    sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12)
    z = (u1 - mu) / sigma if sigma > 0 else 0.0
    return float(u1), float(z)


def sig_stars(z):
    az = abs(z)
    if az > 2.58:
        return "***"
    if az > 1.96:
        return "**"
    if az > 1.64:
        return "*"
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 0: DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load trial data, atlas cliffiness, and cliff taxonomy."""
    # Trial data
    data = {}
    for name, path in CONDITIONS.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        with open(path) as f:
            data[name] = json.load(f)
        print(f"  Loaded {name}: {len(data[name])} trials")

    # Atlas cliffiness
    atlas = None
    if ATLAS_PATH.exists():
        with open(ATLAS_PATH) as f:
            atlas = json.load(f)
        print(f"  Loaded atlas: {len(atlas['probe_results'])} probe points")
    else:
        print(f"  WARNING: {ATLAS_PATH} not found")

    # Cliff taxonomy
    taxonomy = None
    if TAXONOMY_PATH.exists():
        with open(TAXONOMY_PATH) as f:
            taxonomy = json.load(f)
        print(f"  Loaded taxonomy: {len(taxonomy['profiles'])} cliff profiles")
    else:
        print(f"  WARNING: {TAXONOMY_PATH} not found")

    return data, atlas, taxonomy


def extract_matrices(data):
    """Extract per-condition weight matrices (Nx6) and behavioral matrices (Nx8)."""
    cond_weights = {}
    cond_beh = {}
    cond_seeds = {}

    for name, trials in data.items():
        seeds = []
        wmat = []
        bmat = []
        for t in trials:
            seeds.append(t["seed"])
            wmat.append([t["weights"][k] for k in WEIGHT_NAMES])
            bmat.append([t.get(k, 0.0) for k in BEH_KEYS])
        cond_weights[name] = np.array(wmat)
        cond_beh[name] = np.array(bmat)
        cond_seeds[name] = seeds

    return cond_weights, cond_beh, cond_seeds


def extract_atlas_matrices(atlas):
    """Extract atlas weight matrix and cliffiness array."""
    probes = atlas["probe_results"]
    weights = np.array([[p["weights"][k] for k in WEIGHT_NAMES] for p in probes])
    cliffiness = np.array([p["cliffiness"] for p in probes])
    gradient_mag = np.array([p["gradient_magnitude"] for p in probes])
    return weights, cliffiness, gradient_mag


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: DEFINE CATEGORIES
# ═════════════════════════════════════════════════════════════════════════════

def build_semantic_distance(cond_seeds, cond_weights):
    """Build semantic distance at 3 tiers: condition, synonym, structural."""

    # Flatten all seeds into ordered list
    all_seeds = []
    all_conds = []
    for name in ["verbs", "theorems", "bible", "places", "baseline"]:
        if name in cond_seeds:
            for s in cond_seeds[name]:
                all_seeds.append(s)
                all_conds.append(name)

    N = len(all_seeds)

    # Tier 1: condition membership (binary)
    cond_dist = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = 0.0 if all_conds[i] == all_conds[j] else 1.0
            cond_dist[i, j] = d
            cond_dist[j, i] = d

    # Tier 2: synonym membership (0 = same synonym set, 0.5 = same condition, 1 = different)
    # Build seed -> synonym set mapping (match by substring in seed)
    seed_to_syn = {}
    for syn_name, members in SYNONYM_SETS.items():
        for member in members:
            for idx, seed in enumerate(all_seeds):
                if member.lower() in seed.lower().split("(")[0].split(",")[0].strip().lower():
                    seed_to_syn[idx] = syn_name

    syn_dist = cond_dist.copy()
    for i in range(N):
        for j in range(i + 1, N):
            if i in seed_to_syn and j in seed_to_syn:
                if seed_to_syn[i] == seed_to_syn[j]:
                    syn_dist[i, j] = 0.0
                    syn_dist[j, i] = 0.0
            if all_conds[i] == all_conds[j] and cond_dist[i, j] == 0.0:
                if syn_dist[i, j] == 0.0 and i not in seed_to_syn:
                    syn_dist[i, j] = 0.5
                    syn_dist[j, i] = 0.5

    return all_seeds, all_conds, cond_dist, syn_dist, seed_to_syn


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: FUNCTOR F: Sem -> Wt
# ═════════════════════════════════════════════════════════════════════════════

def phase2_functor_F(cond_weights, cond_beh, cond_seeds, data):
    """Analyze the LLM mapping F: Sem -> Wt."""
    results = {}

    # 2A: Weight clustering — faithfulness ratio
    clustering = {}
    for name, wmat in cond_weights.items():
        wtuples = [tuple(np.round(row, 6)) for row in wmat]
        unique = list(set(wtuples))
        counts = defaultdict(int)
        for wt in wtuples:
            counts[wt] += 1
        cluster_sizes = sorted(counts.values(), reverse=True)
        clustering[name] = {
            "n_trials": len(wtuples),
            "n_unique": len(unique),
            "faithfulness": len(unique) / len(wtuples),
            "cluster_sizes": cluster_sizes,
            "largest_cluster": cluster_sizes[0],
        }
    results["clustering"] = clustering

    # 2B: Synonym convergence
    synonym_results = {}
    verb_seeds = cond_seeds.get("verbs", [])
    verb_weights = cond_weights.get("verbs", np.zeros((0, 6)))

    for syn_name, members in SYNONYM_SETS.items():
        # Find indices of synonym members in verbs
        indices = []
        matched_seeds = []
        for i, seed in enumerate(verb_seeds):
            seed_base = seed.lower().split("(")[0].split(",")[0].strip()
            for member in members:
                if member.lower() == seed_base:
                    indices.append(i)
                    matched_seeds.append(seed)
                    break

        if len(indices) < 2:
            continue

        syn_wts = verb_weights[indices]
        # Within-set pairwise distances
        within_dists = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                within_dists.append(np.linalg.norm(syn_wts[i] - syn_wts[j]))

        # Bootstrap: random groups of same size, 1000 permutations
        n_boot = 1000
        random_dists = []
        rng = np.random.RandomState(42)
        for _ in range(n_boot):
            sample_idx = rng.choice(len(verb_weights), size=len(indices), replace=False)
            sample_wts = verb_weights[sample_idx]
            dists = []
            for i in range(len(sample_idx)):
                for j in range(i + 1, len(sample_idx)):
                    dists.append(np.linalg.norm(sample_wts[i] - sample_wts[j]))
            random_dists.append(np.mean(dists))

        synonym_results[syn_name] = {
            "n_members": len(indices),
            "matched_seeds": matched_seeds,
            "mean_within_dist": float(np.mean(within_dists)),
            "max_within_dist": float(np.max(within_dists)),
            "bootstrap_mean": float(np.mean(random_dists)),
            "bootstrap_std": float(np.std(random_dists)),
            "p_value": float(np.mean(np.array(random_dists) <= np.mean(within_dists))),
            "all_identical": float(np.max(within_dists)) < 1e-10,
        }
    results["synonym_convergence"] = synonym_results

    # 2C: Morphism preservation — within vs cross condition weight distances
    morphism = {}
    structured = [n for n in ["verbs", "theorems", "bible", "places"] if n in cond_weights]
    for name in structured:
        wmat = cond_weights[name]
        bmat = cond_beh[name]

        # Within-condition pairwise weight distances
        within_w = []
        for i in range(len(wmat)):
            for j in range(i + 1, len(wmat)):
                within_w.append(np.linalg.norm(wmat[i] - wmat[j]))

        # Cross-condition: compare to baseline
        if "baseline" not in cond_weights:
            continue
        base_wmat = cond_weights["baseline"]
        cross_w = []
        for i in range(len(wmat)):
            for j in range(len(base_wmat)):
                cross_w.append(np.linalg.norm(wmat[i] - base_wmat[j]))

        u_w, z_w = mann_whitney_u(within_w, cross_w)

        # Same for behavioral distances
        within_b = []
        for i in range(len(bmat)):
            for j in range(i + 1, len(bmat)):
                within_b.append(np.linalg.norm(bmat[i] - bmat[j]))
        base_bmat = cond_beh["baseline"]
        cross_b = []
        for i in range(len(bmat)):
            for j in range(len(base_bmat)):
                cross_b.append(np.linalg.norm(bmat[i] - base_bmat[j]))
        u_b, z_b = mann_whitney_u(within_b, cross_b)

        morphism[name] = {
            "within_weight_dist_mean": float(np.mean(within_w)),
            "cross_weight_dist_mean": float(np.mean(cross_w)),
            "weight_U": u_w,
            "weight_z": z_w,
            "weight_sig": sig_stars(z_w),
            "within_beh_dist_mean": float(np.mean(within_b)),
            "cross_beh_dist_mean": float(np.mean(cross_b)),
            "beh_U": u_b,
            "beh_z": z_b,
            "beh_sig": sig_stars(z_b),
        }
    results["morphism_preservation"] = morphism

    # 2D: Collapse analysis — group seeds by identical weight vector
    collapse = {}
    for name in structured:
        wmat = cond_weights[name]
        seeds = cond_seeds[name]
        groups = defaultdict(list)
        for i, row in enumerate(wmat):
            key = tuple(np.round(row, 6))
            groups[key].append(seeds[i])

        # Find groups with multiple seeds (collapsed distinctions)
        collapsed_groups = []
        for wt, seed_list in groups.items():
            if len(seed_list) > 1:
                collapsed_groups.append({
                    "weights": [float(w) for w in wt],
                    "n_seeds": len(seed_list),
                    "seeds": seed_list[:10],  # cap at 10 for readability
                })
        collapsed_groups.sort(key=lambda g: g["n_seeds"], reverse=True)
        collapse[name] = {
            "n_collapsed_groups": len(collapsed_groups),
            "max_group_size": collapsed_groups[0]["n_seeds"] if collapsed_groups else 1,
            "groups": collapsed_groups[:5],  # top 5
        }
    results["collapse_analysis"] = collapse

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: MAP G: Wt -> Beh
# ═════════════════════════════════════════════════════════════════════════════

def phase3_map_G(cond_weights, atlas, taxonomy):
    """Analyze the simulation mapping G: Wt -> Beh."""
    if atlas is None:
        print("  SKIP Phase 3: no atlas data")
        return {}

    atlas_weights, atlas_cliff, atlas_grad = extract_atlas_matrices(atlas)
    results = {}

    # 3A: Atlas continuity — cliffiness histogram statistics
    median_cliff = float(np.median(atlas_cliff))
    smooth_mask = atlas_cliff < median_cliff
    results["atlas_continuity"] = {
        "n_atlas": len(atlas_cliff),
        "median_cliffiness": median_cliff,
        "mean_cliffiness": float(np.mean(atlas_cliff)),
        "std_cliffiness": float(np.std(atlas_cliff)),
        "max_cliffiness": float(np.max(atlas_cliff)),
        "min_cliffiness": float(np.min(atlas_cliff)),
        "n_smooth": int(smooth_mask.sum()),
        "n_rough": int((~smooth_mask).sum()),
        "smooth_threshold": median_cliff,
    }

    # 3B: LLM as regularizer — KNN-interpolate cliffiness for LLM vs baseline points
    regularizer = {}
    K = 5  # nearest neighbors

    for name, wmat in cond_weights.items():
        # Compute distances from each LLM point to all atlas points
        # wmat: (N, 6), atlas_weights: (500, 6)
        dists = np.linalg.norm(wmat[:, None, :] - atlas_weights[None, :, :], axis=2)
        # For each LLM point, find K nearest atlas neighbors
        knn_idx = np.argsort(dists, axis=1)[:, :K]
        knn_dists = np.take_along_axis(dists, knn_idx, axis=1)

        # Inverse-distance weighted cliffiness
        # Guard against zero distance
        knn_dists = np.maximum(knn_dists, 1e-12)
        inv_w = 1.0 / knn_dists
        inv_w_sum = inv_w.sum(axis=1, keepdims=True)
        weights_norm = inv_w / inv_w_sum

        knn_cliff = atlas_cliff[knn_idx]
        interp_cliff = (weights_norm * knn_cliff).sum(axis=1)

        regularizer[name] = {
            "mean_interp_cliffiness": float(np.mean(interp_cliff)),
            "median_interp_cliffiness": float(np.median(interp_cliff)),
            "std_interp_cliffiness": float(np.std(interp_cliff)),
            "min_nearest_atlas_dist": float(np.min(knn_dists[:, 0])),
            "mean_nearest_atlas_dist": float(np.mean(knn_dists[:, 0])),
            "interp_cliffiness_values": interp_cliff.tolist(),
        }

    # Mann-Whitney U: LLM conditions vs baseline cliffiness
    if "baseline" in regularizer:
        base_cliff = regularizer["baseline"]["interp_cliffiness_values"]
        for name in ["verbs", "theorems", "bible", "places"]:
            if name in regularizer:
                llm_cliff = regularizer[name]["interp_cliffiness_values"]
                u, z = mann_whitney_u(llm_cliff, base_cliff)
                regularizer[name]["vs_baseline_U"] = u
                regularizer[name]["vs_baseline_z"] = z
                regularizer[name]["vs_baseline_sig"] = sig_stars(z)

    results["regularizer"] = regularizer

    # 3C: Local continuity from atlas perturbation data
    # Use delta_dxs from atlas probes as measure of local behavioral variation
    probes = atlas["probe_results"]
    local_variations = []
    for p in probes:
        ddx = np.array(p["delta_dxs"])
        local_variations.append(float(np.std(ddx)))

    local_var_arr = np.array(local_variations)

    # For each LLM point, find distance to nearest atlas point and
    # correlate with that atlas point's local variation
    local_continuity = {}
    for name, wmat in cond_weights.items():
        dists = np.linalg.norm(wmat[:, None, :] - atlas_weights[None, :, :], axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(len(wmat)), nearest_idx]
        nearest_var = local_var_arr[nearest_idx]

        local_continuity[name] = {
            "mean_nearest_local_var": float(np.mean(nearest_var)),
            "median_nearest_local_var": float(np.median(nearest_var)),
            "mean_nearest_dist": float(np.mean(nearest_dist)),
        }
    results["local_continuity"] = local_continuity

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4: COMPOSITION G∘F
# ═════════════════════════════════════════════════════════════════════════════

def mantel_test(D1, D2, n_perm=1000):
    """Mantel test: correlation between two distance matrices with permutation p-value."""
    N = D1.shape[0]
    # Extract upper triangle
    idx = np.triu_indices(N, k=1)
    v1 = D1[idx]
    v2 = D2[idx]

    # Observed Pearson correlation
    r_obs = np.corrcoef(v1, v2)[0, 1]

    # Permutation test
    rng = np.random.RandomState(42)
    n_geq = 0
    for _ in range(n_perm):
        perm = rng.permutation(N)
        D2_perm = D2[np.ix_(perm, perm)]
        v2_perm = D2_perm[idx]
        r_perm = np.corrcoef(v1, v2_perm)[0, 1]
        if r_perm >= r_obs:
            n_geq += 1

    p_value = (n_geq + 1) / (n_perm + 1)
    return float(r_obs), float(p_value)


def phase4_composition(cond_weights, cond_beh, cond_seeds, data):
    """Analyze the end-to-end composition G∘F: Sem -> Beh."""
    results = {}

    # 4A: Mantel test — semantic distance vs behavioral distance
    # Use condition-level distances (within vs cross) on all 495 trials
    all_weights = []
    all_beh = []
    all_conds = []
    cond_order = ["verbs", "theorems", "bible", "places", "baseline"]
    for name in cond_order:
        if name not in cond_weights:
            continue
        all_weights.append(cond_weights[name])
        all_beh.append(cond_beh[name])
        all_conds.extend([name] * len(cond_weights[name]))

    all_weights = np.vstack(all_weights)
    all_beh = np.vstack(all_beh)
    N = len(all_weights)

    # Z-score normalize behavioral features
    beh_mu = all_beh.mean(axis=0)
    beh_std = all_beh.std(axis=0)
    beh_std[beh_std < 1e-12] = 1.0
    beh_z = (all_beh - beh_mu) / beh_std

    # Semantic distance matrix (condition membership: 0=same, 1=different)
    sem_dist = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = 0.0 if all_conds[i] == all_conds[j] else 1.0
            sem_dist[i, j] = d
            sem_dist[j, i] = d

    # Weight distance matrix
    # For large N, compute pairwise efficiently
    wt_dist = np.zeros((N, N))
    for i in range(N):
        diff = all_weights[i] - all_weights[i + 1:]
        wt_dist[i, i + 1:] = np.sqrt((diff ** 2).sum(axis=1))
        wt_dist[i + 1:, i] = wt_dist[i, i + 1:]

    # Behavioral distance matrix
    beh_dist = np.zeros((N, N))
    for i in range(N):
        diff = beh_z[i] - beh_z[i + 1:]
        beh_dist[i, i + 1:] = np.sqrt((diff ** 2).sum(axis=1))
        beh_dist[i + 1:, i] = beh_dist[i, i + 1:]

    # Mantel test: sem_dist vs wt_dist
    r_sw, p_sw = mantel_test(sem_dist, wt_dist)
    # Mantel test: wt_dist vs beh_dist
    r_wb, p_wb = mantel_test(wt_dist, beh_dist)
    # Mantel test: sem_dist vs beh_dist (end-to-end)
    r_sb, p_sb = mantel_test(sem_dist, beh_dist)

    results["mantel_tests"] = {
        "sem_vs_wt": {"r": r_sw, "p": p_sw},
        "wt_vs_beh": {"r": r_wb, "p": p_wb},
        "sem_vs_beh": {"r": r_sb, "p": p_sb},
    }

    # Store distance matrices upper triangle means for reporting
    idx = np.triu_indices(N, k=1)
    results["distance_summaries"] = {
        "wt_dist_mean": float(np.mean(wt_dist[idx])),
        "beh_dist_mean": float(np.mean(beh_dist[idx])),
    }

    # 4B: Triptych verification
    triptych = {}
    # Revelation 6:8
    if "bible" in data:
        for t in data["bible"]:
            if "Revelation 6:8" in t["seed"] or "behold a pale horse" in t["seed"]:
                triptych["revelation"] = {
                    "seed": t["seed"],
                    "weights": t["weights"],
                    "dx": t["dx"],
                    "speed": t["speed"],
                    "efficiency": t["efficiency"],
                    "phase_lock": t["phase_lock"],
                    "verified_dx_29m": abs(t["dx"] - 29.17) < 0.5,
                }
                break
    # Ecclesiastes 1:6
    if "bible" in data:
        for t in data["bible"]:
            if "Ecclesiastes 1:6" in t["seed"] or "whirleth" in t["seed"]:
                triptych["ecclesiastes"] = {
                    "seed": t["seed"],
                    "weights": t["weights"],
                    "dx": t["dx"],
                    "efficiency": t["efficiency"],
                    "phase_lock": t["phase_lock"],
                    "verified_eff_00495": abs(t["efficiency"] - 0.00495) < 0.001,
                }
                break
    # Noether's Theorem
    if "theorems" in data:
        for t in data["theorems"]:
            if "Noether" in t["seed"]:
                triptych["noether"] = {
                    "seed": t["seed"],
                    "weights": t["weights"],
                    "dx": t["dx"],
                    "speed": t["speed"],
                    "phase_lock": t["phase_lock"],
                    "verified_dx_003m": abs(t["dx"]) < 0.1,
                }
                break
    results["triptych"] = triptych

    # 4C: Synonym behavioral equivalence — identical weights -> identical behavior
    synonym_beh = {}
    if "verbs" in data:
        verb_trials = data["verbs"]
        for syn_name, members in SYNONYM_SETS.items():
            matched = []
            for t in verb_trials:
                seed_base = t["seed"].lower().split("(")[0].split(",")[0].strip()
                for m in members:
                    if m.lower() == seed_base:
                        matched.append(t)
                        break
            if len(matched) < 2:
                continue

            # Check if all behavioral metrics are identical
            bvecs = np.array([[t.get(k, 0.0) for k in BEH_KEYS] for t in matched])
            max_beh_diff = 0.0
            for i in range(len(bvecs)):
                for j in range(i + 1, len(bvecs)):
                    max_beh_diff = max(max_beh_diff, np.max(np.abs(bvecs[i] - bvecs[j])))

            wvecs = np.array([[t["weights"][k] for k in WEIGHT_NAMES] for t in matched])
            max_wt_diff = 0.0
            for i in range(len(wvecs)):
                for j in range(i + 1, len(wvecs)):
                    max_wt_diff = max(max_wt_diff, np.max(np.abs(wvecs[i] - wvecs[j])))

            synonym_beh[syn_name] = {
                "n_matched": len(matched),
                "weights_identical": max_wt_diff < 1e-10,
                "behavior_identical": max_beh_diff < 1e-10,
                "max_weight_diff": float(max_wt_diff),
                "max_behavior_diff": float(max_beh_diff),
            }
    results["synonym_behavioral_equivalence"] = synonym_beh

    # Store subsampled vectors for scatter plots
    results["_scatter_data"] = {
        "wt_dist_upper": wt_dist[idx].tolist(),
        "beh_dist_upper": beh_dist[idx].tolist(),
        "sem_dist_upper": sem_dist[idx].tolist(),
    }

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5: SHEAF STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

def phase5_sheaf(cond_weights, atlas):
    """Identify smooth patches and LLM patch selection."""
    if atlas is None:
        print("  SKIP Phase 5: no atlas data")
        return {}

    atlas_weights, atlas_cliff, _ = extract_atlas_matrices(atlas)
    N_atlas = len(atlas_cliff)
    median_cliff = float(np.median(atlas_cliff))
    smooth_mask = atlas_cliff < median_cliff  # boolean: True = smooth
    results = {}

    # 5A: Patch identification via BFS on adjacency graph
    # Adjacency: Euclidean distance < radius AND both smooth
    radius = 0.5  # threshold in 6D weight space
    atlas_smooth_idx = np.where(smooth_mask)[0]
    smooth_weights = atlas_weights[atlas_smooth_idx]

    # Build adjacency among smooth points
    n_smooth = len(atlas_smooth_idx)
    adj = defaultdict(set)
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            d = np.linalg.norm(smooth_weights[i] - smooth_weights[j])
            if d < radius:
                adj[i].add(j)
                adj[j].add(i)

    # BFS connected components
    visited = set()
    patches = []
    for start in range(n_smooth):
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        patches.append(component)

    # Map from smooth-local-index to patch ID
    point_to_patch = {}
    for pid, comp in enumerate(patches):
        for node in comp:
            point_to_patch[node] = pid

    patch_sizes = sorted([len(p) for p in patches], reverse=True)
    results["patch_identification"] = {
        "n_smooth_points": n_smooth,
        "n_patches": len(patches),
        "patch_sizes": patch_sizes[:20],
        "radius": radius,
        "median_cliffiness_threshold": median_cliff,
    }

    # 5B: LLM patch selection — map each LLM weight to nearest smooth atlas patch
    llm_patches = {}
    for name, wmat in cond_weights.items():
        if len(smooth_weights) == 0:
            continue
        # Distance from each LLM point to each smooth atlas point
        dists = np.linalg.norm(wmat[:, None, :] - smooth_weights[None, :, :], axis=2)
        nearest_smooth = np.argmin(dists, axis=1)
        patches_used = set()
        for ns in nearest_smooth:
            if ns in point_to_patch:
                patches_used.add(point_to_patch[ns])
        llm_patches[name] = {
            "n_patches_used": len(patches_used),
            "patches_used": sorted(patches_used),
            "mean_dist_to_smooth": float(np.mean(np.min(dists, axis=1))),
        }
    results["llm_patch_selection"] = llm_patches

    # Store patch assignments for plotting
    results["_patch_data"] = {
        "smooth_weights": smooth_weights.tolist(),
        "patch_ids": [point_to_patch.get(i, -1) for i in range(n_smooth)],
        "atlas_smooth_idx": atlas_smooth_idx.tolist(),
    }

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 6: INFORMATION GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════

def phase6_info_geometry(cond_weights):
    """Analyze the LLM's output distribution geometry."""
    results = {}

    # 6A: Per-condition mean/covariance and Mahalanobis distances
    distributions = {}
    centroids = {}
    for name, wmat in cond_weights.items():
        mu = wmat.mean(axis=0)
        cov = np.cov(wmat.T) if wmat.shape[0] > 1 else np.zeros((6, 6))
        # Handle degenerate covariance
        if np.linalg.matrix_rank(cov) < 6:
            cov_reg = cov + 1e-8 * np.eye(6)
        else:
            cov_reg = cov
        distributions[name] = {
            "mean": mu.tolist(),
            "cov_trace": float(np.trace(cov)),
            "cov_det": float(np.linalg.det(cov_reg)),
            "cov_rank": int(np.linalg.matrix_rank(cov, tol=1e-6)),
        }
        centroids[name] = (mu, cov_reg)

    # Mahalanobis distance between condition centroids
    mahal = {}
    cond_order = [n for n in ["verbs", "theorems", "bible", "places", "baseline"] if n in centroids]
    for i, a in enumerate(cond_order):
        for b in cond_order[i + 1:]:
            mu_a, cov_a = centroids[a]
            mu_b, cov_b = centroids[b]
            pooled_cov = (cov_a + cov_b) / 2
            try:
                inv_cov = np.linalg.inv(pooled_cov)
                diff = mu_a - mu_b
                d = float(np.sqrt(diff @ inv_cov @ diff))
            except np.linalg.LinAlgError:
                d = float("nan")
            mahal[f"{a}_vs_{b}"] = d

    distributions["mahalanobis"] = mahal
    results["distributions"] = distributions

    # 6B: Effective dimensionality via PCA eigenvalue participation ratio
    dimensionality = {}
    for name, wmat in cond_weights.items():
        if wmat.shape[0] < 2:
            continue
        mu = wmat.mean(axis=0)
        centered = wmat - mu
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = S ** 2 / (len(wmat) - 1)
        # Participation ratio: (sum λ)^2 / sum(λ^2)
        lam_sum = eigenvalues.sum()
        lam_sq_sum = (eigenvalues ** 2).sum()
        if lam_sq_sum > 0:
            participation_ratio = lam_sum ** 2 / lam_sq_sum
        else:
            participation_ratio = 0.0

        var_explained = eigenvalues / lam_sum * 100 if lam_sum > 0 else eigenvalues * 0

        dimensionality[name] = {
            "eigenvalues": eigenvalues.tolist(),
            "var_explained_pct": var_explained.tolist(),
            "participation_ratio": float(participation_ratio),
            "effective_dims": float(participation_ratio),
        }
    results["dimensionality"] = dimensionality

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 8: FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def generate_figures(cond_weights, cond_beh, cond_seeds, data, atlas,
                     functor_F, map_G, composition_GF, sheaf, info_geo):
    """Generate all 8 figures."""

    cond_order = [n for n in ["verbs", "theorems", "bible", "places", "baseline"]
                  if n in cond_weights]

    # ── Fig 1: Weight-space PCA colored by condition ─────────────────────────
    print("\n  Generating figures...")
    all_w = np.vstack([cond_weights[n] for n in cond_order])
    all_labels = []
    for n in cond_order:
        all_labels.extend([n] * len(cond_weights[n]))

    mu_w = all_w.mean(axis=0)
    std_w = all_w.std(axis=0)
    std_w[std_w < 1e-12] = 1.0
    Zw = (all_w - mu_w) / std_w
    U, S, Vt = np.linalg.svd(Zw, full_matrices=False)
    pc_w = Zw @ Vt[:2].T
    var_w = S[:2] ** 2 / (S ** 2).sum() * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 0
    for name in cond_order:
        n = len(cond_weights[name])
        ax.scatter(pc_w[offset:offset + n, 0], pc_w[offset:offset + n, 1],
                   c=COLORS.get(name, "#CCC"), s=25, alpha=0.7, label=name,
                   edgecolors="white", linewidths=0.3)
        offset += n
    ax.set_xlabel(f"PC1 ({var_w[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_w[1]:.1f}% var)")
    ax.set_title("Weight-Space PCA by Condition")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cs_fig01_weight_pca.png")

    # ── Fig 2: Faithfulness ratio bar chart ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    faith = [functor_F["clustering"][n]["faithfulness"] * 100 for n in cond_order]
    bars = ax.bar(cond_order, faith,
                  color=[COLORS.get(n, "#CCC") for n in cond_order], alpha=0.8)
    for bar, pct in zip(bars, faith):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%", ha="center", fontsize=10)
    ax.set_ylabel("Faithfulness (% unique weight vectors)")
    ax.set_title("Functor F Faithfulness: Unique Weights / Total Trials")
    ax.set_ylim(0, 115)
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cs_fig02_faithfulness.png")

    # ── Fig 3: Semantic distance vs weight distance scatter ──────────────────
    if "_scatter_data" in composition_GF:
        sd = composition_GF["_scatter_data"]
        sem_upper = np.array(sd["sem_dist_upper"])
        wt_upper = np.array(sd["wt_dist_upper"])

        # Subsample for readability (can be >100k pairs)
        rng = np.random.RandomState(42)
        n_pairs = len(sem_upper)
        if n_pairs > 5000:
            idx = rng.choice(n_pairs, 5000, replace=False)
        else:
            idx = np.arange(n_pairs)

        fig, ax = plt.subplots(figsize=(8, 6))
        within_mask = sem_upper[idx] == 0
        cross_mask = sem_upper[idx] == 1
        ax.scatter(sem_upper[idx][within_mask] + rng.uniform(-0.05, 0.05, within_mask.sum()),
                   wt_upper[idx][within_mask], s=3, alpha=0.3, c="#348ABD", label="Within condition")
        ax.scatter(sem_upper[idx][cross_mask] + rng.uniform(-0.05, 0.05, cross_mask.sum()),
                   wt_upper[idx][cross_mask], s=3, alpha=0.3, c="#E24A33", label="Cross condition")
        ax.set_xlabel("Semantic Distance (condition membership)")
        ax.set_ylabel("Weight-Space Distance (L2)")
        r_sw = composition_GF["mantel_tests"]["sem_vs_wt"]["r"]
        p_sw = composition_GF["mantel_tests"]["sem_vs_wt"]["p"]
        ax.set_title(f"Sem vs Wt Distance (Mantel r={r_sw:.3f}, p={p_sw:.3f})")
        ax.legend()
        clean_ax(ax)
        fig.tight_layout()
        save_fig(fig, "cs_fig03_sem_vs_wt.png")

    # ── Fig 4: Weight distance vs behavioral distance scatter ────────────────
    if "_scatter_data" in composition_GF:
        beh_upper = np.array(sd["beh_dist_upper"])

        fig, ax = plt.subplots(figsize=(8, 6))
        # subsample
        n_pairs = len(wt_upper)
        if n_pairs > 5000:
            idx = rng.choice(n_pairs, 5000, replace=False)
        else:
            idx = np.arange(n_pairs)
        ax.scatter(wt_upper[idx], beh_upper[idx], s=3, alpha=0.2, c="#555555")
        ax.set_xlabel("Weight-Space Distance (L2)")
        ax.set_ylabel("Behavioral Distance (L2, z-scored)")
        r_wb = composition_GF["mantel_tests"]["wt_vs_beh"]["r"]
        p_wb = composition_GF["mantel_tests"]["wt_vs_beh"]["p"]
        ax.set_title(f"Wt vs Beh Distance (Mantel r={r_wb:.3f}, p={p_wb:.3f})")
        clean_ax(ax)
        fig.tight_layout()
        save_fig(fig, "cs_fig04_wt_vs_beh.png")

    # ── Fig 5: Smoothness comparison — LLM vs baseline cliffiness ────────────
    if map_G and "regularizer" in map_G:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Combine all LLM conditions
        llm_cliff = []
        for name in ["verbs", "theorems", "bible", "places"]:
            if name in map_G["regularizer"]:
                llm_cliff.extend(map_G["regularizer"][name]["interp_cliffiness_values"])
        llm_cliff = np.array(llm_cliff)

        base_cliff = np.array(map_G["regularizer"].get("baseline", {}).get(
            "interp_cliffiness_values", []))

        bins = np.linspace(0, max(np.max(llm_cliff) if len(llm_cliff) else 20,
                                   np.max(base_cliff) if len(base_cliff) else 20) * 1.1, 40)

        if len(llm_cliff) > 0:
            ax.hist(llm_cliff, bins=bins, alpha=0.6, color="#348ABD",
                    label=f"LLM conditions (n={len(llm_cliff)}, "
                          f"med={np.median(llm_cliff):.1f})", density=True)
        if len(base_cliff) > 0:
            ax.hist(base_cliff, bins=bins, alpha=0.6, color="#777777",
                    label=f"Baseline (n={len(base_cliff)}, "
                          f"med={np.median(base_cliff):.1f})", density=True)
        ax.set_xlabel("Interpolated Cliffiness (from 5-NN atlas)")
        ax.set_ylabel("Density")
        ax.set_title("LLM as Regularizer: Cliffiness at LLM vs Baseline Points")
        ax.legend()
        clean_ax(ax)
        fig.tight_layout()
        save_fig(fig, "cs_fig05_smoothness.png")

    # ── Fig 6: Sheaf patch map ───────────────────────────────────────────────
    if sheaf and "_patch_data" in sheaf:
        pd = sheaf["_patch_data"]
        smooth_w = np.array(pd["smooth_weights"])
        patch_ids = np.array(pd["patch_ids"])

        if len(smooth_w) > 0:
            # PCA of atlas smooth points
            mu_s = smooth_w.mean(axis=0)
            std_s = smooth_w.std(axis=0)
            std_s[std_s < 1e-12] = 1.0
            Zs = (smooth_w - mu_s) / std_s
            U_s, S_s, Vt_s = np.linalg.svd(Zs, full_matrices=False)
            pc_s = Zs @ Vt_s[:2].T

            fig, ax = plt.subplots(figsize=(10, 8))

            # Color patches
            n_patches = len(set(patch_ids))
            cmap = plt.cm.tab20
            colors_p = [cmap(pid % 20) for pid in patch_ids]
            ax.scatter(pc_s[:, 0], pc_s[:, 1], c=colors_p, s=15, alpha=0.5,
                       label=f"Atlas smooth ({len(smooth_w)} pts, {n_patches} patches)")

            # Overlay LLM points
            for name in ["verbs", "theorems", "bible", "places"]:
                if name not in cond_weights:
                    continue
                wmat = cond_weights[name]
                Zl = (wmat - mu_s) / std_s
                pc_l = Zl @ Vt_s[:2].T
                ax.scatter(pc_l[:, 0], pc_l[:, 1], c=COLORS[name], s=40, alpha=0.8,
                           marker="^", edgecolors="black", linewidths=0.5, label=name)

            ax.set_xlabel("PC1 (atlas smooth space)")
            ax.set_ylabel("PC2 (atlas smooth space)")
            ax.set_title(f"Sheaf Patch Map: Atlas Smooth Patches + LLM Points")
            ax.legend(fontsize=8, loc="upper right")
            clean_ax(ax)
            fig.tight_layout()
            save_fig(fig, "cs_fig06_sheaf_patches.png")

    # ── Fig 7: Effective dimensionality bar chart with eigenvalue inset ──────
    if info_geo and "dimensionality" in info_geo:
        dim = info_geo["dimensionality"]
        fig, ax = plt.subplots(figsize=(10, 6))
        names_dim = [n for n in cond_order if n in dim]
        pr_values = [dim[n]["participation_ratio"] for n in names_dim]
        bars = ax.bar(names_dim, pr_values,
                      color=[COLORS.get(n, "#CCC") for n in names_dim], alpha=0.8)
        for bar, pr in zip(bars, pr_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{pr:.2f}", ha="center", fontsize=10)
        ax.set_ylabel("Participation Ratio (effective dimensions)")
        ax.set_title("Effective Dimensionality of Weight-Space Occupancy")
        ax.axhline(y=6, color="#AAA", linestyle="--", alpha=0.5, label="Max (6D)")
        ax.legend()
        clean_ax(ax)

        fig.tight_layout()

        # Eigenvalue inset (added after tight_layout to avoid conflict)
        ax_inset = fig.add_axes([0.55, 0.5, 0.35, 0.35])
        for name in names_dim:
            ev = dim[name]["var_explained_pct"]
            ax_inset.plot(range(1, len(ev) + 1), ev, "o-",
                          color=COLORS.get(name, "#CCC"), markersize=4, label=name)
        ax_inset.set_xlabel("PC", fontsize=8)
        ax_inset.set_ylabel("% var", fontsize=8)
        ax_inset.set_title("Eigenvalue spectrum", fontsize=9)
        ax_inset.tick_params(labelsize=7)
        clean_ax(ax_inset)

        save_fig(fig, "cs_fig07_dimensionality.png")

    # ── Fig 8: End-to-end 2x2 grid ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    # (0,0) Within vs cross condition distances (weight space)
    ax = axes[0, 0]
    within_dists = {}
    cross_dists = {}
    for name in ["verbs", "theorems", "bible", "places"]:
        if name not in cond_weights or "baseline" not in cond_weights:
            continue
        wmat = cond_weights[name]
        base_wmat = cond_weights["baseline"]
        wd = []
        for i in range(len(wmat)):
            for j in range(i + 1, len(wmat)):
                wd.append(np.linalg.norm(wmat[i] - wmat[j]))
        cd = []
        for i in range(len(wmat)):
            for j in range(len(base_wmat)):
                cd.append(np.linalg.norm(wmat[i] - base_wmat[j]))
        within_dists[name] = np.mean(wd) if wd else 0
        cross_dists[name] = np.mean(cd)

    names_bar = list(within_dists.keys())
    x_pos = np.arange(len(names_bar))
    w_bar = 0.35
    ax.bar(x_pos - w_bar / 2, [within_dists[n] for n in names_bar], w_bar,
           color=[COLORS[n] for n in names_bar], alpha=0.7, label="Within")
    ax.bar(x_pos + w_bar / 2, [cross_dists[n] for n in names_bar], w_bar,
           color=[COLORS[n] for n in names_bar], alpha=0.3, label="Cross (vs baseline)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_bar)
    ax.set_ylabel("Mean Weight Distance")
    ax.set_title("Within vs Cross-Condition Weight Distances")
    ax.legend()
    clean_ax(ax)

    # (0,1) Triptych triangle
    ax = axes[0, 1]
    if composition_GF.get("triptych"):
        trip = composition_GF["triptych"]
        labels = []
        coords = []
        trip_colors = []
        if "revelation" in trip:
            labels.append(f"Revelation\nDX={trip['revelation']['dx']:.1f}m")
            coords.append([1, 0])
            trip_colors.append("#E24A33")
        if "ecclesiastes" in trip:
            labels.append(f"Ecclesiastes\neff={trip['ecclesiastes']['efficiency']:.5f}")
            coords.append([0, 0])
            trip_colors.append("#348ABD")
        if "noether" in trip:
            labels.append(f"Noether\nDX={trip['noether']['dx']:.3f}m")
            coords.append([0.5, 0.866])
            trip_colors.append("#988ED5")

        if len(coords) == 3:
            coords = np.array(coords)
            # Draw triangle
            triangle = plt.Polygon(coords, fill=False, edgecolor="#AAA",
                                   linewidth=2, linestyle="--")
            ax.add_patch(triangle)
            for i, (lbl, c, col) in enumerate(zip(labels, coords, trip_colors)):
                ax.scatter(c[0], c[1], s=200, c=col, zorder=5, edgecolors="black")
                ax.annotate(lbl, (c[0], c[1]),
                            textcoords="offset points", xytext=(0, 15),
                            ha="center", fontsize=9, fontweight="bold")
            # Edge labels
            ax.annotate("asymmetry", xy=(0.75, 0.433), fontsize=8, color="#888",
                        ha="center", rotation=-60)
            ax.annotate("efficiency", xy=(0.25, 0.433), fontsize=8, color="#888",
                        ha="center", rotation=60)
            ax.annotate("symmetry ← → violence", xy=(0.5, -0.08), fontsize=8,
                        color="#888", ha="center")
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.2, 1.1)
        ax.set_aspect("equal")
        ax.set_title("The Triptych: Structural Transfer")
        ax.axis("off")

    # (1,0) Mantel scatter: sem vs beh
    ax = axes[1, 0]
    if "_scatter_data" in composition_GF:
        sd = composition_GF["_scatter_data"]
        sem_u = np.array(sd["sem_dist_upper"])
        beh_u = np.array(sd["beh_dist_upper"])
        rng = np.random.RandomState(99)
        n_p = len(sem_u)
        idx2 = rng.choice(n_p, min(5000, n_p), replace=False)
        within_m = sem_u[idx2] == 0
        cross_m = sem_u[idx2] == 1
        ax.scatter(sem_u[idx2][within_m] + rng.uniform(-0.05, 0.05, within_m.sum()),
                   beh_u[idx2][within_m], s=3, alpha=0.3, c="#348ABD", label="Within")
        ax.scatter(sem_u[idx2][cross_m] + rng.uniform(-0.05, 0.05, cross_m.sum()),
                   beh_u[idx2][cross_m], s=3, alpha=0.3, c="#E24A33", label="Cross")
        r_sb = composition_GF["mantel_tests"]["sem_vs_beh"]["r"]
        p_sb = composition_GF["mantel_tests"]["sem_vs_beh"]["p"]
        ax.set_xlabel("Semantic Distance")
        ax.set_ylabel("Behavioral Distance (z-scored)")
        ax.set_title(f"End-to-End: Sem vs Beh (r={r_sb:.3f}, p={p_sb:.3f})")
        ax.legend()
        clean_ax(ax)

    # (1,1) Smoothness summary
    ax = axes[1, 1]
    if map_G and "regularizer" in map_G:
        reg = map_G["regularizer"]
        names_r = [n for n in cond_order if n in reg]
        med_cliff = [reg[n]["median_interp_cliffiness"] for n in names_r]
        bars = ax.bar(names_r, med_cliff,
                      color=[COLORS.get(n, "#CCC") for n in names_r], alpha=0.8)
        for bar, mc in zip(bars, med_cliff):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{mc:.1f}", ha="center", fontsize=9)
        ax.set_ylabel("Median Interpolated Cliffiness")
        ax.set_title("LLM Regularization: Cliffiness per Condition")
        if atlas:
            atlas_med = np.median([p["cliffiness"] for p in atlas["probe_results"]])
            ax.axhline(y=atlas_med, color="#AAA", linestyle="--", alpha=0.7,
                       label=f"Atlas median ({atlas_med:.1f})")
            ax.legend()
        clean_ax(ax)

    save_fig(fig, "cs_fig08_endtoend.png")


# ═════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ═════════════════════════════════════════════════════════════════════════════

def print_report(functor_F, map_G, composition_GF, sheaf, info_geo):
    """Print structured console summary."""

    print("\n" + "=" * 74)
    print("FORMAL CATEGORICAL STRUCTURE VALIDATION")
    print("=" * 74)

    # Functor F
    print("\n── FUNCTOR F: Sem → Wt ──────────────────────────────────────────────")

    print(f"\n  Weight Clustering (Faithfulness):")
    print(f"  {'Condition':<12} {'Trials':>6} {'Unique':>6} {'Faith%':>7} {'Largest':>8}")
    print("  " + "-" * 44)
    for name in ["verbs", "theorems", "bible", "places", "baseline"]:
        if name not in functor_F["clustering"]:
            continue
        c = functor_F["clustering"][name]
        print(f"  {name:<12} {c['n_trials']:6d} {c['n_unique']:6d} "
              f"{c['faithfulness'] * 100:6.1f}% {c['largest_cluster']:8d}")

    print(f"\n  Synonym Convergence:")
    for syn_name, sr in functor_F["synonym_convergence"].items():
        status = "IDENTICAL" if sr["all_identical"] else f"dist={sr['mean_within_dist']:.4f}"
        print(f"  {syn_name:<12} n={sr['n_members']:2d}  {status:<20s}  "
              f"bootstrap={sr['bootstrap_mean']:.4f}±{sr['bootstrap_std']:.4f}")

    print(f"\n  Morphism Preservation (within vs cross-condition distances):")
    print(f"  {'Condition':<12} {'WithinWt':>10} {'CrossWt':>10} {'z_wt':>8} "
          f"{'WithinBeh':>10} {'CrossBeh':>10} {'z_beh':>8}")
    print("  " + "-" * 72)
    for name, m in functor_F["morphism_preservation"].items():
        print(f"  {name:<12} {m['within_weight_dist_mean']:10.4f} "
              f"{m['cross_weight_dist_mean']:10.4f} {m['weight_z']:+8.3f}{m['weight_sig']}"
              f" {m['within_beh_dist_mean']:10.4f} {m['cross_beh_dist_mean']:10.4f} "
              f"{m['beh_z']:+8.3f}{m['beh_sig']}")

    print(f"\n  Collapse Analysis (top group per condition):")
    for name, ca in functor_F["collapse_analysis"].items():
        if ca["groups"]:
            g = ca["groups"][0]
            print(f"  {name:<12} {ca['n_collapsed_groups']} collapsed groups, "
                  f"largest={g['n_seeds']} seeds → {g['weights']}")

    # Map G
    if map_G:
        print("\n── MAP G: Wt → Beh ─────────────────────────────────────────────────")

        if "atlas_continuity" in map_G:
            ac = map_G["atlas_continuity"]
            print(f"\n  Atlas Continuity: {ac['n_atlas']} points, "
                  f"median cliffiness={ac['median_cliffiness']:.2f}, "
                  f"mean={ac['mean_cliffiness']:.2f}±{ac['std_cliffiness']:.2f}")
            print(f"  Smooth subcategory: {ac['n_smooth']} points "
                  f"(cliffiness < {ac['smooth_threshold']:.2f})")

        if "regularizer" in map_G:
            print(f"\n  LLM as Regularizer (interpolated cliffiness):")
            print(f"  {'Condition':<12} {'MedCliff':>10} {'vs_base_z':>10} {'Sig':>4}")
            print("  " + "-" * 38)
            for name in ["verbs", "theorems", "bible", "places", "baseline"]:
                r = map_G["regularizer"].get(name)
                if not r:
                    continue
                z_str = f"{r.get('vs_baseline_z', 0):+10.3f}" if "vs_baseline_z" in r else "      —"
                sig_str = r.get("vs_baseline_sig", "")
                print(f"  {name:<12} {r['median_interp_cliffiness']:10.2f} {z_str} {sig_str}")

    # Composition
    print("\n── COMPOSITION G∘F: Sem → Beh ───────────────────────────────────────")

    if "mantel_tests" in composition_GF:
        mt = composition_GF["mantel_tests"]
        print(f"\n  Mantel Tests (1000 permutations):")
        for pair, res in mt.items():
            sig = "***" if res["p"] < 0.01 else "**" if res["p"] < 0.05 else "*" if res["p"] < 0.10 else ""
            print(f"  {pair:<15} r={res['r']:+.4f}  p={res['p']:.4f} {sig}")

    if "triptych" in composition_GF:
        print(f"\n  Triptych Verification:")
        for name, t in composition_GF["triptych"].items():
            checks = [k for k in t if k.startswith("verified_")]
            status = all(t[k] for k in checks)
            print(f"  {name:<15} {'PASS' if status else 'FAIL'}: "
                  f"DX={t.get('dx', 'N/A')}, eff={t.get('efficiency', 'N/A')}")

    if "synonym_behavioral_equivalence" in composition_GF:
        print(f"\n  Synonym Behavioral Equivalence:")
        for syn_name, sbe in composition_GF["synonym_behavioral_equivalence"].items():
            wt_status = "wt=IDENT" if sbe["weights_identical"] else f"wt_diff={sbe['max_weight_diff']:.6f}"
            beh_status = "beh=IDENT" if sbe["behavior_identical"] else f"beh_diff={sbe['max_behavior_diff']:.6f}"
            print(f"  {syn_name:<12} n={sbe['n_matched']:2d}  {wt_status:<20s}  {beh_status}")

    # Sheaf
    if sheaf:
        print("\n── SHEAF STRUCTURE ──────────────────────────────────────────────────")
        if "patch_identification" in sheaf:
            pi = sheaf["patch_identification"]
            print(f"\n  Patches: {pi['n_patches']} connected components "
                  f"from {pi['n_smooth_points']} smooth points (radius={pi['radius']})")
            print(f"  Top patch sizes: {pi['patch_sizes'][:10]}")

        if "llm_patch_selection" in sheaf:
            print(f"\n  LLM Patch Selection:")
            for name, lp in sheaf["llm_patch_selection"].items():
                print(f"  {name:<12} {lp['n_patches_used']} patches used, "
                      f"mean dist to smooth={lp['mean_dist_to_smooth']:.4f}")

    # Information Geometry
    if info_geo:
        print("\n── INFORMATION GEOMETRY ─────────────────────────────────────────────")
        if "dimensionality" in info_geo:
            print(f"\n  Effective Dimensionality (participation ratio):")
            for name in ["verbs", "theorems", "bible", "places", "baseline"]:
                if name not in info_geo["dimensionality"]:
                    continue
                d = info_geo["dimensionality"][name]
                ev_str = ", ".join(f"{v:.1f}%" for v in d["var_explained_pct"][:3])
                print(f"  {name:<12} PR={d['participation_ratio']:.2f}  "
                      f"top-3 PCs: [{ev_str}]")

        if "distributions" in info_geo and "mahalanobis" in info_geo["distributions"]:
            print(f"\n  Mahalanobis Distances Between Condition Centroids:")
            for pair, d in info_geo["distributions"]["mahalanobis"].items():
                print(f"  {pair:<25} {d:.3f}")

    print("\n" + "=" * 74)
    print("VALIDATION COMPLETE")
    print("=" * 74 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 74)
    print("CATEGORICAL STRUCTURE VALIDATION — Data Loading")
    print("=" * 74 + "\n")

    # Phase 0: Load data
    data, atlas, taxonomy = load_all_data()
    if len(data) < 2:
        print("Need at least 2 conditions. Aborting.")
        return

    cond_weights, cond_beh, cond_seeds = extract_matrices(data)
    total_trials = sum(len(v) for v in data.values())
    print(f"\n  Total: {total_trials} trials across {len(data)} conditions")

    # Phase 1: Semantic distances (used internally)
    all_seeds, all_conds, cond_dist, syn_dist, seed_to_syn = build_semantic_distance(
        cond_seeds, cond_weights)
    print(f"  Semantic distances: {len(all_seeds)} seeds, "
          f"{len(seed_to_syn)} synonym-tagged")

    # Phase 2: Functor F
    print("\n  Phase 2: Functor F: Sem → Wt...")
    functor_F = phase2_functor_F(cond_weights, cond_beh, cond_seeds, data)

    # Phase 3: Map G
    print("  Phase 3: Map G: Wt → Beh...")
    map_G = phase3_map_G(cond_weights, atlas, taxonomy)

    # Phase 4: Composition
    print("  Phase 4: Composition G∘F...")
    composition_GF = phase4_composition(cond_weights, cond_beh, cond_seeds, data)

    # Phase 5: Sheaf structure
    print("  Phase 5: Sheaf structure...")
    sheaf = phase5_sheaf(cond_weights, atlas)

    # Phase 6: Information geometry
    print("  Phase 6: Information geometry...")
    info_geo = phase6_info_geometry(cond_weights)

    # Phase 8: Output

    # Figures first (before stripping internal data)
    generate_figures(cond_weights, cond_beh, cond_seeds, data, atlas,
                     functor_F, map_G, composition_GF, sheaf, info_geo)

    # Build results dict, filtering internal keys
    results = {
        "functor_F": functor_F,
        "map_G": {},
        "composition_GF": {k: v for k, v in composition_GF.items()
                           if not k.startswith("_")},
        "sheaf_structure": {k: v for k, v in sheaf.items()
                            if not k.startswith("_")},
        "information_geometry": info_geo,
    }

    # Deep-copy map_G, stripping large arrays
    results["map_G"] = copy.deepcopy(map_G)
    if "regularizer" in results["map_G"]:
        for name in results["map_G"]["regularizer"]:
            results["map_G"]["regularizer"][name].pop("interp_cliffiness_values", None)

    out_path = PROJECT / "artifacts" / "categorical_structure_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  WROTE {out_path}")

    # Console report
    print_report(functor_F, map_G, composition_GF, sheaf, info_geo)


if __name__ == "__main__":
    main()
