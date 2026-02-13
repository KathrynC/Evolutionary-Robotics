#!/usr/bin/env python3
"""
analyze_celebrities.py

Analysis of the celebrity/public figure structured random search experiment.
Examines weight clustering across 12 domains (politics, entertainment, tech,
sports, musicians, Kardashians, historical, etc.), identifies archetypes,
and compares with other structured conditions.

Produces:
  - Console report with domain statistics, clustering, significance tests
  - artifacts/structured_random_celebrities_analysis.json
  - 8 figures in artifacts/plots/cel_fig01-08_*.png

Usage:
    python3 analyze_celebrities.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from compute_beer_analytics import NumpyEncoder

PLOT_DIR = PROJECT / "artifacts" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Domain colors — 12 domains
DOMAIN_COLORS = {
    "trump_family":  "#E24A33",
    "trump_admin":   "#C44E52",
    "us_politics":   "#348ABD",
    "international": "#467821",
    "controversial": "#8B0000",
    "kardashian":    "#FF69B4",
    "tech":          "#00CED1",
    "musician":      "#FFD700",
    "entertainment": "#FF8C00",
    "sports":        "#32CD32",
    "cultural":      "#9370DB",
    "historical":    "#8B4513",
}

# Broader super-categories for higher-level analysis
SUPER_CATEGORIES = {
    "politics": ["trump_family", "trump_admin", "us_politics", "international", "controversial"],
    "pop_culture": ["kardashian", "musician", "entertainment", "sports"],
    "intellectual": ["tech", "cultural", "historical"],
}

OTHER_CONDITION_FILES = {
    "verbs":    PROJECT / "artifacts" / "structured_random_verbs.json",
    "theorems": PROJECT / "artifacts" / "structured_random_theorems.json",
    "bible":    PROJECT / "artifacts" / "structured_random_bible.json",
    "places":   PROJECT / "artifacts" / "structured_random_places.json",
    "baseline": PROJECT / "artifacts" / "structured_random_baseline.json",
    "politics": PROJECT / "artifacts" / "structured_random_politics.json",
}

OTHER_COLORS = {
    "verbs":       "#FFA07A",
    "theorems":    "#87CEEB",
    "bible":       "#DDA0DD",
    "places":      "#90EE90",
    "baseline":    "#AAAAAA",
    "politics":    "#FFD700",
    "celebrities": "#FF4500",
}

WEIGHT_KEYS = ["w03", "w04", "w13", "w14", "w23", "w24"]


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


def mann_whitney_u(x, y):
    """Mann-Whitney U test (numpy-only)."""
    x, y = np.array(x), np.array(y)
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
    z = (u1 - mu) / sigma if sigma > 0 else 0
    return u1, z


def load_celebrities():
    """Load celebrity results and split by domain."""
    path = PROJECT / "artifacts" / "structured_random_celebrities.json"
    with open(path) as f:
        trials = json.load(f)

    domains = {d: [] for d in DOMAIN_COLORS}
    for r in trials:
        seed = r["seed"]
        for d in domains:
            if f"[{d}]" in seed:
                r["domain"] = d
                r["name"] = seed.split(" [")[0]
                domains[d].append(r)
                break
    return trials, domains


def load_other_conditions():
    data = {}
    for name, path in OTHER_CONDITION_FILES.items():
        if path.exists():
            with open(path) as f:
                data[name] = json.load(f)
    return data


def main():
    print("\n" + "=" * 70)
    print("CELEBRITY / PUBLIC FIGURE: STRUCTURED RANDOM ANALYSIS")
    print("=" * 70)

    trials, domains = load_celebrities()
    other = load_other_conditions()

    domain_names = [d for d in DOMAIN_COLORS if domains[d]]  # only non-empty

    # ── 1. Overview ──────────────────────────────────────────────────────────
    print(f"\nTotal trials: {len(trials)}")
    for d in domain_names:
        print(f"  {d:<20} {len(domains[d]):3d} trials")

    # ── 2. Per-domain statistics ─────────────────────────────────────────────
    print(f"\n{'Domain':<20} {'N':>3} {'Dead%':>6} {'Med|DX|':>8} {'Max|DX|':>8} "
          f"{'MeanSpd':>8} {'MeanPL':>7}")
    print("-" * 76)

    domain_metrics = {}
    for d in domain_names:
        items = domains[d]
        abs_dx = [abs(r["dx"]) for r in items]
        dead = sum(1 for v in abs_dx if v < 1.0)
        m = {
            "n": len(items),
            "abs_dx": abs_dx,
            "dead_frac": dead / len(items) if items else 0,
            "median_dx": float(np.median(abs_dx)),
            "max_dx": float(max(abs_dx)),
            "mean_speed": float(np.mean([r["speed"] for r in items])),
            "mean_phase_lock": float(np.mean([r["phase_lock"] for r in items])),
            "mean_efficiency": float(np.mean([r["efficiency"] for r in items])),
        }
        domain_metrics[d] = m
        print(f"{d:<20} {m['n']:3d} {m['dead_frac']*100:5.1f}% {m['median_dx']:8.2f} "
              f"{m['max_dx']:8.2f} {m['mean_speed']:8.3f} {m['mean_phase_lock']:7.3f}")

    # ── 3. Weight clustering ─────────────────────────────────────────────────
    print("\n── WEIGHT CLUSTERING ──")

    weight_tuples = []
    for r in trials:
        w = tuple(round(r["weights"][k], 1) for k in WEIGHT_KEYS)
        weight_tuples.append(w)

    unique_weights = set(weight_tuples)
    faithfulness = len(unique_weights) / len(trials)
    print(f"  Unique weight vectors: {len(unique_weights)} / {len(trials)} "
          f"(faithfulness = {faithfulness:.3f})")

    cluster_counts = Counter(weight_tuples)
    cluster_sizes = sorted(cluster_counts.values(), reverse=True)
    print(f"  Cluster sizes: {cluster_sizes[:20]}{'...' if len(cluster_sizes) > 20 else ''}")

    # Show largest clusters with members
    print(f"\n  Largest weight clusters:")
    for wt, count in cluster_counts.most_common(12):
        members = [r["name"] for r, t in zip(trials, weight_tuples) if t == wt]
        member_domains = [r["domain"] for r, t in zip(trials, weight_tuples) if t == wt]
        domain_dist = Counter(member_domains)
        w_str = " ".join(f"{k}={v:+.1f}" for k, v in zip(WEIGHT_KEYS, wt))
        dx_val = next(r["dx"] for r, t in zip(trials, weight_tuples) if t == wt)
        domain_str = ", ".join(f"{d}:{c}" for d, c in domain_dist.most_common(4))
        print(f"    [{count:3d}] ({w_str})  DX={dx_val:+.2f}")
        print(f"           domains: {domain_str}")
        member_str = ", ".join(members[:8])
        if len(members) > 8:
            member_str += f"... +{len(members)-8}"
        print(f"           e.g.: {member_str}")

    # Per-domain faithfulness
    print(f"\n  Per-domain faithfulness:")
    for d in domain_names:
        items = domains[d]
        d_wts = set()
        for r in items:
            w = tuple(round(r["weights"][k], 1) for k in WEIGHT_KEYS)
            d_wts.add(w)
        f_ratio = len(d_wts) / len(items) if items else 0
        print(f"    {d:<20} {len(d_wts):3d} / {len(items):3d} = {f_ratio:.3f}")

    # ── 4. Super-category analysis ───────────────────────────────────────────
    print("\n── SUPER-CATEGORY ANALYSIS ──")
    for supercat, cat_domains in SUPER_CATEGORIES.items():
        cat_items = []
        for d in cat_domains:
            cat_items.extend(domains.get(d, []))
        if not cat_items:
            continue
        cat_abs_dx = [abs(r["dx"]) for r in cat_items]
        cat_wts = set()
        for r in cat_items:
            w = tuple(round(r["weights"][k], 1) for k in WEIGHT_KEYS)
            cat_wts.add(w)
        print(f"  {supercat:<15} N={len(cat_items):3d}  unique_wt={len(cat_wts):3d}  "
              f"faithfulness={len(cat_wts)/len(cat_items):.3f}  "
              f"med|DX|={np.median(cat_abs_dx):.2f}  max|DX|={max(cat_abs_dx):.2f}")

    # ── 5. Cross-domain cluster overlap ──────────────────────────────────────
    print("\n── CROSS-DOMAIN CLUSTER OVERLAP ──")
    print("  Which clusters span multiple domains?")
    for wt, count in cluster_counts.most_common(8):
        members = [(r["name"], r["domain"]) for r, t in zip(trials, weight_tuples) if t == wt]
        domain_dist = Counter(d for _, d in members)
        if len(domain_dist) >= 3:
            print(f"    Cluster (n={count}): spans {len(domain_dist)} domains — "
                  + ", ".join(f"{d}:{c}" for d, c in domain_dist.most_common()))

    # ── 6. Top and bottom gaits ──────────────────────────────────────────────
    print("\n── TOP 15 GAITS (by |DX|) ──")
    sorted_trials = sorted(trials, key=lambda r: abs(r["dx"]), reverse=True)
    for i, r in enumerate(sorted_trials[:15]):
        print(f"  {i+1:2d}. {r['name']:<25} [{r['domain']:<15}]  DX={r['dx']:+7.2f}  "
              f"speed={r['speed']:.3f}  PL={r['phase_lock']:.3f}")

    print("\n── BOTTOM 10 GAITS (by |DX|) ──")
    for i, r in enumerate(sorted_trials[-10:]):
        print(f"  {len(sorted_trials)-9+i:2d}. {r['name']:<25} [{r['domain']:<15}]  "
              f"DX={r['dx']:+7.2f}")

    # ── 7. Direction analysis ────────────────────────────────────────────────
    print("\n── DIRECTION ANALYSIS ──")
    for d in domain_names:
        items = domains[d]
        pos = sum(1 for r in items if r["dx"] > 0)
        neg = sum(1 for r in items if r["dx"] < 0)
        print(f"  {d:<20}  forward: {pos:2d}  backward: {neg:2d}")

    # ── 8. Comparison with other conditions ──────────────────────────────────
    print("\n── COMPARISON WITH OTHER CONDITIONS ──")
    cel_abs_dx = [abs(r["dx"]) for r in trials]
    cel_pl = [r["phase_lock"] for r in trials]

    print(f"\n  {'Condition':<14} {'N':>4} {'Dead%':>6} {'Med|DX|':>8} {'Max|DX|':>8} "
          f"{'MeanPL':>7}  {'Faithful':>8}  {'U-test z':>8}")
    print("  " + "-" * 80)

    # Celebrities row
    dead_cel = sum(1 for d in cel_abs_dx if d < 1.0)
    print(f"  {'celebrities':<14} {len(trials):4d} {dead_cel/len(trials)*100:5.1f}% "
          f"{np.median(cel_abs_dx):8.2f} {max(cel_abs_dx):8.2f} "
          f"{np.mean(cel_pl):7.3f}  {faithfulness:8.3f}  {'(self)':>8}")

    for cond_name in ["politics", "verbs", "theorems", "bible", "places", "baseline"]:
        if cond_name not in other:
            continue
        cond_trials = other[cond_name]
        cond_abs_dx = [abs(r["dx"]) for r in cond_trials]
        cond_pl = [r["phase_lock"] for r in cond_trials]
        dead = sum(1 for d in cond_abs_dx if d < 1.0)
        # Compute faithfulness for this condition
        cond_wts = set()
        for r in cond_trials:
            w = tuple(round(r["weights"][k], 1) for k in WEIGHT_KEYS)
            cond_wts.add(w)
        cond_faith = len(cond_wts) / len(cond_trials) if cond_trials else 0
        _, z = mann_whitney_u(cel_abs_dx, cond_abs_dx)
        sig = "***" if abs(z) > 2.58 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.64 else ""
        print(f"  {cond_name:<14} {len(cond_trials):4d} {dead/len(cond_trials)*100:5.1f}% "
              f"{np.median(cond_abs_dx):8.2f} {max(cond_abs_dx):8.2f} "
              f"{np.mean(cond_pl):7.3f}  {cond_faith:8.3f}  {z:+8.3f} {sig}")

    # ── FIGURES ──────────────────────────────────────────────────────────────
    print("\n── GENERATING FIGURES ──")

    # Fig 1: DX by domain (box plot)
    fig, ax = plt.subplots(figsize=(16, 7))
    box_data = [domain_metrics[d]["abs_dx"] for d in domain_names]
    bp = ax.boxplot(box_data, labels=[d.replace("_", "\n") for d in domain_names],
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=4, alpha=0.5))
    for patch, d in zip(bp["boxes"], domain_names):
        patch.set_facecolor(DOMAIN_COLORS[d])
        patch.set_alpha(0.7)
    for i, d in enumerate(domain_names):
        jitter = np.random.normal(0, 0.05, len(domain_metrics[d]["abs_dx"]))
        ax.scatter(np.full(len(domain_metrics[d]["abs_dx"]), i + 1) + jitter,
                   domain_metrics[d]["abs_dx"],
                   c=DOMAIN_COLORS[d], s=12, alpha=0.4, zorder=3)
    ax.set_ylabel("|DX| (meters)")
    ax.set_title("Displacement by Celebrity Domain")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cel_fig01_dx_by_domain.png")

    # Fig 2: Weight space PCA colored by domain
    fig, ax = plt.subplots(figsize=(12, 9))
    all_weights = np.array([[r["weights"][k] for k in WEIGHT_KEYS] for r in trials])
    w_mu = all_weights.mean(axis=0)
    w_std = all_weights.std(axis=0)
    w_std[w_std < 1e-12] = 1
    w_Z = (all_weights - w_mu) / w_std
    w_U, w_S, w_Vt = np.linalg.svd(w_Z, full_matrices=False)
    w_pc = w_Z @ w_Vt[:2].T
    w_var = w_S[:2]**2 / (w_S**2).sum() * 100

    for d in domain_names:
        mask = [i for i, r in enumerate(trials) if r.get("domain") == d]
        ax.scatter(w_pc[mask, 0], w_pc[mask, 1],
                   c=DOMAIN_COLORS[d], s=30, alpha=0.7, label=d.replace("_", " "),
                   edgecolors="white", linewidths=0.3)

    # Label outliers
    for i, r in enumerate(trials):
        if abs(r["dx"]) > 3.5 or r["name"] in ["Julian Assange", "Edward Snowden",
                                                  "Donald Trump", "Elon Musk",
                                                  "Kim Kardashian", "Beyonce",
                                                  "Albert Einstein"]:
            ax.annotate(r["name"], (w_pc[i, 0], w_pc[i, 1]),
                        fontsize=6, alpha=0.7, ha="left",
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({w_var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({w_var[1]:.1f}% var)")
    ax.set_title("Weight Space PCA — Celebrity Names by Domain")
    ax.legend(fontsize=7, ncol=3, loc="best")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cel_fig02_weight_pca.png")

    # Fig 3: Weight heatmap — all figures, sorted by domain then DX
    sorted_all = []
    for d in domain_names:
        d_items = sorted(domains[d], key=lambda r: r["dx"], reverse=True)
        sorted_all.extend(d_items)

    fig, ax = plt.subplots(figsize=(8, max(20, len(sorted_all) * 0.22)))
    weight_matrix = np.array([[r["weights"][k] for k in WEIGHT_KEYS] for r in sorted_all])
    names_sorted = [r["name"] for r in sorted_all]

    im = ax.imshow(weight_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(WEIGHT_KEYS, fontsize=10)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=4)
    ax.set_title("Synapse Weights by Public Figure (sorted by domain, then DX)")

    # Domain separators
    cum = 0
    for d in domain_names:
        n_d = len(domains[d])
        if n_d == 0:
            continue
        if cum > 0:
            ax.axhline(cum - 0.5, color="white", linewidth=1.5)
        ax.text(-0.7, cum + n_d / 2, d.replace("_", "\n"),
                fontsize=6, fontweight="bold", ha="right", va="center",
                color=DOMAIN_COLORS[d])
        cum += n_d

    fig.colorbar(im, ax=ax, shrink=0.3, label="Weight value")
    fig.tight_layout()
    save_fig(fig, "cel_fig03_weight_heatmap.png")

    # Fig 4: Cluster composition — domain breakdown of top clusters
    fig, ax = plt.subplots(figsize=(14, 7))
    top_clusters = cluster_counts.most_common(15)
    x_pos = np.arange(len(top_clusters))
    bottoms = np.zeros(len(top_clusters))

    # Build cluster labels
    c_labels = []
    for wt, count in top_clusters:
        dx_val = next(r["dx"] for r, t in zip(trials, weight_tuples) if t == wt)
        c_labels.append(f"n={count}\nDX={dx_val:+.1f}")

    for d in domain_names:
        heights = []
        for wt, count in top_clusters:
            members = [r for r, t in zip(trials, weight_tuples) if t == wt]
            d_count = sum(1 for r in members if r.get("domain") == d)
            heights.append(d_count)
        ax.bar(x_pos, heights, bottom=bottoms, color=DOMAIN_COLORS[d],
               label=d.replace("_", " "), alpha=0.8)
        bottoms += heights

    ax.set_xticks(x_pos)
    ax.set_xticklabels(c_labels, fontsize=7, ha="center")
    ax.set_ylabel("Number of figures in cluster")
    ax.set_title("Top 15 Weight Clusters — Domain Composition")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cel_fig04_cluster_composition.png")

    # Fig 5: Super-category comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_i, (supercat, cat_domains) in enumerate(SUPER_CATEGORIES.items()):
        ax = axes[ax_i]
        cat_items = []
        for d in cat_domains:
            cat_items.extend(domains.get(d, []))
        if not cat_items:
            continue
        cat_abs_dx = [abs(r["dx"]) for r in cat_items]
        cat_speeds = [r["speed"] for r in cat_items]
        cat_pl = [r["phase_lock"] for r in cat_items]

        # Color by sub-domain
        for d in cat_domains:
            d_items = domains.get(d, [])
            if not d_items:
                continue
            ax.scatter([abs(r["dx"]) for r in d_items],
                       [r["phase_lock"] for r in d_items],
                       c=DOMAIN_COLORS.get(d, "#999"),
                       s=25, alpha=0.7, label=d.replace("_", " "),
                       edgecolors="white", linewidths=0.3)
        ax.set_xlabel("|DX| (meters)")
        ax.set_ylabel("Phase Lock")
        ax.set_title(f"{supercat.replace('_', ' ').title()}")
        ax.legend(fontsize=6)
        clean_ax(ax)

    fig.suptitle("Coordination vs Displacement by Super-Category", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "cel_fig05_supercategory_scatter.png")

    # Fig 6: Faithfulness by domain (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    domain_faithfulness = {}
    for d in domain_names:
        items = domains[d]
        d_wts = set()
        for r in items:
            w = tuple(round(r["weights"][k], 1) for k in WEIGHT_KEYS)
            d_wts.add(w)
        domain_faithfulness[d] = len(d_wts) / len(items) if items else 0

    bars = ax.bar(range(len(domain_names)),
                  [domain_faithfulness[d] for d in domain_names],
                  color=[DOMAIN_COLORS[d] for d in domain_names],
                  alpha=0.8)
    ax.set_xticks(range(len(domain_names)))
    ax.set_xticklabels([d.replace("_", "\n") for d in domain_names], fontsize=8)
    ax.set_ylabel("Faithfulness (unique weights / total)")
    ax.set_title("Faithfulness by Celebrity Domain")
    ax.axhline(faithfulness, color="black", linestyle="--", alpha=0.5,
               label=f"Overall: {faithfulness:.3f}")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cel_fig06_faithfulness_by_domain.png")

    # Fig 7: Cross-condition PCA (celebrities + all other conditions)
    all_vecs = []
    all_labels = []

    for r in trials:
        all_vecs.append([r["speed"], r["efficiency"], r["phase_lock"], r["entropy"]])
        all_labels.append("celebrities")

    for cond_name, cond_trials in other.items():
        for r in cond_trials:
            all_vecs.append([r["speed"], r["efficiency"], r["phase_lock"], r["entropy"]])
            all_labels.append(cond_name)

    all_vecs = np.array(all_vecs)
    mu_v = all_vecs.mean(axis=0)
    std_v = all_vecs.std(axis=0)
    std_v[std_v < 1e-12] = 1
    Z = (all_vecs - mu_v) / std_v
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pc = Z @ Vt[:2].T
    var_exp = S[:2]**2 / (S**2).sum() * 100

    fig, ax = plt.subplots(figsize=(12, 9))
    offset = 0
    # Plot celebrities first
    n_cel = len(trials)
    # Plot other conditions as background
    idx = n_cel
    for cond_name in ["baseline", "politics", "verbs", "theorems", "bible", "places"]:
        if cond_name not in other:
            continue
        n_c = len(other[cond_name])
        c_idx = list(range(idx, idx + n_c))
        ax.scatter(pc[c_idx, 0], pc[c_idx, 1],
                   c=OTHER_COLORS.get(cond_name, "#CCC"), s=10, alpha=0.25,
                   label=cond_name, edgecolors="none")
        idx += n_c

    # Plot celebrities on top, colored by domain
    for d in domain_names:
        d_mask = [i for i, r in enumerate(trials) if r.get("domain") == d]
        ax.scatter(pc[d_mask, 0], pc[d_mask, 1],
                   c=DOMAIN_COLORS[d], s=25, alpha=0.7,
                   label=f"cel:{d.replace('_',' ')}", edgecolors="white",
                   linewidths=0.3, marker="D")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
    ax.set_title("Behavioral PCA — Celebrities vs All Conditions")
    ax.legend(fontsize=6, ncol=3, loc="best")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cel_fig07_cross_condition_pca.png")

    # Fig 8: Effective dimensionality comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    dims = {}
    # Celebrity condition
    cel_weights = np.array([[r["weights"][k] for k in WEIGHT_KEYS] for r in trials])
    cel_mu = cel_weights.mean(axis=0)
    cel_std_v = cel_weights.std(axis=0)
    cel_std_v[cel_std_v < 1e-12] = 1
    cel_Z = (cel_weights - cel_mu) / cel_std_v
    _, cel_S, _ = np.linalg.svd(cel_Z, full_matrices=False)
    cel_eigs = cel_S**2
    dims["celebrities"] = float((cel_eigs.sum())**2 / (cel_eigs**2).sum())

    for cond_name, cond_trials in other.items():
        cw = np.array([[r["weights"][k] for k in WEIGHT_KEYS] for r in cond_trials])
        cmu = cw.mean(axis=0)
        cstd = cw.std(axis=0)
        cstd[cstd < 1e-12] = 1
        cZ = (cw - cmu) / cstd
        _, cS, _ = np.linalg.svd(cZ, full_matrices=False)
        ceigs = cS**2
        dims[cond_name] = float((ceigs.sum())**2 / (ceigs**2).sum())

    cond_order = ["celebrities", "politics", "verbs", "theorems", "bible", "places", "baseline"]
    cond_order = [c for c in cond_order if c in dims]
    colors_list = [OTHER_COLORS.get(c, "#FF4500") for c in cond_order]
    ax.bar(range(len(cond_order)),
           [dims[c] for c in cond_order],
           color=colors_list, alpha=0.8)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_order, fontsize=9)
    ax.set_ylabel("Effective Dimensionality (Participation Ratio)")
    ax.set_title("Effective Dimensionality by Condition")
    ax.axhline(6.0, color="gray", linestyle=":", alpha=0.5, label="Max (6D)")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "cel_fig08_dimensionality.png")

    # ── Save analysis JSON ───────────────────────────────────────────────────
    analysis = {
        "n_total": len(trials),
        "n_unique_weights": len(unique_weights),
        "faithfulness": faithfulness,
        "domain_metrics": {
            d: {k: v for k, v in m.items() if k != "abs_dx"}
            for d, m in domain_metrics.items()
        },
        "domain_faithfulness": domain_faithfulness,
        "cluster_sizes": cluster_sizes,
        "top_clusters": [
            {
                "weights": dict(zip(WEIGHT_KEYS, [float(v) for v in wt])),
                "count": count,
                "members": [r["name"] for r, t in zip(trials, weight_tuples) if t == wt],
                "domains": dict(Counter(
                    r["domain"] for r, t in zip(trials, weight_tuples) if t == wt
                )),
                "dx": float(next(r["dx"] for r, t in zip(trials, weight_tuples) if t == wt)),
            }
            for wt, count in cluster_counts.most_common(20)
        ],
        "top_15_gaits": [
            {"name": r["name"], "domain": r["domain"], "dx": r["dx"],
             "weights": r["weights"], "speed": r["speed"],
             "phase_lock": r["phase_lock"]}
            for r in sorted_trials[:15]
        ],
        "effective_dimensionality": dims,
        "super_categories": {
            supercat: {
                "n": sum(len(domains.get(d, [])) for d in cat_domains),
                "domains": cat_domains,
            }
            for supercat, cat_domains in SUPER_CATEGORIES.items()
        },
    }

    out_path = PROJECT / "artifacts" / "structured_random_celebrities_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    print(f"\n  WROTE {out_path}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
