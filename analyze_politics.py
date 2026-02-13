#!/usr/bin/env python3
"""
analyze_politics.py

Analysis of the political figures structured random search experiment.
Examines weight clustering, group relationships, behavioral phenotypes,
and comparison with other structured conditions.

Produces:
  - Console report with group statistics, clustering, significance tests
  - artifacts/structured_random_politics_analysis.json
  - 6 figures in artifacts/plots/pol_fig01-06_*.png

Usage:
    python3 analyze_politics.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from compute_beer_analytics import NumpyEncoder

PLOT_DIR = PROJECT / "artifacts" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Group colors
GROUP_COLORS = {
    "family":     "#E24A33",   # warm red
    "admin":      "#348ABD",   # cool blue
    "adjacent":   "#8EBA42",   # green
    "opposition": "#988ED5",   # purple
}

OTHER_CONDITION_FILES = {
    "verbs":    PROJECT / "artifacts" / "structured_random_verbs.json",
    "theorems": PROJECT / "artifacts" / "structured_random_theorems.json",
    "bible":    PROJECT / "artifacts" / "structured_random_bible.json",
    "places":   PROJECT / "artifacts" / "structured_random_places.json",
    "baseline": PROJECT / "artifacts" / "structured_random_baseline.json",
}

OTHER_COLORS = {
    "verbs":    "#FFA07A",
    "theorems": "#87CEEB",
    "bible":    "#DDA0DD",
    "places":   "#90EE90",
    "baseline": "#AAAAAA",
    "politics": "#FFD700",
}


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


def load_politics():
    """Load politics results and split by group."""
    path = PROJECT / "artifacts" / "structured_random_politics.json"
    with open(path) as f:
        trials = json.load(f)

    groups = {"family": [], "admin": [], "adjacent": [], "opposition": []}
    for r in trials:
        seed = r["seed"]
        for g in groups:
            if f"[{g}]" in seed:
                r["group"] = g
                r["name"] = seed.split(" [")[0]
                groups[g].append(r)
                break
    return trials, groups


def load_other_conditions():
    """Load results from all other structured random conditions."""
    data = {}
    for name, path in OTHER_CONDITION_FILES.items():
        if path.exists():
            with open(path) as f:
                data[name] = json.load(f)
    return data


def main():
    print("\n" + "=" * 70)
    print("POLITICAL FIGURES: STRUCTURED RANDOM ANALYSIS")
    print("=" * 70)

    trials, groups = load_politics()
    other = load_other_conditions()

    # ── 1. Overview ──────────────────────────────────────────────────────────
    print(f"\nTotal trials: {len(trials)}")
    for g, items in groups.items():
        print(f"  {g:<12} {len(items):3d} trials")

    # ── 2. Per-group statistics ──────────────────────────────────────────────
    print(f"\n{'Group':<12} {'N':>3} {'Dead%':>6} {'Med|DX|':>8} {'Max|DX|':>8} "
          f"{'MeanSpd':>8} {'MeanPL':>7} {'MeanEff':>9}")
    print("-" * 72)

    group_metrics = {}
    for g in ["family", "admin", "adjacent", "opposition"]:
        items = groups[g]
        abs_dx = [abs(r["dx"]) for r in items]
        dead = sum(1 for d in abs_dx if d < 1.0)
        m = {
            "n": len(items),
            "abs_dx": abs_dx,
            "dead_frac": dead / len(items) if items else 0,
            "median_dx": float(np.median(abs_dx)),
            "max_dx": max(abs_dx),
            "mean_speed": float(np.mean([r["speed"] for r in items])),
            "mean_phase_lock": float(np.mean([r["phase_lock"] for r in items])),
            "mean_efficiency": float(np.mean([r["efficiency"] for r in items])),
        }
        group_metrics[g] = m
        print(f"{g:<12} {m['n']:3d} {m['dead_frac']*100:5.1f}% {m['median_dx']:8.2f} "
              f"{m['max_dx']:8.2f} {m['mean_speed']:8.3f} {m['mean_phase_lock']:7.3f} "
              f"{m['mean_efficiency']:9.5f}")

    # ── 3. Weight clustering ─────────────────────────────────────────────────
    print("\n── WEIGHT CLUSTERING ──")

    # Round weights to 1 decimal place for clustering (matches LLM output precision)
    weight_keys = ["w03", "w04", "w13", "w14", "w23", "w24"]
    weight_tuples = []
    for r in trials:
        w = tuple(round(r["weights"][k], 1) for k in weight_keys)
        weight_tuples.append(w)

    unique_weights = set(weight_tuples)
    faithfulness = len(unique_weights) / len(trials)
    print(f"  Unique weight vectors: {len(unique_weights)} / {len(trials)} "
          f"(faithfulness = {faithfulness:.3f})")

    # Count cluster sizes
    cluster_counts = Counter(weight_tuples)
    cluster_sizes = sorted(cluster_counts.values(), reverse=True)
    print(f"  Cluster sizes: {cluster_sizes[:15]}{'...' if len(cluster_sizes) > 15 else ''}")

    # Show the biggest clusters and their members
    print(f"\n  Largest weight clusters:")
    for wt, count in cluster_counts.most_common(8):
        members = [r["name"] for r, t in zip(trials, weight_tuples) if t == wt]
        w_str = " ".join(f"{k}={v:+.1f}" for k, v in zip(weight_keys, wt))
        # Get DX for this weight vector
        dx_val = next(r["dx"] for r, t in zip(trials, weight_tuples) if t == wt)
        print(f"    [{count:2d}] ({w_str})  DX={dx_val:+.2f}  members: {', '.join(members[:10])}"
              + (f"... +{len(members)-10}" if len(members) > 10 else ""))

    # Per-group faithfulness
    print(f"\n  Per-group faithfulness:")
    for g in ["family", "admin", "adjacent", "opposition"]:
        items = groups[g]
        group_wts = set()
        for r in items:
            w = tuple(round(r["weights"][k], 1) for k in weight_keys)
            group_wts.add(w)
        f_ratio = len(group_wts) / len(items) if items else 0
        print(f"    {g:<12} {len(group_wts):3d} / {len(items):3d} = {f_ratio:.3f}")

    # ── 4. Cross-group weight distances ──────────────────────────────────────
    print("\n── CROSS-GROUP WEIGHT DISTANCES ──")

    # Build weight matrix per group
    group_weight_matrices = {}
    for g in ["family", "admin", "adjacent", "opposition"]:
        items = groups[g]
        mat = np.array([[r["weights"][k] for k in weight_keys] for r in items])
        group_weight_matrices[g] = mat

    # Mean within-group and cross-group distances
    group_names = ["family", "admin", "adjacent", "opposition"]
    print(f"\n  {'':>12}", end="")
    for g2 in group_names:
        print(f"  {g2:>12}", end="")
    print()
    print("  " + "-" * 62)

    distance_matrix = {}
    for g1 in group_names:
        print(f"  {g1:<12}", end="")
        for g2 in group_names:
            m1, m2 = group_weight_matrices[g1], group_weight_matrices[g2]
            # All pairwise distances
            dists = []
            for i in range(len(m1)):
                for j in range(len(m2)):
                    if g1 == g2 and i == j:
                        continue
                    dists.append(np.linalg.norm(m1[i] - m2[j]))
            mean_d = np.mean(dists) if dists else 0
            distance_matrix[(g1, g2)] = mean_d
            print(f"  {mean_d:12.4f}", end="")
        print()

    # ── 5. Pairwise significance tests between groups ────────────────────────
    print("\n── MANN-WHITNEY U TESTS (|DX|) BETWEEN GROUPS ──")
    print(f"  {'Pair':<28} {'z':>8}  {'sig'}")
    print("  " + "-" * 42)
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i+1:]:
            _, z = mann_whitney_u(group_metrics[g1]["abs_dx"], group_metrics[g2]["abs_dx"])
            sig = "***" if abs(z) > 2.58 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.64 else ""
            print(f"  {g1+' vs '+g2:<28} {z:+8.3f}  {sig}")

    # ── 6. Best and worst gaits ──────────────────────────────────────────────
    print("\n── TOP 10 GAITS (by |DX|) ──")
    sorted_trials = sorted(trials, key=lambda r: abs(r["dx"]), reverse=True)
    for i, r in enumerate(sorted_trials[:10]):
        print(f"  {i+1:2d}. {r['name']:<20} [{r['group']}]  DX={r['dx']:+7.2f}  "
              f"speed={r['speed']:.3f}  PL={r['phase_lock']:.3f}")

    print("\n── BOTTOM 10 GAITS (by |DX|) ──")
    for i, r in enumerate(sorted_trials[-10:]):
        print(f"  {len(sorted_trials)-9+i:2d}. {r['name']:<20} [{r['group']}]  DX={r['dx']:+7.2f}  "
              f"speed={r['speed']:.3f}  PL={r['phase_lock']:.3f}")

    # ── 7. Direction analysis ────────────────────────────────────────────────
    print("\n── DIRECTION ANALYSIS ──")
    for g in group_names:
        items = groups[g]
        pos = sum(1 for r in items if r["dx"] > 0)
        neg = sum(1 for r in items if r["dx"] < 0)
        near_zero = sum(1 for r in items if abs(r["dx"]) < 0.5)
        print(f"  {g:<12}  forward: {pos:2d}  backward: {neg:2d}  near-zero: {near_zero:2d}")

    # ── 8. Comparison with other conditions ──────────────────────────────────
    print("\n── COMPARISON WITH OTHER CONDITIONS ──")
    politics_abs_dx = [abs(r["dx"]) for r in trials]
    politics_pl = [r["phase_lock"] for r in trials]

    print(f"\n  {'Condition':<12} {'N':>4} {'Dead%':>6} {'Med|DX|':>8} {'Max|DX|':>8} "
          f"{'MeanPL':>7}  {'U-test z (|DX|)':>15}")
    print("  " + "-" * 72)

    # Politics row
    dead_pol = sum(1 for d in politics_abs_dx if d < 1.0)
    print(f"  {'politics':<12} {len(trials):4d} {dead_pol/len(trials)*100:5.1f}% "
          f"{np.median(politics_abs_dx):8.2f} {max(politics_abs_dx):8.2f} "
          f"{np.mean(politics_pl):7.3f}  {'(self)':>15}")

    for cond_name in ["verbs", "theorems", "bible", "places", "baseline"]:
        if cond_name not in other:
            continue
        cond_trials = other[cond_name]
        cond_abs_dx = [abs(r["dx"]) for r in cond_trials]
        cond_pl = [r["phase_lock"] for r in cond_trials]
        dead = sum(1 for d in cond_abs_dx if d < 1.0)
        _, z = mann_whitney_u(politics_abs_dx, cond_abs_dx)
        sig = "***" if abs(z) > 2.58 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.64 else ""
        print(f"  {cond_name:<12} {len(cond_trials):4d} {dead/len(cond_trials)*100:5.1f}% "
              f"{np.median(cond_abs_dx):8.2f} {max(cond_abs_dx):8.2f} "
              f"{np.mean(cond_pl):7.3f}  {z:+8.3f} {sig}")

    # ── 9. Behavioral PCA across all conditions ──────────────────────────────
    # Build combined feature matrix for PCA
    all_vecs = []
    all_labels = []
    all_conditions = []

    # Add politics data
    for r in trials:
        all_vecs.append([r["speed"], r["efficiency"], r["phase_lock"], r["entropy"]])
        all_labels.append(r["group"])
        all_conditions.append("politics")

    # Add other conditions
    for cond_name, cond_trials in other.items():
        for r in cond_trials:
            all_vecs.append([r["speed"], r["efficiency"], r["phase_lock"], r["entropy"]])
            all_labels.append(cond_name)
            all_conditions.append(cond_name)

    all_vecs = np.array(all_vecs)
    mu_vec = all_vecs.mean(axis=0)
    std_vec = all_vecs.std(axis=0)
    std_vec[std_vec < 1e-12] = 1
    Z = (all_vecs - mu_vec) / std_vec
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pc = Z @ Vt[:2].T
    var_explained = S[:2]**2 / (S**2).sum() * 100

    # ── FIGURES ──────────────────────────────────────────────────────────────
    print("\n── GENERATING FIGURES ──")

    # Fig 1: DX by political group
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [group_metrics[g]["abs_dx"] for g in group_names]
    bp = ax.boxplot(box_data, labels=[g.title() for g in group_names],
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=4, alpha=0.5))
    for patch, g in zip(bp["boxes"], group_names):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.7)
    # Overlay individual points
    for i, g in enumerate(group_names):
        jitter = np.random.normal(0, 0.04, len(group_metrics[g]["abs_dx"]))
        ax.scatter(np.full(len(group_metrics[g]["abs_dx"]), i + 1) + jitter,
                   group_metrics[g]["abs_dx"],
                   c=GROUP_COLORS[g], s=15, alpha=0.5, zorder=3)
    ax.set_ylabel("|DX| (meters)")
    ax.set_title("Displacement by Political Group")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "pol_fig01_dx_by_group.png")

    # Fig 2: Weight space PCA colored by group
    fig, ax = plt.subplots(figsize=(10, 8))
    # PCA on weight vectors only for politics
    politics_weights = np.array([[r["weights"][k] for k in weight_keys] for r in trials])
    pw_mu = politics_weights.mean(axis=0)
    pw_std = politics_weights.std(axis=0)
    pw_std[pw_std < 1e-12] = 1
    pw_Z = (politics_weights - pw_mu) / pw_std
    pw_U, pw_S, pw_Vt = np.linalg.svd(pw_Z, full_matrices=False)
    pw_pc = pw_Z @ pw_Vt[:2].T
    pw_var = pw_S[:2]**2 / (pw_S**2).sum() * 100

    for g in group_names:
        mask = [r["group"] == g for r in trials]
        idx = [i for i, m in enumerate(mask) if m]
        ax.scatter(pw_pc[idx, 0], pw_pc[idx, 1],
                   c=GROUP_COLORS[g], s=40, alpha=0.7, label=g.title(),
                   edgecolors="white", linewidths=0.5)
        # Label interesting points
        for i in idx:
            if abs(trials[i]["dx"]) > 3.0 or trials[i]["name"] in ["realDonaldTrump", "Putin", "Assange"]:
                ax.annotate(trials[i]["name"], (pw_pc[i, 0], pw_pc[i, 1]),
                            fontsize=7, alpha=0.8, ha="left",
                            xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pw_var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pw_var[1]:.1f}% var)")
    ax.set_title("Weight Space PCA — Political Figures by Group")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "pol_fig02_weight_pca.png")

    # Fig 3: Cross-condition behavioral PCA
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot other conditions first (background)
    offset = len(trials)  # politics are first in the array
    for cond_name in ["baseline", "verbs", "theorems", "bible", "places"]:
        if cond_name not in other:
            continue
        n_cond = len(other[cond_name])
        cond_mask = [i for i in range(offset, offset + n_cond)]
        ax.scatter(pc[cond_mask, 0], pc[cond_mask, 1],
                   c=OTHER_COLORS.get(cond_name, "#CCC"), s=12, alpha=0.3,
                   label=cond_name, edgecolors="none")
        offset += n_cond

    # Plot politics on top, colored by group
    for g in group_names:
        mask = [i for i, r in enumerate(trials) if r["group"] == g]
        ax.scatter(pc[mask, 0], pc[mask, 1],
                   c=GROUP_COLORS[g], s=35, alpha=0.8,
                   label=f"pol:{g}", edgecolors="white", linewidths=0.5,
                   marker="D")

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)")
    ax.set_title("Behavioral PCA — Politics vs All Other Conditions")
    ax.legend(fontsize=8, ncol=2)
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "pol_fig03_cross_condition_pca.png")

    # Fig 4: Weight heatmap — all 79 figures, sorted by group then DX
    fig, ax = plt.subplots(figsize=(8, 16))
    sorted_by_group = []
    for g in group_names:
        group_items = sorted(groups[g], key=lambda r: r["dx"], reverse=True)
        sorted_by_group.extend(group_items)

    weight_matrix = np.array([[r["weights"][k] for k in weight_keys] for r in sorted_by_group])
    names_sorted = [r["name"] for r in sorted_by_group]

    im = ax.imshow(weight_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(weight_keys, fontsize=10)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=6)
    ax.set_title("Synapse Weights by Political Figure")

    # Add group separators
    cum = 0
    for g in group_names:
        n_g = len(groups[g])
        if cum > 0:
            ax.axhline(cum - 0.5, color="white", linewidth=2)
        ax.text(-0.7, cum + n_g / 2, g.title(), fontsize=9, fontweight="bold",
                ha="right", va="center", rotation=0, color=GROUP_COLORS[g])
        cum += n_g

    fig.colorbar(im, ax=ax, shrink=0.5, label="Weight value")
    fig.tight_layout()
    save_fig(fig, "pol_fig04_weight_heatmap.png")

    # Fig 5: Phase lock vs |DX| scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    for g in group_names:
        items = groups[g]
        dx_vals = [abs(r["dx"]) for r in items]
        pl_vals = [r["phase_lock"] for r in items]
        ax.scatter(dx_vals, pl_vals, c=GROUP_COLORS[g], s=30, alpha=0.7,
                   label=g.title(), edgecolors="white", linewidths=0.3)
        # Label outliers
        for r in items:
            if abs(r["dx"]) > 3.5:
                ax.annotate(r["name"], (abs(r["dx"]), r["phase_lock"]),
                            fontsize=7, alpha=0.8, xytext=(5, 5),
                            textcoords="offset points")
    ax.set_xlabel("|DX| (meters)")
    ax.set_ylabel("Phase Lock Score")
    ax.set_title("Coordination vs Displacement — Political Figures")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "pol_fig05_phaselock_vs_dx.png")

    # Fig 6: Cluster membership visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    top_clusters = cluster_counts.most_common(10)
    cluster_labels = []
    cluster_members_by_group = {g: [] for g in group_names}

    for ci, (wt, count) in enumerate(top_clusters):
        members = [(r["name"], r["group"]) for r, t in zip(trials, weight_tuples) if t == wt]
        w_str = ",".join(f"{v:+.1f}" for v in wt)
        dx_val = next(r["dx"] for r, t in zip(trials, weight_tuples) if t == wt)
        cluster_labels.append(f"({w_str})\nDX={dx_val:+.1f}\nn={count}")

    # Stacked bar: group composition of each cluster
    x_pos = np.arange(len(top_clusters))
    bottoms = np.zeros(len(top_clusters))
    for g in group_names:
        heights = []
        for wt, count in top_clusters:
            members = [r for r, t in zip(trials, weight_tuples) if t == wt]
            g_count = sum(1 for r in members if r["group"] == g)
            heights.append(g_count)
        ax.bar(x_pos, heights, bottom=bottoms, color=GROUP_COLORS[g],
               label=g.title(), alpha=0.8)
        bottoms += heights

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cluster_labels, fontsize=7, ha="center")
    ax.set_ylabel("Number of figures in cluster")
    ax.set_title("Top 10 Weight Clusters — Group Composition")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "pol_fig06_cluster_composition.png")

    # ── Save analysis JSON ───────────────────────────────────────────────────
    analysis = {
        "n_total": len(trials),
        "n_unique_weights": len(unique_weights),
        "faithfulness": faithfulness,
        "group_metrics": {g: {k: v for k, v in m.items() if k != "abs_dx"}
                          for g, m in group_metrics.items()},
        "cluster_sizes": cluster_sizes,
        "top_clusters": [
            {
                "weights": dict(zip(weight_keys, [float(v) for v in wt])),
                "count": count,
                "members": [r["name"] for r, t in zip(trials, weight_tuples) if t == wt],
                "dx": next(r["dx"] for r, t in zip(trials, weight_tuples) if t == wt),
            }
            for wt, count in cluster_counts.most_common(15)
        ],
        "top_10_gaits": [
            {"name": r["name"], "group": r["group"], "dx": r["dx"],
             "weights": r["weights"], "speed": r["speed"], "phase_lock": r["phase_lock"]}
            for r in sorted_trials[:10]
        ],
        "pca_variance_explained": [float(v) for v in pw_var],
    }

    out_path = PROJECT / "artifacts" / "structured_random_politics_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    print(f"\n  WROTE {out_path}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
