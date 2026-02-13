#!/usr/bin/env python3
"""
structured_random_compare.py

Comparison analysis for the structured random search experiment.

Loads results from all 5 conditions (verbs, theorems, bible, places, baseline),
computes comparative statistics, runs significance tests, and generates
diagnostic plots.

Can also run all 5 conditions sequentially if result files don't exist yet.

Usage:
    python3 structured_random_compare.py              # analyze existing results
    python3 structured_random_compare.py --run-all     # run all conditions then analyze
"""

import json
import sys
from pathlib import Path

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

COLORS = {
    "verbs":    "#E24A33",
    "theorems": "#348ABD",
    "bible":    "#988ED5",
    "places":   "#8EBA42",
    "baseline": "#777777",
}


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {path}")


def load_results():
    """Load all condition result files."""
    data = {}
    for name, path in CONDITIONS.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        with open(path) as f:
            data[name] = json.load(f)
        print(f"  Loaded {name}: {len(data[name])} trials")
    return data


def mann_whitney_u(x, y):
    """Simple Mann-Whitney U test (no scipy dependency).

    Returns U statistic and approximate z-score for large samples.
    """
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.empty_like(combined, dtype=float)
    order = np.argsort(combined)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Handle ties by averaging ranks
    for val in np.unique(combined):
        mask = combined == val
        ranks[mask] = ranks[mask].mean()

    r1 = ranks[:nx].sum()
    u1 = r1 - nx * (nx + 1) / 2
    mu = nx * ny / 2
    sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12)
    z = (u1 - mu) / sigma if sigma > 0 else 0
    return u1, z


def main():
    if "--run-all" in sys.argv:
        run_all_conditions()

    print("\n" + "=" * 70)
    print("STRUCTURED RANDOM SEARCH: COMPARATIVE ANALYSIS")
    print("=" * 70 + "\n")

    data = load_results()
    if len(data) < 2:
        print("Need at least 2 conditions to compare. Run with --run-all first.")
        return

    # ── Extract metrics per condition ────────────────────────────────────────
    metrics = {}
    for name, trials in data.items():
        abs_dx = [abs(r["dx"]) for r in trials]
        metrics[name] = {
            "dx": [r["dx"] for r in trials],
            "abs_dx": abs_dx,
            "speed": [r["speed"] for r in trials],
            "efficiency": [r["efficiency"] for r in trials],
            "phase_lock": [r["phase_lock"] for r in trials],
            "entropy": [r["entropy"] for r in trials],
            "n": len(trials),
            "dead_frac": sum(1 for d in abs_dx if d < 1.0) / len(trials),
            "median_dx": float(np.median(abs_dx)),
            "max_dx": max(abs_dx),
            "mean_speed": float(np.mean([r["speed"] for r in trials])),
            "mean_efficiency": float(np.mean([r["efficiency"] for r in trials])),
            "mean_phase_lock": float(np.mean([r["phase_lock"] for r in trials])),
        }

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{'Condition':<12} {'N':>4} {'Dead%':>6} {'Med|DX|':>8} {'Max|DX|':>8} "
          f"{'MeanSpd':>8} {'MeanEff':>8} {'MeanPL':>7}")
    print("-" * 72)
    for name in ["verbs", "theorems", "bible", "places", "baseline"]:
        if name not in metrics:
            continue
        m = metrics[name]
        print(f"{name:<12} {m['n']:4d} {m['dead_frac']*100:5.1f}% {m['median_dx']:8.2f} "
              f"{m['max_dx']:8.2f} {m['mean_speed']:8.3f} {m['mean_efficiency']:8.5f} "
              f"{m['mean_phase_lock']:7.3f}")

    # ── Pairwise significance tests vs baseline ──────────────────────────────
    if "baseline" in metrics:
        print(f"\nMann-Whitney U tests (|DX|) vs baseline:")
        print(f"  {'Condition':<12} {'U':>10} {'z':>8} {'Median diff':>12}")
        print("  " + "-" * 44)
        for name in ["verbs", "theorems", "bible", "places"]:
            if name not in metrics:
                continue
            u, z = mann_whitney_u(metrics[name]["abs_dx"], metrics["baseline"]["abs_dx"])
            diff = metrics[name]["median_dx"] - metrics["baseline"]["median_dx"]
            sig = "***" if abs(z) > 2.58 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.64 else ""
            print(f"  {name:<12} {u:10.0f} {z:+8.3f} {diff:+12.2f}m {sig}")

    # ── Pairwise comparisons between structured conditions ───────────────────
    structured = [n for n in ["verbs", "theorems", "bible", "places"] if n in metrics]
    if len(structured) > 1:
        print(f"\nPairwise Mann-Whitney U tests (|DX|) between structured conditions:")
        print(f"  {'Pair':<25} {'z':>8}")
        print("  " + "-" * 35)
        for i, a in enumerate(structured):
            for b in structured[i+1:]:
                _, z = mann_whitney_u(metrics[a]["abs_dx"], metrics[b]["abs_dx"])
                sig = "***" if abs(z) > 2.58 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.64 else ""
                print(f"  {a+' vs '+b:<25} {z:+8.3f} {sig}")

    # ── Best gaits per condition ─────────────────────────────────────────────
    print(f"\nBest gait per condition:")
    for name, trials in data.items():
        best = max(trials, key=lambda r: abs(r["dx"]))
        print(f"  {name:<12} DX={best['dx']:+8.2f}  seed={str(best['seed'])[:60]}")

    # ── Save comparison JSON ─────────────────────────────────────────────────
    comp_path = PROJECT / "artifacts" / "structured_random_comparison.json"
    comp = {name: {k: v for k, v in m.items() if k not in ("dx", "abs_dx", "speed", "efficiency", "phase_lock", "entropy")}
            for name, m in metrics.items()}
    with open(comp_path, "w") as f:
        json.dump(comp, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {comp_path}")

    # ── FIGURES ──────────────────────────────────────────────────────────────
    print("\nGenerating comparison plots...")
    cond_order = [n for n in ["verbs", "theorems", "bible", "places", "baseline"] if n in metrics]

    # Fig 1: Box/violin plot of |DX| by condition
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [metrics[n]["abs_dx"] for n in cond_order]
    bp = ax.boxplot(box_data, labels=cond_order, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, name in zip(bp["boxes"], cond_order):
        patch.set_facecolor(COLORS.get(name, "#CCCCCC"))
        patch.set_alpha(0.7)
    ax.set_ylabel("|DX| (meters)")
    ax.set_title("Displacement by Condition")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "sr_fig01_dx_by_condition.png")

    # Fig 2: Dead fraction bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    dead_fracs = [metrics[n]["dead_frac"] * 100 for n in cond_order]
    bars = ax.bar(cond_order, dead_fracs,
                  color=[COLORS.get(n, "#CCC") for n in cond_order], alpha=0.8)
    for bar, pct in zip(bars, dead_fracs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.0f}%", ha="center", fontsize=10)
    ax.set_ylabel("Dead fraction (%)")
    ax.set_title("Fraction of Dead Gaits (|DX| < 1m) by Condition")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "sr_fig02_dead_fraction.png")

    # Fig 3: Phase lock distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in cond_order:
        ax.hist(metrics[name]["phase_lock"], bins=25, alpha=0.5,
                color=COLORS.get(name, "#CCC"), label=name, density=True)
    ax.set_xlabel("Phase Lock Score")
    ax.set_ylabel("Density")
    ax.set_title("Phase Lock Distribution by Condition")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "sr_fig03_phase_lock_by_condition.png")

    # Fig 4: Speed vs efficiency scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    for name in cond_order:
        trials = data[name]
        spd = [r["speed"] for r in trials]
        eff = [r["efficiency"] for r in trials]
        ax.scatter(spd, eff, c=COLORS.get(name, "#CCC"), s=20, alpha=0.6,
                   label=name, edgecolors="white", linewidths=0.3)
    ax.set_xlabel("Mean Speed")
    ax.set_ylabel("Efficiency (distance/work)")
    ax.set_title("Speed vs Efficiency by Condition")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "sr_fig04_speed_efficiency.png")

    # Fig 5: Best-of-N curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in cond_order:
        abs_dx = np.array(metrics[name]["abs_dx"])
        best_so_far = np.maximum.accumulate(abs_dx)
        ax.plot(np.arange(1, len(abs_dx) + 1), best_so_far,
                color=COLORS.get(name, "#CCC"), lw=2, label=name)
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Best |DX| so far (meters)")
    ax.set_title("Best-of-N Discovery Curves by Condition")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "sr_fig05_best_of_n.png")

    # Fig 6: Behavioral diversity — 2D PCA of (speed, efficiency, phase_lock, entropy)
    fig, ax = plt.subplots(figsize=(10, 8))
    all_vecs = []
    all_labels = []
    for name in cond_order:
        trials = data[name]
        for r in trials:
            all_vecs.append([r["speed"], r["efficiency"], r["phase_lock"], r["entropy"]])
            all_labels.append(name)
    all_vecs = np.array(all_vecs)
    # Standardize
    mu = all_vecs.mean(axis=0)
    std = all_vecs.std(axis=0)
    std[std < 1e-12] = 1
    Z = (all_vecs - mu) / std
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pc = Z @ Vt[:2].T  # project onto first 2 PCs
    var_explained = S[:2]**2 / (S**2).sum() * 100

    offset = 0
    for name in cond_order:
        n = len(data[name])
        ax.scatter(pc[offset:offset+n, 0], pc[offset:offset+n, 1],
                   c=COLORS.get(name, "#CCC"), s=20, alpha=0.6, label=name,
                   edgecolors="white", linewidths=0.3)
        offset += n
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)")
    ax.set_title("Behavioral Diversity: PCA of Beer-Framework Metrics")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "sr_fig06_diversity.png")

    print("\nComparison complete.")


def run_all_conditions():
    """Run all 5 conditions sequentially."""
    from structured_random_common import run_uniform_baseline

    # Run baseline
    baseline_path = CONDITIONS["baseline"]
    if not baseline_path.exists():
        run_uniform_baseline(baseline_path)

    # Run structured conditions
    for script in ["structured_random_verbs", "structured_random_theorems",
                    "structured_random_bible", "structured_random_places"]:
        module = __import__(script)
        module.main()


if __name__ == "__main__":
    main()
