#!/usr/bin/env python3
"""
structured_random_compare.py

Comparative Analysis for the Structured Random Search Experiment
=================================================================

This script is the analysis hub for the experiment. It loads results from all
5 conditions (4 LLM-mediated + 1 baseline), computes cross-condition statistics,
runs significance tests, and generates 6 diagnostic plots.

EXPERIMENTAL DESIGN
-------------------
The experiment asks: does an LLM function as a *structured* sampler of neural
network weight space? We compare 5 conditions, each generating 100 weight
vectors for a 3-link walking robot:

  1. VERBS:    Multilingual verbs → LLM → weights (action qualities)
  2. THEOREMS: Mathematical theorems → LLM → weights (structural principles)
  3. BIBLE:    KJV Bible verses → LLM → weights (imagery, emotion)
  4. PLACES:   Global place names → LLM → weights (terrain, climate, energy)
  5. BASELINE: Uniform random U[-1,1]^6 → weights (no LLM involvement)

All conditions use the same robot, simulation parameters, and Beer-framework
analytics. The only variable is how the 6 synapse weights are generated.

ANALYSIS OUTPUTS
-----------------
Console:
  - Summary table: N, dead fraction, median/max |DX|, mean speed/efficiency/phase lock
  - Mann-Whitney U tests vs baseline (all conditions)
  - Pairwise Mann-Whitney U tests between structured conditions
  - Best gait per condition

JSON:
  - artifacts/structured_random_comparison.json: summary statistics per condition

Plots (artifacts/plots/sr_fig01-06):
  1. Box plot of |DX| by condition — shows distribution shape and outliers
  2. Dead fraction bar chart — what % of gaits are immobile (<1m)?
  3. Phase lock distributions — coordination quality histograms
  4. Speed vs efficiency scatter — Pareto frontier exploration
  5. Best-of-N discovery curves — how quickly does each condition find its best?
  6. PCA behavioral diversity — 2D projection of (speed, efficiency, phase_lock,
     entropy) showing which behavioral subspace each condition occupies

KEY FINDINGS (from initial run)
---------------------------------
  - Baseline dominates on median |DX| (6.64m vs 1.18-2.79m for structured)
  - ALL structured conditions significantly lower than baseline (p < 0.001)
  - Bible: 0% dead, produced the overall champion (Revelation 6:8, DX=+29.17m)
  - Places: 0% dead but only 5.64m max — most conservative condition
  - Theorems: highest phase lock (0.904), 18/20 top phase-lock slots
  - The LLM is a conservative sampler: it avoids extremes (both death and
    greatness) and clusters in a tight behavioral subspace with high coordination
  - PCA shows structured conditions occupy a small submanifold of the full
    behavioral space that baseline explores

STATISTICAL TESTS
------------------
Mann-Whitney U test (implemented without scipy to maintain numpy-only constraint):
  - Non-parametric rank-based test suitable for non-normal distributions
  - Uses normal approximation for z-score (valid for n ≥ 20)
  - Significance: * p<0.10, ** p<0.05, *** p<0.01

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

# Result files for each condition. Each is a JSON list of per-trial dicts
# produced by run_structured_search() or run_uniform_baseline().
CONDITIONS = {
    "verbs":    PROJECT / "artifacts" / "structured_random_verbs.json",
    "theorems": PROJECT / "artifacts" / "structured_random_theorems.json",
    "bible":    PROJECT / "artifacts" / "structured_random_bible.json",
    "places":   PROJECT / "artifacts" / "structured_random_places.json",
    "baseline": PROJECT / "artifacts" / "structured_random_baseline.json",
}

# Color scheme for plots: warm red for verbs, cool blue for theorems,
# muted purple for bible, green for places, gray for baseline.
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
    """Mann-Whitney U test implemented from scratch (no scipy dependency).

    This is a non-parametric rank-sum test that compares two independent
    samples without assuming normality. It works by:
    1. Combining both samples and ranking all values
    2. Handling ties by assigning the mean rank to tied values
    3. Computing U1 = sum of ranks in sample x minus the minimum possible
    4. Using the normal approximation z = (U1 - μ) / σ for the p-value

    The normal approximation is valid for sample sizes ≥ 20, which all our
    conditions satisfy (n = 95-100 per condition).

    Returns:
        Tuple of (U statistic, z-score). |z| > 1.96 → p < 0.05,
        |z| > 2.58 → p < 0.01.
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
    # For each condition, compute both per-trial arrays (for plotting and
    # statistical tests) and scalar summaries (for the summary table).
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
    # Six diagnostic plots, each highlighting a different aspect of the
    # structured vs. random comparison.
    print("\nGenerating comparison plots...")
    cond_order = [n for n in ["verbs", "theorems", "bible", "places", "baseline"] if n in metrics]

    # Fig 1: Box plot of |DX| by condition — shows the full distribution shape.
    # Key visual: baseline's box is wide (high variance) while structured
    # conditions are compressed near zero with occasional high outliers.
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

    # Fig 2: Dead fraction bar chart — what % of gaits are immobile?
    # Key visual: bible and places at 0%, baseline at 8%. The LLM avoids
    # the dead zone of weight space entirely for these conditions.
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

    # Fig 3: Phase lock distributions — inter-joint coordination quality.
    # Phase lock ∈ [0,1] measures how consistently the two joints maintain a
    # fixed phase relationship (via Hilbert transform). Higher = more periodic.
    # Key visual: baseline peaks around 0.5-0.7, structured conditions peak 0.9+.
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

    # Fig 4: Speed vs efficiency scatter — the Pareto frontier.
    # Efficiency = distance / work_proxy. Fast gaits tend to be inefficient
    # (high energy for moderate distance); efficient gaits tend to be slow
    # (low energy for modest distance). The Pareto frontier shows the best
    # achievable tradeoff. Key visual: structured conditions cluster in the
    # low-speed, moderate-efficiency region.
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

    # Fig 5: Best-of-N discovery curves — how quickly does each condition
    # find its best gait? Plots the running maximum of |DX| as trials
    # accumulate. Steeper initial slope = faster discovery. A flat plateau
    # means the condition has exhausted its diversity. Key visual: baseline
    # rises steadily throughout; Bible jumps early (Revelation 6:8 at trial ~30).
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

    # Fig 6: Behavioral diversity — 2D PCA of (speed, efficiency, phase_lock, entropy).
    # Projects each gait from 4D behavioral space into 2D via SVD on the
    # standardized feature matrix. This shows which behavioral subspace each
    # condition explores. Key visual: baseline (gray) fills the entire space;
    # structured conditions cluster tightly in the low-PC1 region, occupying
    # a tiny submanifold of the full behavioral repertoire.
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
    """Run all 5 conditions sequentially: baseline first, then structured.

    Baseline runs first because it requires no LLM (just uniform random
    sampling), so it can verify the simulation pipeline before committing
    to the ~6 minutes of Ollama calls. Each structured condition imports
    its own module and calls main(), which handles seed selection, prompt
    construction, and the full LLM→simulation→analytics pipeline.
    """
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
