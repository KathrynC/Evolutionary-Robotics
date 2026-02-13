#!/usr/bin/env python3
"""
analyze_archetypometrics.py

Analysis of the archetypometrics structured random search experiment.
2000 fictional characters from 341 stories (UVM Computational Story Lab),
run through the LLM → synapse weight → PyBullet simulation pipeline.

Produces:
  - Console report with story statistics, clustering, cross-condition comparison
  - artifacts/structured_random_archetypometrics_analysis.json
  - 8 figures in artifacts/plots/arc_fig01-08_*.png

Usage:
    python3 analyze_archetypometrics.py
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

WEIGHT_KEYS = ["w03", "w04", "w13", "w14", "w23", "w24"]

# Genre categories derived from story names (approximate grouping)
GENRE_KEYWORDS = {
    "anime": ["Naruto", "My Hero Academia", "Fullmetal Alchemist",
              "Death Note", "Attack on Titan", "Neon Genesis Evangelion",
              "Dragon Ball Z", "Cowboy Bebop", "One Piece",
              "Sailor Moon", "My Little Pony"],
    "fantasy": ["Lord of the Rings", "Harry Potter", "Game of Thrones",
                "Chronicles of Narnia", "His Dark Materials", "Wheel of Time",
                "Percy Jackson", "Twilight", "Once Upon a Time",
                "The Witcher", "Shrek", "Frozen", "Aladdin",
                "The Princess Bride", "Avatar: The Last Airbender",
                "Buffy the Vampire Slayer", "True Blood", "Supernatural",
                "The Vampire Diaries", "Wynonna Earp"],
    "scifi": ["Star Wars", "Star Trek", "Battlestar Galactica",
              "Westworld", "The Expanse", "Firefly", "Doctor Who",
              "Stranger Things", "The Matrix", "Terminator",
              "Back to the Future", "The Hitchhiker", "Ender's Game",
              "Hunger Games", "Divergent", "The 100", "Alien",
              "Jurassic Park", "Wall-E", "The Mandalorian"],
    "superhero": ["Marvel Cinematic Universe", "The Boys", "X-Men",
                  "DC Extended Universe", "Batman", "Spider-Man",
                  "Watchmen", "The Incredibles"],
    "crime_drama": ["The Wire", "Breaking Bad", "Better Call Saul",
                    "The Sopranos", "Peaky Blinders", "Ozark",
                    "Fargo", "Dexter", "Hannibal", "Sherlock",
                    "The Godfather", "Pulp Fiction", "Goodfellas",
                    "Reservoir Dogs", "Scarface", "Kill Bill",
                    "Casino Royale", "Fight Club"],
    "drama": ["Mad Men", "Downton Abbey", "Succession",
              "The Crown", "Yellowstone", "Grey's Anatomy",
              "Twin Peaks", "LOST", "Desperate Housewives",
              "Gilmore Girls", "Riverdale", "Gossip Girl",
              "Sex and the City", "Bridgerton", "Big Little Lies",
              "Pride and Prejudice", "The Great Gatsby",
              "Gone with the Wind", "Little Women",
              "The Handmaid's Tale", "To Kill a Mockingbird",
              "The Fault in Our Stars", "Mean Girls",
              "Legally Blonde", "Clueless", "Titanic"],
    "comedy": ["The Office", "The Simpsons", "Friends",
               "Parks and Recreation", "Brooklyn Nine-Nine",
               "Schitt's Creek", "The Good Place", "Seinfeld",
               "Arrested Development", "Community",
               "How I Met Your Mother", "New Girl",
               "Glee", "30 Rock", "Bob's Burgers",
               "South Park", "Fleabag", "Ted Lasso",
               "It's Always Sunny", "Toy Story", "Finding Nemo",
               "Monsters, Inc.", "The Big Lebowski",
               "Groundhog Day", "Ghostbusters", "The Addams Family"],
    "horror": ["Stranger Things", "The Walking Dead",
               "American Horror Story", "IT", "Get Out",
               "Scream", "Halloween", "The Shining",
               "Psycho", "Frankenstein", "Dracula"],
    "classic_lit": ["1984", "Sherlock Holmes", "Hamlet",
                    "Romeo and Juliet", "Great Expectations",
                    "Jane Eyre", "Wuthering Heights",
                    "Alice in Wonderland", "Peter Pan",
                    "The Wizard of Oz", "Moby Dick",
                    "A Christmas Carol", "The Count of Monte Cristo",
                    "Les Misérables", "Don Quixote",
                    "Frankenstein", "Dracula"],
    "disney_pixar": ["Toy Story", "Finding Nemo", "The Lion King",
                     "Frozen", "Moana", "Aladdin", "Mulan",
                     "The Little Mermaid", "Cinderella",
                     "Beauty and the Beast", "Tangled",
                     "Monsters, Inc.", "Up", "Inside Out",
                     "Wall-E", "Shrek", "The Incredibles",
                     "Ratatouille", "Coco", "Encanto",
                     "Zootopia", "Big Hero 6"],
}

GENRE_COLORS = {
    "anime":        "#FF6B6B",
    "fantasy":      "#9B59B6",
    "scifi":        "#3498DB",
    "superhero":    "#E74C3C",
    "crime_drama":  "#2C3E50",
    "drama":        "#E67E22",
    "comedy":       "#2ECC71",
    "horror":       "#8B0000",
    "classic_lit":  "#8B4513",
    "disney_pixar": "#F39C12",
    "other":        "#95A5A6",
}

OTHER_CONDITION_FILES = {
    "celebrities": PROJECT / "artifacts" / "structured_random_celebrities.json",
    "politics":    PROJECT / "artifacts" / "structured_random_politics.json",
    "verbs":       PROJECT / "artifacts" / "structured_random_verbs.json",
    "theorems":    PROJECT / "artifacts" / "structured_random_theorems.json",
    "bible":       PROJECT / "artifacts" / "structured_random_bible.json",
    "places":      PROJECT / "artifacts" / "structured_random_places.json",
    "baseline":    PROJECT / "artifacts" / "structured_random_baseline.json",
}

OTHER_COLORS = {
    "celebrities":      "#FF4500",
    "politics":         "#FFD700",
    "verbs":            "#FFA07A",
    "theorems":         "#87CEEB",
    "bible":            "#DDA0DD",
    "places":           "#90EE90",
    "baseline":         "#AAAAAA",
    "archetypometrics": "#E74C3C",
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


def classify_genre(story_name):
    """Classify a story into a genre based on keyword matching."""
    for genre, keywords in GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in story_name.lower():
                return genre
    return "other"


def load_archetypometrics():
    """Load archetypometrics results and split by story."""
    path = PROJECT / "artifacts" / "structured_random_archetypometrics.json"
    with open(path) as f:
        trials = json.load(f)

    stories = {}
    for r in trials:
        seed = r["seed"]
        bracket = seed.rfind(" [")
        if bracket > 0:
            name = seed[:bracket]
            story = seed[bracket + 2:-1]
        else:
            name = seed
            story = "Unknown"
        r["character"] = name
        r["story"] = story
        r["genre"] = classify_genre(story)
        stories.setdefault(story, []).append(r)

    return trials, stories


def load_other_conditions():
    data = {}
    for name, path in OTHER_CONDITION_FILES.items():
        if path.exists():
            with open(path) as f:
                data[name] = json.load(f)
    return data


def main():
    print("\n" + "=" * 70)
    print("ARCHETYPOMETRICS: STRUCTURED RANDOM ANALYSIS")
    print("2000 Fictional Characters from 341 Stories")
    print("=" * 70)

    trials, stories = load_archetypometrics()
    other = load_other_conditions()

    # ── 1. Overview ──────────────────────────────────────────────────────────
    print(f"\nTotal trials: {len(trials)}")
    print(f"Total stories: {len(stories)}")

    # Genre breakdown
    genre_counts = Counter(r["genre"] for r in trials)
    print(f"\nGenre breakdown:")
    for genre, count in genre_counts.most_common():
        print(f"  {genre:<15} {count:4d} characters")

    # Top stories by character count
    top_stories = sorted(stories.items(), key=lambda x: -len(x[1]))[:20]
    print(f"\nTop 20 stories by character count:")
    for story, chars in top_stories:
        genre = classify_genre(story)
        abs_dx = [abs(r["dx"]) for r in chars]
        print(f"  {story:40s} {len(chars):3d} chars  genre={genre:<12} "
              f"med|DX|={np.median(abs_dx):.2f}")

    # ── 2. Weight clustering ─────────────────────────────────────────────────
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
        members = [(r["character"], r["story"]) for r, t in zip(trials, weight_tuples) if t == wt]
        genre_dist = Counter(r["genre"] for r, t in zip(trials, weight_tuples) if t == wt)
        w_str = " ".join(f"{k}={v:+.1f}" for k, v in zip(WEIGHT_KEYS, wt))
        dx_val = next(r["dx"] for r, t in zip(trials, weight_tuples) if t == wt)
        genre_str = ", ".join(f"{g}:{c}" for g, c in genre_dist.most_common(5))
        story_dist = Counter(r["story"] for r, t in zip(trials, weight_tuples) if t == wt)
        n_stories = len(story_dist)
        print(f"    [{count:4d}] ({w_str})  DX={dx_val:+.2f}")
        print(f"           genres: {genre_str}  ({n_stories} stories)")
        # Show sample members: mix of famous and obscure
        sample = [m[0] for m in members[:6]]
        if len(members) > 6:
            sample_str = ", ".join(sample) + f"... +{len(members)-6}"
        else:
            sample_str = ", ".join(sample)
        print(f"           e.g.: {sample_str}")

    # ── 3. Per-genre statistics ──────────────────────────────────────────────
    print(f"\n── PER-GENRE STATISTICS ──")
    print(f"\n{'Genre':<15} {'N':>5} {'Dead%':>6} {'Med|DX|':>8} {'Max|DX|':>8} "
          f"{'MeanSpd':>8} {'MeanPL':>7} {'UniqueWt':>8} {'Faithful':>8}")
    print("-" * 90)

    genre_metrics = {}
    for genre in sorted(genre_counts.keys(), key=lambda g: -genre_counts[g]):
        items = [r for r in trials if r["genre"] == genre]
        abs_dx = [abs(r["dx"]) for r in items]
        dead = sum(1 for v in abs_dx if v < 1.0)
        g_wts = set()
        for r in items:
            w = tuple(round(r["weights"][k], 1) for k in WEIGHT_KEYS)
            g_wts.add(w)
        m = {
            "n": len(items),
            "abs_dx": abs_dx,
            "dead_frac": dead / len(items) if items else 0,
            "median_dx": float(np.median(abs_dx)),
            "max_dx": float(max(abs_dx)),
            "mean_speed": float(np.mean([r["speed"] for r in items])),
            "mean_phase_lock": float(np.mean([r["phase_lock"] for r in items])),
            "n_unique_wt": len(g_wts),
            "faithfulness": len(g_wts) / len(items) if items else 0,
        }
        genre_metrics[genre] = m
        print(f"{genre:<15} {m['n']:5d} {m['dead_frac']*100:5.1f}% {m['median_dx']:8.2f} "
              f"{m['max_dx']:8.2f} {m['mean_speed']:8.3f} {m['mean_phase_lock']:7.3f} "
              f"{m['n_unique_wt']:8d} {m['faithfulness']:8.3f}")

    # ── 4. Per-story faithfulness ────────────────────────────────────────────
    print(f"\n── PER-STORY FAITHFULNESS (stories with 10+ characters) ──")
    story_faith = {}
    for story, chars in stories.items():
        s_wts = set()
        for r in chars:
            w = tuple(round(r["weights"][k], 1) for k in WEIGHT_KEYS)
            s_wts.add(w)
        story_faith[story] = {
            "n": len(chars),
            "unique": len(s_wts),
            "faithfulness": len(s_wts) / len(chars) if chars else 0,
        }

    large_stories = [(s, f) for s, f in story_faith.items() if f["n"] >= 10]
    large_stories.sort(key=lambda x: x[1]["faithfulness"])
    print(f"\n  Most collapsed (lowest faithfulness):")
    for story, f in large_stories[:15]:
        print(f"    {story:40s} {f['unique']:2d}/{f['n']:2d} = {f['faithfulness']:.3f}")

    print(f"\n  Least collapsed (highest faithfulness):")
    for story, f in large_stories[-15:]:
        print(f"    {story:40s} {f['unique']:2d}/{f['n']:2d} = {f['faithfulness']:.3f}")

    # ── 5. Are the same 4 celebrity archetypes present? ──────────────────────
    print(f"\n── COMPARISON WITH CELEBRITY ARCHETYPES ──")
    # The 4 celebrity archetypes (rounded to 1 decimal)
    celebrity_archetypes = {
        "Default":     (0.6, -0.4, 0.2, -0.8, 0.5, -0.3),
        "Assertive":   (0.8, -0.6, 0.2, -0.9, 0.5, -0.4),
        "Transgressor": (0.6, -0.4, -0.2, 0.8, 0.3, -0.5),
        "Contrarian":  (0.8, -0.6, -0.2, 0.9, 0.5, -0.4),
    }
    for arch_name, arch_wt in celebrity_archetypes.items():
        count = sum(1 for t in weight_tuples if t == arch_wt)
        pct = count / len(trials) * 100
        print(f"  {arch_name:15s}: {count:4d} / {len(trials)} ({pct:.1f}%)")

    other_count = sum(1 for t in weight_tuples
                      if t not in celebrity_archetypes.values())
    print(f"  {'Other':15s}: {other_count:4d} / {len(trials)} ({other_count/len(trials)*100:.1f}%)")

    # ── 6. Top and bottom gaits ──────────────────────────────────────────────
    print("\n── TOP 20 GAITS (by |DX|) ──")
    sorted_trials = sorted(trials, key=lambda r: abs(r["dx"]), reverse=True)
    for i, r in enumerate(sorted_trials[:20]):
        print(f"  {i+1:3d}. {r['character']:<25} [{r['story']:<30}]  "
              f"DX={r['dx']:+7.2f}  speed={r['speed']:.3f}  PL={r['phase_lock']:.3f}")

    print("\n── BOTTOM 10 GAITS (by |DX|) ──")
    for i, r in enumerate(sorted_trials[-10:]):
        print(f"  {len(sorted_trials)-9+i:4d}. {r['character']:<25} [{r['story']:<30}]  "
              f"DX={r['dx']:+7.2f}")

    # ── 7. Cross-condition comparison ────────────────────────────────────────
    print("\n── COMPARISON WITH OTHER CONDITIONS ──")
    arc_abs_dx = [abs(r["dx"]) for r in trials]

    print(f"\n  {'Condition':<17} {'N':>5} {'Dead%':>6} {'Med|DX|':>8} {'Max|DX|':>8} "
          f"{'MeanPL':>7}  {'Faithful':>8}  {'U-test z':>8}")
    print("  " + "-" * 85)

    # Archetypometrics row
    dead_arc = sum(1 for d in arc_abs_dx if d < 1.0)
    print(f"  {'archetypometrics':<17} {len(trials):5d} {dead_arc/len(trials)*100:5.1f}% "
          f"{np.median(arc_abs_dx):8.2f} {max(arc_abs_dx):8.2f} "
          f"{np.mean([r['phase_lock'] for r in trials]):7.3f}  "
          f"{faithfulness:8.3f}  {'(self)':>8}")

    for cond_name in ["celebrities", "politics", "verbs", "theorems",
                      "bible", "places", "baseline"]:
        if cond_name not in other:
            continue
        cond_trials = other[cond_name]
        cond_abs_dx = [abs(r["dx"]) for r in cond_trials]
        dead = sum(1 for d in cond_abs_dx if d < 1.0)
        cond_wts = set()
        for r in cond_trials:
            w = tuple(round(r["weights"][k], 1) for k in WEIGHT_KEYS)
            cond_wts.add(w)
        cond_faith = len(cond_wts) / len(cond_trials) if cond_trials else 0
        _, z = mann_whitney_u(arc_abs_dx, cond_abs_dx)
        sig = "***" if abs(z) > 2.58 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.64 else ""
        print(f"  {cond_name:<17} {len(cond_trials):5d} {dead/len(cond_trials)*100:5.1f}% "
              f"{np.median(cond_abs_dx):8.2f} {max(cond_abs_dx):8.2f} "
              f"{np.mean([r['phase_lock'] for r in cond_trials]):7.3f}  "
              f"{cond_faith:8.3f}  {z:+8.3f} {sig}")

    # ── 8. Direction analysis ────────────────────────────────────────────────
    print("\n── DIRECTION ANALYSIS ──")
    pos = sum(1 for r in trials if r["dx"] > 0)
    neg = sum(1 for r in trials if r["dx"] < 0)
    dead_count = sum(1 for r in trials if abs(r["dx"]) < 1.0)
    print(f"  Forward: {pos}  Backward: {neg}  Dead: {dead_count}")
    for genre in sorted(genre_counts.keys(), key=lambda g: -genre_counts[g]):
        items = [r for r in trials if r["genre"] == genre]
        g_pos = sum(1 for r in items if r["dx"] > 0)
        g_neg = sum(1 for r in items if r["dx"] < 0)
        print(f"  {genre:<15}  forward: {g_pos:4d}  backward: {g_neg:4d}")

    # ── FIGURES ──────────────────────────────────────────────────────────────
    print("\n── GENERATING FIGURES ──")

    # Fig 1: DX distribution by genre (box plot)
    fig, ax = plt.subplots(figsize=(14, 7))
    genre_order = sorted(genre_counts.keys(), key=lambda g: -genre_counts[g])
    box_data = [genre_metrics[g]["abs_dx"] for g in genre_order]
    bp = ax.boxplot(box_data,
                    tick_labels=[g.replace("_", "\n") for g in genre_order],
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=4, alpha=0.3))
    for patch, g in zip(bp["boxes"], genre_order):
        patch.set_facecolor(GENRE_COLORS.get(g, "#95A5A6"))
        patch.set_alpha(0.7)
    ax.set_ylabel("|DX| (meters)")
    ax.set_title("Displacement by Genre — 2000 Fictional Characters")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "arc_fig01_dx_by_genre.png")

    # Fig 2: Weight space PCA colored by genre
    fig, ax = plt.subplots(figsize=(12, 9))
    all_weights = np.array([[r["weights"][k] for k in WEIGHT_KEYS] for r in trials])
    w_mu = all_weights.mean(axis=0)
    w_std = all_weights.std(axis=0)
    w_std[w_std < 1e-12] = 1
    w_Z = (all_weights - w_mu) / w_std
    _, w_S, w_Vt = np.linalg.svd(w_Z, full_matrices=False)
    w_pc = w_Z @ w_Vt[:2].T
    w_var = w_S[:2]**2 / (w_S**2).sum() * 100

    for g in genre_order:
        mask = [i for i, r in enumerate(trials) if r["genre"] == g]
        if not mask:
            continue
        ax.scatter(w_pc[mask, 0], w_pc[mask, 1],
                   c=GENRE_COLORS.get(g, "#95A5A6"), s=8, alpha=0.5,
                   label=f"{g} ({len(mask)})", edgecolors="none")

    ax.set_xlabel(f"PC1 ({w_var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({w_var[1]:.1f}% var)")
    ax.set_title("Weight Space PCA — 2000 Fictional Characters by Genre")
    ax.legend(fontsize=7, ncol=3, loc="best")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "arc_fig02_weight_pca.png")

    # Fig 3: Cluster size distribution (histogram)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: histogram of cluster sizes
    ax1.hist(cluster_sizes, bins=50, color="#3498DB", alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Cluster size (number of characters)")
    ax1.set_ylabel("Count")
    ax1.set_title("Weight Cluster Size Distribution")
    ax1.set_yscale("log")
    clean_ax(ax1)

    # Right: cumulative — what fraction of characters are in the top N clusters?
    cum_sizes = np.cumsum(sorted(cluster_sizes, reverse=True))
    ax2.plot(range(1, len(cum_sizes) + 1), cum_sizes / len(trials) * 100,
             color="#E74C3C", linewidth=2)
    ax2.set_xlabel("Number of clusters")
    ax2.set_ylabel("% of characters covered")
    ax2.set_title("Cumulative Coverage by Top Clusters")
    ax2.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90%")
    ax2.axhline(50, color="gray", linestyle=":", alpha=0.5, label="50%")
    ax2.legend()
    clean_ax(ax2)

    fig.tight_layout()
    save_fig(fig, "arc_fig03_cluster_distribution.png")

    # Fig 4: Faithfulness by story (for large stories only)
    fig, ax = plt.subplots(figsize=(16, 8))
    large_story_data = [(s, f) for s, f in story_faith.items() if f["n"] >= 10]
    large_story_data.sort(key=lambda x: x[1]["faithfulness"])
    s_names = [s for s, _ in large_story_data]
    s_faith = [f["faithfulness"] for _, f in large_story_data]
    s_colors = [GENRE_COLORS.get(classify_genre(s), "#95A5A6") for s in s_names]
    bars = ax.barh(range(len(s_names)), s_faith, color=s_colors, alpha=0.8)
    ax.set_yticks(range(len(s_names)))
    ax.set_yticklabels(s_names, fontsize=6)
    ax.set_xlabel("Faithfulness (unique weights / total)")
    ax.set_title("Faithfulness by Story (stories with 10+ characters)")
    ax.axvline(faithfulness, color="black", linestyle="--", alpha=0.5,
               label=f"Overall: {faithfulness:.3f}")
    # Genre legend
    legend_patches = [Patch(facecolor=GENRE_COLORS[g], alpha=0.8,
                            label=g.replace("_", " "))
                      for g in genre_order if g in GENRE_COLORS]
    ax.legend(handles=legend_patches, fontsize=6, ncol=2, loc="lower right")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "arc_fig04_faithfulness_by_story.png")

    # Fig 5: Celebrity archetype overlap — how much of fiction matches the 4 archetypes?
    fig, ax = plt.subplots(figsize=(10, 6))
    arch_counts = {}
    for arch_name, arch_wt in celebrity_archetypes.items():
        arch_counts[arch_name] = sum(1 for t in weight_tuples if t == arch_wt)
    arch_counts["Other"] = sum(1 for t in weight_tuples
                                if t not in celebrity_archetypes.values())
    arch_colors = {
        "Default": "#3498DB",
        "Assertive": "#E74C3C",
        "Transgressor": "#2C3E50",
        "Contrarian": "#F39C12",
        "Other": "#95A5A6",
    }
    labels = list(arch_counts.keys())
    sizes = [arch_counts[l] for l in labels]
    colors = [arch_colors[l] for l in labels]
    ax.pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
           colors=colors, autopct="%1.1f%%", startangle=140,
           textprops={"fontsize": 10})
    ax.set_title(f"How Fiction Maps to the 4 Celebrity Archetypes\n(N={len(trials)})")
    fig.tight_layout()
    save_fig(fig, "arc_fig05_archetype_overlap.png")

    # Fig 6: Cross-condition PCA (archetypometrics + all others)
    all_vecs = []
    all_labels = []

    for r in trials:
        all_vecs.append([r["speed"], r["efficiency"], r["phase_lock"], r["entropy"]])
        all_labels.append("archetypometrics")

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
    n_arc = len(trials)
    # Plot other conditions as background
    idx = n_arc
    for cond_name in ["baseline", "politics", "celebrities", "verbs",
                      "theorems", "bible", "places"]:
        if cond_name not in other:
            continue
        n_c = len(other[cond_name])
        c_idx = list(range(idx, idx + n_c))
        ax.scatter(pc[c_idx, 0], pc[c_idx, 1],
                   c=OTHER_COLORS.get(cond_name, "#CCC"), s=10, alpha=0.2,
                   label=cond_name, edgecolors="none")
        idx += n_c

    # Plot archetypometrics on top
    ax.scatter(pc[:n_arc, 0], pc[:n_arc, 1],
               c="#E74C3C", s=6, alpha=0.3,
               label=f"fiction ({n_arc})", edgecolors="none", marker="D")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
    ax.set_title("Behavioral PCA — 2000 Fictional Characters vs All Conditions")
    ax.legend(fontsize=7, ncol=2, loc="best")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "arc_fig06_cross_condition_pca.png")

    # Fig 7: Effective dimensionality comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    dims = {}
    # Archetypometrics
    arc_weights = np.array([[r["weights"][k] for k in WEIGHT_KEYS] for r in trials])
    arc_mu = arc_weights.mean(axis=0)
    arc_std_v = arc_weights.std(axis=0)
    arc_std_v[arc_std_v < 1e-12] = 1
    arc_Z = (arc_weights - arc_mu) / arc_std_v
    _, arc_S, _ = np.linalg.svd(arc_Z, full_matrices=False)
    arc_eigs = arc_S**2
    dims["archetypometrics"] = float((arc_eigs.sum())**2 / (arc_eigs**2).sum())

    for cond_name, cond_trials in other.items():
        cw = np.array([[r["weights"][k] for k in WEIGHT_KEYS] for r in cond_trials])
        cmu = cw.mean(axis=0)
        cstd = cw.std(axis=0)
        cstd[cstd < 1e-12] = 1
        cZ = (cw - cmu) / cstd
        _, cS, _ = np.linalg.svd(cZ, full_matrices=False)
        ceigs = cS**2
        dims[cond_name] = float((ceigs.sum())**2 / (ceigs**2).sum())

    cond_order = ["archetypometrics", "celebrities", "politics", "verbs",
                  "theorems", "bible", "places", "baseline"]
    cond_order = [c for c in cond_order if c in dims]
    colors_list = [OTHER_COLORS.get(c, "#E74C3C") for c in cond_order]
    ax.bar(range(len(cond_order)),
           [dims[c] for c in cond_order],
           color=colors_list, alpha=0.8)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_order, fontsize=8, rotation=15)
    ax.set_ylabel("Effective Dimensionality (Participation Ratio)")
    ax.set_title("Effective Dimensionality by Condition (max = 6)")
    ax.axhline(6.0, color="gray", linestyle=":", alpha=0.5, label="Max (6D)")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "arc_fig07_dimensionality.png")

    # Fig 8: Story-internal diversity — for each large story, how many archetypes?
    fig, ax = plt.subplots(figsize=(14, 7))
    story_data = []
    for story, chars in stories.items():
        if len(chars) < 5:
            continue
        s_wts = set()
        for r in chars:
            w = tuple(round(r["weights"][k], 1) for k in WEIGHT_KEYS)
            s_wts.add(w)
        story_data.append((story, len(chars), len(s_wts)))

    story_data.sort(key=lambda x: -x[1])
    s_names = [s[0] for s in story_data[:40]]
    s_total = [s[1] for s in story_data[:40]]
    s_unique = [s[2] for s in story_data[:40]]
    s_colors = [GENRE_COLORS.get(classify_genre(s), "#95A5A6") for s in s_names]

    x = np.arange(len(s_names))
    ax.bar(x - 0.2, s_total, 0.4, color=s_colors, alpha=0.4, label="Total chars")
    ax.bar(x + 0.2, s_unique, 0.4, color=s_colors, alpha=0.9, label="Unique weights")
    ax.set_xticks(x)
    ax.set_xticklabels(s_names, fontsize=5, rotation=60, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Story Size vs Weight Diversity (top 40 stories)")
    ax.legend()
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, "arc_fig08_story_diversity.png")

    # ── Save analysis JSON ───────────────────────────────────────────────────
    analysis = {
        "n_total": len(trials),
        "n_stories": len(stories),
        "n_unique_weights": len(unique_weights),
        "faithfulness": faithfulness,
        "genre_metrics": {
            g: {k: v for k, v in m.items() if k != "abs_dx"}
            for g, m in genre_metrics.items()
        },
        "cluster_sizes": cluster_sizes,
        "top_clusters": [
            {
                "weights": dict(zip(WEIGHT_KEYS, [float(v) for v in wt])),
                "count": count,
                "dx": float(next(r["dx"] for r, t in zip(trials, weight_tuples) if t == wt)),
                "n_stories": len(set(r["story"] for r, t in zip(trials, weight_tuples) if t == wt)),
                "genre_dist": dict(Counter(
                    r["genre"] for r, t in zip(trials, weight_tuples) if t == wt
                )),
                "sample_members": [
                    f"{r['character']} ({r['story']})"
                    for r, t in zip(trials, weight_tuples) if t == wt
                ][:10],
            }
            for wt, count in cluster_counts.most_common(20)
        ],
        "celebrity_archetype_overlap": {
            arch_name: sum(1 for t in weight_tuples if t == arch_wt)
            for arch_name, arch_wt in celebrity_archetypes.items()
        },
        "story_faithfulness": {
            s: f for s, f in story_faith.items() if f["n"] >= 10
        },
        "top_20_gaits": [
            {"character": r["character"], "story": r["story"],
             "genre": r["genre"], "dx": r["dx"],
             "weights": r["weights"], "speed": r["speed"],
             "phase_lock": r["phase_lock"]}
            for r in sorted_trials[:20]
        ],
        "effective_dimensionality": dims,
    }

    out_path = PROJECT / "artifacts" / "structured_random_archetypometrics_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    print(f"\n  WROTE {out_path}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
