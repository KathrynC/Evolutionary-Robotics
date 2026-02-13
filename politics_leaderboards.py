#!/usr/bin/env python3
"""
politics_leaderboards.py

Generate leaderboards for the political figures gait experiment.
79 figures, 4 distinct gaits, 8 metrics — ranked and visualized.

Output:
  - Console: formatted leaderboard tables
  - artifacts/plots/pol_leaderboards.png (multi-panel figure)
  - artifacts/politics_leaderboards.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from compute_beer_analytics import NumpyEncoder

PLOT_DIR = PROJECT / "artifacts" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Archetype names and colors for the 4 clusters
ARCHETYPE_COLORS = {
    "Assertive":     "#E24A33",
    "Majority":      "#348ABD",
    "Contrarian":    "#8EBA42",
    "Whistleblower": "#FFD700",
}

GROUP_BADGES = {
    "family":     "FAM",
    "admin":      "ADM",
    "adjacent":   "ADJ",
    "opposition": "OPP",
}

GROUP_COLORS = {
    "family":     "#E24A33",
    "admin":      "#348ABD",
    "adjacent":   "#8EBA42",
    "opposition": "#988ED5",
}


def classify_archetype(weights):
    """Assign an archetype name based on the weight vector."""
    w13 = round(weights["w13"], 1)
    w14 = round(weights["w14"], 1)
    w03 = round(weights["w03"], 1)

    if w13 == -0.2 and w14 == 0.8 and w03 == 0.6:
        return "Whistleblower"
    elif w13 == -0.2 and w14 == 0.9 and w03 == 0.8:
        return "Contrarian"
    elif w03 == 0.8 and w13 == 0.2:
        return "Assertive"
    elif w03 == 0.6 and w13 == 0.2:
        return "Majority"
    return "Unknown"


def load_data():
    path = PROJECT / "artifacts" / "structured_random_politics.json"
    with open(path) as f:
        trials = json.load(f)

    for r in trials:
        seed = r["seed"]
        for g in ["family", "admin", "adjacent", "opposition"]:
            if f"[{g}]" in seed:
                r["group"] = g
                r["name"] = seed.split(" [")[0]
                break
        r["archetype"] = classify_archetype(r["weights"])
        # Derived metrics
        r["abs_dx"] = abs(r["dx"])
        r["abs_dy"] = abs(r["dy"])
        r["total_displacement"] = np.sqrt(r["dx"]**2 + r["dy"]**2)
        r["yaw_degrees"] = abs(r["yaw_net_rad"]) * 180 / np.pi
    return trials


def print_leaderboard(title, trials, metric_key, metric_label, ascending=False, top_n=79):
    """Print a ranked leaderboard for a given metric."""
    sorted_t = sorted(trials, key=lambda r: r[metric_key], reverse=not ascending)[:top_n]

    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"  {'Rank':<5} {'Name':<20} {'Group':<5} {'Archetype':<14} {metric_label:>12}")
    print(f"  {'-'*60}")

    prev_val = None
    rank = 0
    for i, r in enumerate(sorted_t):
        val = r[metric_key]
        if val != prev_val:
            rank = i + 1
            prev_val = val
        # Medal emoji for top 3
        medal = {1: " [1st]", 2: " [2nd]", 3: " [3rd]"}.get(rank, "")

        if isinstance(val, float):
            val_str = f"{val:+.4f}" if val < 0 or metric_key == "dx" else f"{val:.4f}"
        else:
            val_str = str(val)

        print(f"  {rank:<5} {r['name']:<20} {GROUP_BADGES[r['group']]:<5} "
              f"{r['archetype']:<14} {val_str:>12}{medal}")


def make_leaderboard_figure(trials):
    """Create a multi-panel leaderboard figure."""

    # Define leaderboards
    boards = [
        ("DISTANCE CHAMPION", "total_displacement", "Distance (m)", False),
        ("SPEED DEMON", "speed", "Speed (m/s)", False),
        ("MOST EFFICIENT", "efficiency", "Efficiency", False),
        ("MOST COORDINATED", "phase_lock", "Phase Lock", False),
        ("HARDEST WORKER", "work_proxy", "Work (J-proxy)", False),
        ("MOST COMPLEX GAIT", "entropy", "Entropy (bits)", False),
        ("BIGGEST SPINNER", "yaw_degrees", "Yaw (degrees)", False),
        ("MOST LATERAL", "abs_dy", "|DY| (m)", False),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle("POLITICAL FIGURES GAIT LEADERBOARDS",
                 fontsize=22, fontweight="bold", y=0.98, color="#1a1a2e")
    fig.patch.set_facecolor("#f8f8f0")

    for idx, (title, metric, label, ascending) in enumerate(boards):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor("#f8f8f0")
        ax.axis("off")

        sorted_t = sorted(trials, key=lambda r: r[metric], reverse=not ascending)

        # Show top 15
        show_n = 15
        top = sorted_t[:show_n]

        ax.text(0.5, 0.97, title, fontsize=14, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes, color="#1a1a2e")
        ax.text(0.5, 0.92, f"Ranked by {label}", fontsize=9,
                ha="center", va="top", transform=ax.transAxes, color="#666666")

        # Header
        y_start = 0.86
        row_h = 0.053

        ax.text(0.02, y_start, "#", fontsize=8, fontweight="bold",
                transform=ax.transAxes, color="#888888")
        ax.text(0.08, y_start, "Name", fontsize=8, fontweight="bold",
                transform=ax.transAxes, color="#888888")
        ax.text(0.48, y_start, "Group", fontsize=8, fontweight="bold",
                transform=ax.transAxes, color="#888888")
        ax.text(0.62, y_start, "Archetype", fontsize=8, fontweight="bold",
                transform=ax.transAxes, color="#888888")
        ax.text(0.95, y_start, label, fontsize=8, fontweight="bold",
                ha="right", transform=ax.transAxes, color="#888888")

        prev_val = None
        rank = 0
        for i, r in enumerate(top):
            val = r[metric]
            if val != prev_val:
                rank = i + 1
                prev_val = val

            y = y_start - row_h * (i + 1)

            # Alternating row background
            if i % 2 == 0:
                ax.axhspan(y - row_h * 0.3, y + row_h * 0.4,
                           xmin=0, xmax=1, alpha=0.08, color="#1a1a2e",
                           transform=ax.transAxes)

            # Rank with medal
            rank_str = str(rank)
            rank_color = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}.get(rank, "#333333")
            rank_weight = "bold" if rank <= 3 else "normal"
            ax.text(0.03, y, rank_str, fontsize=9, fontweight=rank_weight,
                    transform=ax.transAxes, color=rank_color, ha="center")

            # Name
            name_weight = "bold" if rank <= 3 else "normal"
            ax.text(0.08, y, r["name"], fontsize=9, fontweight=name_weight,
                    transform=ax.transAxes, color="#1a1a2e")

            # Group badge
            badge = GROUP_BADGES[r["group"]]
            badge_color = GROUP_COLORS[r["group"]]
            ax.text(0.50, y, badge, fontsize=8, fontweight="bold",
                    transform=ax.transAxes, color=badge_color, ha="center")

            # Archetype
            arch_color = ARCHETYPE_COLORS.get(r["archetype"], "#333")
            ax.text(0.62, y, r["archetype"], fontsize=8,
                    transform=ax.transAxes, color=arch_color)

            # Value
            if metric == "dx":
                val_str = f"{val:+.3f}"
            elif metric in ("efficiency",):
                val_str = f"{val:.6f}"
            elif metric in ("phase_lock", "roll_dom"):
                val_str = f"{val:.4f}"
            else:
                val_str = f"{val:.3f}"
            ax.text(0.95, y, val_str, fontsize=9, fontweight=name_weight,
                    ha="right", transform=ax.transAxes, color="#1a1a2e",
                    fontfamily="monospace")

    # Add legend at bottom
    fig.text(0.5, 0.01,
             "4 archetypes from 79 names  |  "
             "Assertive (25): Trump, Putin, Musk, Bannon...  |  "
             "Majority (50): Biden, Clinton, Ivanka, Pelosi...  |  "
             "Contrarian (2): McCarthy, Spicer  |  "
             "Whistleblower (2): Assange, Snowden",
             fontsize=9, ha="center", color="#666666", style="italic")

    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    path = PLOT_DIR / "pol_leaderboards.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  WROTE {path}")


def make_archetype_card(trials):
    """Create a summary card showing the 4 archetypes."""
    archetypes = {}
    for r in trials:
        a = r["archetype"]
        if a not in archetypes:
            archetypes[a] = {
                "members": [],
                "weights": r["weights"],
                "dx": r["dx"],
                "speed": r["speed"],
                "efficiency": r["efficiency"],
                "phase_lock": r["phase_lock"],
                "entropy": r["entropy"],
                "work_proxy": r["work_proxy"],
                "yaw_degrees": r["yaw_degrees"],
                "total_displacement": r["total_displacement"],
            }
        archetypes[a]["members"].append(r["name"])

    fig, axes = plt.subplots(1, 4, figsize=(20, 8))
    fig.suptitle("THE FOUR ARCHETYPES", fontsize=20, fontweight="bold",
                 y=0.98, color="#1a1a2e")
    fig.patch.set_facecolor("#f8f8f0")

    weight_keys = ["w03", "w04", "w13", "w14", "w23", "w24"]
    arch_order = ["Whistleblower", "Contrarian", "Assertive", "Majority"]

    for i, arch_name in enumerate(arch_order):
        ax = axes[i]
        ax.set_facecolor("#f8f8f0")
        ax.axis("off")

        data = archetypes[arch_name]
        color = ARCHETYPE_COLORS[arch_name]
        n_members = len(data["members"])

        # Title
        ax.text(0.5, 0.95, arch_name.upper(), fontsize=14, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes, color=color)
        ax.text(0.5, 0.89, f"{n_members} members", fontsize=10,
                ha="center", va="top", transform=ax.transAxes, color="#888888")

        # Weights
        y = 0.80
        ax.text(0.5, y, "WEIGHTS", fontsize=9, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes, color="#444444")
        for j, wk in enumerate(weight_keys):
            wv = data["weights"][wk]
            wcolor = "#ff6b6b" if wv < 0 else "#51cf66"
            ax.text(0.25, y - 0.055 * (j + 1), wk, fontsize=9,
                    ha="left", transform=ax.transAxes, color="#666666",
                    fontfamily="monospace")
            ax.text(0.75, y - 0.055 * (j + 1), f"{wv:+.1f}", fontsize=9,
                    ha="right", transform=ax.transAxes, color=wcolor,
                    fontfamily="monospace", fontweight="bold")

        # Metrics
        y_m = 0.40
        ax.text(0.5, y_m, "PERFORMANCE", fontsize=9, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes, color="#444444")
        metrics = [
            ("DX", f"{data['dx']:+.2f}m"),
            ("Speed", f"{data['speed']:.3f} m/s"),
            ("Efficiency", f"{data['efficiency']:.5f}"),
            ("Phase Lock", f"{data['phase_lock']:.3f}"),
            ("Entropy", f"{data['entropy']:.3f} bits"),
        ]
        for j, (label, val) in enumerate(metrics):
            ax.text(0.10, y_m - 0.055 * (j + 1), label, fontsize=8,
                    ha="left", transform=ax.transAxes, color="#666666")
            ax.text(0.90, y_m - 0.055 * (j + 1), val, fontsize=8,
                    ha="right", transform=ax.transAxes, color="#1a1a2e",
                    fontfamily="monospace")

        # Members (first 8 + ...)
        y_mem = 0.10
        ax.text(0.5, y_mem, "MEMBERS", fontsize=8, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes, color="#444444")
        show_members = data["members"][:8]
        if len(data["members"]) > 8:
            member_str = ", ".join(show_members) + f"... +{len(data['members'])-8}"
        else:
            member_str = ", ".join(show_members)
        ax.text(0.5, y_mem - 0.04, member_str, fontsize=7,
                ha="center", va="top", transform=ax.transAxes, color="#888888",
                wrap=True, style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = PLOT_DIR / "pol_archetype_cards.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  WROTE {path}")


def main():
    trials = load_data()

    print("\n" + "#" * 72)
    print("#" + " " * 22 + "GAIT LEADERBOARDS" + " " * 31 + "#")
    print("#" + " " * 16 + "79 Political Figures, 4 Gaits" + " " * 25 + "#")
    print("#" * 72)

    # Archetype summary first
    archetypes = {}
    for r in trials:
        a = r["archetype"]
        if a not in archetypes:
            archetypes[a] = []
        archetypes[a].append(r["name"])

    print(f"\n  ARCHETYPE KEY:")
    for arch in ["Whistleblower", "Contrarian", "Assertive", "Majority"]:
        members = archetypes.get(arch, [])
        preview = ", ".join(members[:6])
        if len(members) > 6:
            preview += f"... (+{len(members)-6})"
        print(f"    {arch:<14} [{len(members):2d}]  {preview}")

    # Leaderboard 1: Total displacement (the main event)
    print_leaderboard(
        "DISTANCE CHAMPION — Total Displacement",
        trials, "total_displacement", "Dist (m)")

    # Leaderboard 2: Forward progress (signed DX)
    print_leaderboard(
        "FORWARD MARCH — Net X Displacement (positive = forward)",
        trials, "dx", "DX (m)")

    # Leaderboard 3: Speed
    print_leaderboard(
        "SPEED DEMON — Mean Instantaneous Speed",
        trials, "speed", "Speed (m/s)")

    # Leaderboard 4: Efficiency
    print_leaderboard(
        "EFFICIENCY EXPERT — Distance per Unit Work",
        trials, "efficiency", "Efficiency")

    # Leaderboard 5: Phase lock (coordination)
    print_leaderboard(
        "MOST COORDINATED — Inter-Joint Phase Lock",
        trials, "phase_lock", "Phase Lock")

    # Leaderboard 6: Gait complexity (entropy)
    print_leaderboard(
        "MOST COMPLEX GAIT — Contact State Entropy",
        trials, "entropy", "Entropy (bits)")

    # Leaderboard 7: Work output
    print_leaderboard(
        "HARDEST WORKER — Total Work Proxy",
        trials, "work_proxy", "Work (J)")

    # Leaderboard 8: Lateral deviation
    print_leaderboard(
        "MOST LATERAL — Absolute Y Displacement",
        trials, "abs_dy", "|DY| (m)")

    # Leaderboard 9: Biggest spinner
    print_leaderboard(
        "BIGGEST SPINNER — Absolute Yaw Rotation",
        trials, "yaw_degrees", "Yaw (deg)")

    # Reverse leaderboards
    print_leaderboard(
        "GOING BACKWARDS — Most Negative X Displacement",
        trials, "dx", "DX (m)", ascending=True)

    # Figures
    print("\n\nGenerating leaderboard figures...")
    make_leaderboard_figure(trials)
    make_archetype_card(trials)

    # Save JSON
    leaderboard_data = {
        "archetypes": {},
        "leaderboards": {},
    }
    for r in trials:
        a = r["archetype"]
        if a not in leaderboard_data["archetypes"]:
            leaderboard_data["archetypes"][a] = {
                "count": 0,
                "weights": r["weights"],
                "dx": r["dx"],
                "speed": r["speed"],
                "efficiency": r["efficiency"],
                "phase_lock": r["phase_lock"],
                "entropy": r["entropy"],
                "members": [],
            }
        leaderboard_data["archetypes"][a]["count"] += 1
        leaderboard_data["archetypes"][a]["members"].append({
            "name": r["name"],
            "group": r["group"],
        })

    for metric_key, metric_name in [
        ("total_displacement", "distance"),
        ("speed", "speed"),
        ("efficiency", "efficiency"),
        ("phase_lock", "coordination"),
        ("entropy", "complexity"),
        ("work_proxy", "work"),
        ("yaw_degrees", "spin"),
        ("abs_dy", "lateral"),
    ]:
        sorted_t = sorted(trials, key=lambda r: r[metric_key], reverse=True)
        leaderboard_data["leaderboards"][metric_name] = [
            {"rank": i + 1, "name": r["name"], "group": r["group"],
             "archetype": r["archetype"], "value": r[metric_key]}
            for i, r in enumerate(sorted_t)
        ]

    out_path = PROJECT / "artifacts" / "politics_leaderboards.json"
    with open(out_path, "w") as f:
        json.dump(leaderboard_data, f, indent=2, cls=NumpyEncoder)
    print(f"  WROTE {out_path}")

    # Final summary
    print(f"\n{'='*72}")
    print("  SUMMARY: The LLM's Political Taxonomy")
    print(f"{'='*72}")
    print(f"""
  79 political figures collapsed into exactly 4 robot gaits:

  1. THE WHISTLEBLOWERS (2): Assange, Snowden
     The fastest, most efficient, walk BACKWARD — the only pair
     with a truly unique gait. DX = -5.64m, Speed = 0.433 m/s.

  2. THE CONTRARIANS (2): McCarthy, Spicer
     Also walk backward but slower. Highest work output and
     gait complexity. DX = -1.19m, Entropy = 1.239 bits.

  3. THE ASSERTIVE (25): Trump, Putin, Musk, Bannon, Bolton, Flynn,
     Cruz, Elon, Farage, Giuliani, Gates, Epstein, Bernie, Castro,
     Cheney, Sanders, Scalia, Sessions, Pruitt, Barr, Barrett,
     Conway, McConnell, Kushner, Mercer
     Walk forward with highest coordination.
     DX = +1.55m, Phase Lock = 0.942.

  4. THE MAJORITY (50): Biden, Clinton, Hillary, Pelosi, Ivanka,
     Melania, Tiffany, Lara, Pence, Romney, Mueller, Comey,
     Warren, Schumer, Hicks, Haley, Huckabee, Kavanaugh, Kennedy,
     and 31 others from ALL groups.
     The default gait. Slowest but steadiest.
     DX = +1.18m, Phase Lock = 0.886.
""")


if __name__ == "__main__":
    main()
