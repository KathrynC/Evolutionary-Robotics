#!/usr/bin/env python3
"""plot_gaitspace.py — Publication-quality figures for Synapse Gait Zoo v2.

Reads synapse_gait_zoo_v2.json, produces 7 PNG figures in artifacts/plots/.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
ZOO_PATH = os.path.join(_DIR, "synapse_gait_zoo_v2.json")
OUT_DIR = os.path.join(_DIR, "artifacts", "plots")

# ── notable gaits for labeling ─────────────────────────────────────────────
NOTABLE = {
    "43_hidden_cpg_champion": "CPG Champion",
    "18_curie":               "Curie",
    "44_spinner_champion":    "Spinner",
    "7_fuller_dymaxion":      "Fuller",
    "93_borges_mirror":       "Borges Mirror",
    "68_grunbaum_penrose":    "Grünbaum",
    "56_evolved_crab_v2":     "Crab",
}

# ── color scheme ───────────────────────────────────────────────────────────
# persona_gaits → gray; 10 other categories → tab10
_TAB10 = plt.cm.tab10.colors
_OTHER_CATS = sorted([
    "bifurcation_gaits", "crab_walkers", "cross_wired_cpg", "evolved",
    "hidden_neurons", "homework", "market_mathematics", "pareto_walk_spin",
    "spinners", "time_signatures",
])
CAT_COLORS = {"persona_gaits": "#7f7f7f"}
for _i, _cat in enumerate(_OTHER_CATS):
    CAT_COLORS[_cat] = _TAB10[_i]


# ── data loading ───────────────────────────────────────────────────────────
def load_zoo(path=ZOO_PATH):
    """Flatten the nested JSON into a list of dicts with all analytics at top level."""
    with open(path) as f:
        zoo = json.load(f)
    gaits = []
    for cat_name, cat_data in zoo["categories"].items():
        for gait_name, gait_data in cat_data.get("gaits", {}).items():
            g = {"name": gait_name, "category": cat_name}
            a = gait_data.get("analytics", {})
            # outcome
            for k, v in a.get("outcome", {}).items():
                g[k] = v
            # contact (skip nested lists/dicts)
            for k, v in a.get("contact", {}).items():
                if not isinstance(v, (list, dict)):
                    g[k] = v
            # coordination
            coord = a.get("coordination", {})
            g["phase_lock_score"] = coord.get("phase_lock_score", 0)
            g["delta_phi_rad"] = coord.get("delta_phi_rad", 0)
            j0 = coord.get("joint_0", {})
            g["joint_0_freq"] = j0.get("dominant_freq_hz", 0)
            g["joint_0_amp"] = j0.get("dominant_amplitude", 0)
            j1 = coord.get("joint_1", {})
            g["joint_1_freq"] = j1.get("dominant_freq_hz", 0)
            g["joint_1_amp"] = j1.get("dominant_amplitude", 0)
            # rotation_axis
            rot = a.get("rotation_axis", {})
            dom = rot.get("axis_dominance", [0.33, 0.33, 0.34])
            g["roll_dominance"] = dom[0]
            g["pitch_dominance"] = dom[1]
            g["yaw_dominance"] = dom[2]
            g["axis_switching_rate"] = rot.get("axis_switching_rate_hz", 0)
            per = rot.get("periodicity", {})
            g["roll_freq"] = per.get("roll_freq_hz", 0)
            g["pitch_freq"] = per.get("pitch_freq_hz", 0)
            g["yaw_freq"] = per.get("yaw_freq_hz", 0)
            # preserved
            pres = a.get("preserved", {})
            g["attractor_type"] = pres.get("attractor_type", "unknown")
            g["pareto_optimal"] = pres.get("pareto_optimal", False)
            gaits.append(g)
    print(f"Loaded {len(gaits)} gaits")
    return gaits


# ── helpers ────────────────────────────────────────────────────────────────
def correlation_r(x, y):
    """Pearson r, numpy-only."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = np.sqrt(((x - mx) ** 2).sum() * ((y - my) ** 2).sum())
    return num / den if den > 0 else 0.0


def ternary_to_cartesian(a, b, c):
    """Barycentric (a=top, b=bottom-left, c=bottom-right) → 2D Cartesian.

    Vertices: top=(0.5, √3/2), bottom-left=(0,0), bottom-right=(1,0).
    """
    s = np.asarray(a, float) + np.asarray(b, float) + np.asarray(c, float)
    a, b, c = np.asarray(a, float) / s, np.asarray(b, float) / s, np.asarray(c, float) / s
    x = c + 0.5 * a
    y = (np.sqrt(3) / 2) * a
    return x, y


def clean_ax(ax):
    """Remove top/right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def category_scatter(ax, gaits, x_key, y_key, s=20, alpha=0.7):
    """Scatter color-coded by category. Draws persona_gaits first (background)."""
    for cat in ["persona_gaits"] + _OTHER_CATS:
        pts = [g for g in gaits if g["category"] == cat]
        if not pts:
            continue
        xs = [g[x_key] for g in pts]
        ys = [g[y_key] for g in pts]
        color = CAT_COLORS[cat]
        zorder = 1 if cat == "persona_gaits" else 2
        ax.scatter(xs, ys, c=[color], s=s, alpha=alpha, edgecolors="none",
                   label=cat.replace("_", " "), zorder=zorder)


def label_notable(ax, gaits, x_key, y_key, fontsize=7):
    """Annotate notable gaits with arrows."""
    for g in gaits:
        label = NOTABLE.get(g["name"])
        if label:
            ax.annotate(
                label, (g[x_key], g[y_key]),
                textcoords="offset points", xytext=(8, 6),
                fontsize=fontsize, fontstyle="italic",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6),
            )


def save_fig(fig, name):
    """Save figure to artifacts/plots/ with consistent settings."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {path}")


def _get(gaits, name):
    """Get a single gait dict by name."""
    for g in gaits:
        if g["name"] == name:
            return g
    return None


# ── Figure 1: Phase Lock Bimodality ───────────────────────────────────────
def fig01(gaits):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    scores = np.array([g["phase_lock_score"] for g in gaits])

    # Left: histogram
    ax1.hist(scores, bins=25, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Phase Lock Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Phase Lock Distribution (Bimodal)")
    clean_ax(ax1)

    # Right: scatter — phase lock vs speed
    category_scatter(ax2, gaits, "mean_speed", "phase_lock_score")
    label_notable(ax2, gaits, "mean_speed", "phase_lock_score")
    ax2.set_xlabel("Mean Speed")
    ax2.set_ylabel("Phase Lock Score")
    ax2.set_title("Phase Lock vs Speed")
    clean_ax(ax2)

    fig.tight_layout()
    save_fig(fig, "fig01_phase_lock_bimodal.png")


# ── Figure 2: Contact Entropy Independence ────────────────────────────────
def fig02(gaits):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [
        ("mean_speed",        "Speed"),
        ("phase_lock_score",  "Phase Lock"),
        ("distance_per_work", "Efficiency"),
    ]
    for ax, (x_key, xlabel) in zip(axes, pairs):
        xs = np.array([g[x_key] for g in gaits])
        ys = np.array([g["contact_entropy_bits"] for g in gaits])
        category_scatter(ax, gaits, x_key, "contact_entropy_bits")
        # For efficiency, clip axis to exclude extreme outlier and compute r on inliers
        if x_key == "distance_per_work":
            clip = np.percentile(xs, 97)
            mask = xs <= clip
            r = correlation_r(xs[mask], ys[mask])
            ax.set_xlim(-clip * 0.05, clip * 1.15)
        else:
            r = correlation_r(xs, ys)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Contact Entropy (bits)")
        ax.set_title(f"r = {r:.3f}")
        clean_ax(ax)

    fig.suptitle("Contact Entropy Is Independent of Performance", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig02_contact_entropy_independence.png")


# ── Figure 3: Axis Dominance ──────────────────────────────────────────────
def fig03(gaits):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # -- Left: ternary simplex --
    # Triangle outline: Roll(top) → Pitch(bottom-left) → Yaw(bottom-right) → Roll
    tri_x, tri_y = ternary_to_cartesian(
        [1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]
    )
    ax1.plot(tri_x, tri_y, "k-", lw=1)
    ax1.text(tri_x[0], tri_y[0] + 0.03, "Roll", ha="center", va="bottom", fontweight="bold")
    ax1.text(tri_x[1] - 0.03, tri_y[1] - 0.03, "Pitch", ha="right", va="top", fontweight="bold")
    ax1.text(tri_x[2] + 0.03, tri_y[2] - 0.03, "Yaw", ha="left", va="top", fontweight="bold")

    # Plot gaits on simplex
    for cat in ["persona_gaits"] + _OTHER_CATS:
        pts = [g for g in gaits if g["category"] == cat]
        if not pts:
            continue
        roll = [g["roll_dominance"] for g in pts]
        pitch = [g["pitch_dominance"] for g in pts]
        yaw = [g["yaw_dominance"] for g in pts]
        x, y = ternary_to_cartesian(roll, pitch, yaw)
        color = CAT_COLORS[cat]
        zorder = 1 if cat == "persona_gaits" else 2
        ax1.scatter(x, y, c=[color], s=20, alpha=0.7, edgecolors="none", zorder=zorder)

    # Label notables on ternary
    for g in gaits:
        label = NOTABLE.get(g["name"])
        if label:
            x, y = ternary_to_cartesian(
                g["roll_dominance"], g["pitch_dominance"], g["yaw_dominance"]
            )
            ax1.annotate(
                label, (x, y), textcoords="offset points", xytext=(8, 4),
                fontsize=6, fontstyle="italic",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5),
            )

    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Axis Dominance Simplex")

    # -- Right: bar chart of dominance categories --
    dom_cats = {"Roll": 0, "Pitch": 0, "Yaw": 0}
    for g in gaits:
        r, p, y = g["roll_dominance"], g["pitch_dominance"], g["yaw_dominance"]
        if r >= p and r >= y:
            dom_cats["Roll"] += 1
        elif p >= r and p >= y:
            dom_cats["Pitch"] += 1
        else:
            dom_cats["Yaw"] += 1

    bars = ax2.bar(
        dom_cats.keys(), dom_cats.values(),
        color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="white",
    )
    for bar, count in zip(bars, dom_cats.values()):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(count), ha="center", va="bottom", fontweight="bold",
        )
    ax2.set_ylabel("Number of Gaits")
    ax2.set_title("Dominant Rotation Axis")
    clean_ax(ax2)

    fig.tight_layout()
    save_fig(fig, "fig03_axis_dominance.png")


# ── Figure 4: Speed vs Efficiency ─────────────────────────────────────────
def fig04(gaits):
    fig, ax = plt.subplots(figsize=(8, 6))

    speeds = np.array([g["mean_speed"] for g in gaits])
    effs = np.array([g["distance_per_work"] for g in gaits])

    # Use IQR fence to exclude only truly extreme outliers
    q1, q3 = np.percentile(effs, [25, 75])
    eff_fence = q3 + 5 * (q3 - q1)
    inlier_max = effs[effs <= eff_fence].max() if (effs <= eff_fence).any() else effs.max()
    eff_ylim = inlier_max * 1.25

    category_scatter(ax, gaits, "mean_speed", "distance_per_work", s=30)

    # Median quadrant lines
    med_s = np.median(speeds)
    med_e = np.median(effs)
    ax.axvline(med_s, color="0.7", ls="--", lw=0.8, zorder=0)
    ax.axhline(med_e, color="0.7", ls="--", lw=0.8, zorder=0)

    # Pareto stars
    for g in gaits:
        if g.get("pareto_optimal"):
            ax.scatter(
                g["mean_speed"], g["distance_per_work"],
                marker="*", s=150, c="gold", edgecolors="k",
                linewidths=0.5, zorder=5,
            )

    # Label notables (only those within view)
    for g in gaits:
        label = NOTABLE.get(g["name"])
        if label and g["distance_per_work"] <= eff_ylim:
            ax.annotate(
                label, (g["mean_speed"], g["distance_per_work"]),
                textcoords="offset points", xytext=(8, 6),
                fontsize=7, fontstyle="italic",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6),
            )

    # Annotate outliers above view with arrow at top
    for g in gaits:
        if g["distance_per_work"] > eff_ylim:
            label = NOTABLE.get(g["name"], g["name"].split("_", 1)[1])
            ax.annotate(
                f'{label}\n(eff={g["distance_per_work"]:.2e})',
                xy=(g["mean_speed"], eff_ylim * 0.96),
                fontsize=7, fontstyle="italic", ha="center", va="top",
            )

    ax.set_ylim(-eff_ylim * 0.03, eff_ylim)
    ax.set_xlabel("Mean Speed")
    ax.set_ylabel("Distance per Work (Efficiency)")
    ax.set_title("Speed–Efficiency Landscape")
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "fig04_speed_efficiency.png")


# ── Figure 5: Champion Comparison ─────────────────────────────────────────
def fig05(gaits):
    trio = {
        "CPG Champion": _get(gaits, "43_hidden_cpg_champion"),
        "Curie":        _get(gaits, "18_curie"),
        "Spinner":      _get(gaits, "44_spinner_champion"),
    }
    metrics = [
        ("mean_speed",          "Speed"),
        ("distance_per_work",   "Efficiency"),
        ("phase_lock_score",    "Phase Lock"),
        ("contact_entropy_bits","Entropy"),
        ("roll_dominance",      "Roll Dom."),
        ("axis_switching_rate", "Axis Switch"),
    ]

    # Normalize each metric to [0, 1] using robust percentile range
    ranges = {}
    for key, _ in metrics:
        vals = np.array([g[key] for g in gaits])
        lo = np.percentile(vals, 2)
        hi = np.percentile(vals, 98)
        ranges[key] = (lo, hi)

    def norm(val, key):
        lo, hi = ranges[key]
        return np.clip((val - lo) / (hi - lo), 0, 1) if hi > lo else 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, (label, g) in enumerate(trio.items()):
        vals = [norm(g[key], key) for key, _ in metrics]
        ax.bar(x + i * width, vals, width, label=label, color=colors[i],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Normalized Score")
    ax.set_title("Champion Comparison: CPG vs Curie vs Spinner")
    ax.legend()
    ax.set_ylim(0, 1.15)
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "fig05_champion_comparison.png")


# ── Figure 6: Topology Bifurcation (Radar) ────────────────────────────────
def fig06(gaits):
    g43 = _get(gaits, "43_hidden_cpg_champion")
    g44 = _get(gaits, "44_spinner_champion")

    labels = ["Speed", "|DX|", "Phase Lock", "Roll Dom.", "Entropy", "|Yaw|"]
    keys_fn = [
        lambda g: g["mean_speed"],
        lambda g: abs(g["dx"]),
        lambda g: g["phase_lock_score"],
        lambda g: g["roll_dominance"],
        lambda g: g["contact_entropy_bits"],
        lambda g: abs(g["yaw_net_rad"]),
    ]

    v43 = [fn(g43) for fn in keys_fn]
    v44 = [fn(g44) for fn in keys_fn]

    # Normalize each axis to max of the two
    maxvals = [max(a, b) if max(a, b) > 0 else 1 for a, b in zip(v43, v44)]
    v43_n = [a / m for a, m in zip(v43, maxvals)]
    v44_n = [a / m for a, m in zip(v44, maxvals)]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Close the polygon
    v43_n += v43_n[:1]
    v44_n += v44_n[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    ax.plot(angles, v43_n, "o-", color="#e74c3c", linewidth=2,
            label="Gait 43 (CPG Champion)")
    ax.fill(angles, v43_n, alpha=0.15, color="#e74c3c")
    ax.plot(angles, v44_n, "s-", color="#3498db", linewidth=2,
            label="Gait 44 (Spinner)")
    ax.fill(angles, v44_n, alpha=0.15, color="#3498db")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("Same Topology, Opposite Behavior\nGait 43 vs 44", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05))

    fig.tight_layout()
    save_fig(fig, "fig06_topology_bifurcation.png")


# ── Figure 7: Category Overview (2×2) ─────────────────────────────────────
def fig07(gaits):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    projections = [
        ("mean_speed", "phase_lock_score",      "Speed",          "Phase Lock"),
        ("mean_speed", "contact_entropy_bits",   "Speed",          "Contact Entropy"),
        ("roll_dominance", "pitch_dominance",    "Roll Dominance", "Pitch Dominance"),
        ("dx",         "yaw_net_rad",            "DX (displacement)", "Yaw (rad)"),
    ]
    for ax, (xk, yk, xl, yl) in zip(axes.flat, projections):
        category_scatter(ax, gaits, xk, yk)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        clean_ax(ax)

    # Shared legend
    handles = []
    labels = []
    for cat in ["persona_gaits"] + _OTHER_CATS:
        color = CAT_COLORS[cat]
        handles.append(plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=color, markersize=6,
        ))
        labels.append(cat.replace("_", " "))

    fig.legend(
        handles, labels, loc="lower center", ncol=4,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Gaitspace Projections by Category", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_fig(fig, "fig07_category_overview.png")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gaits = load_zoo()
    fig01(gaits)
    fig02(gaits)
    fig03(gaits)
    fig04(gaits)
    fig05(gaits)
    fig06(gaits)
    fig07(gaits)
    print("Done — all 7 figures written.")
