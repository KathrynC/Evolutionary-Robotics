#!/usr/bin/env python3
"""
render_dictionary_pdf.py

Renders the motion gait dictionary as a nicely formatted multi-page PDF
using matplotlib's PdfPages backend.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm

# Register PingFang for CJK support, fall back to monospace
_CJK_FONT = None
for _candidate in [
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]:
    if Path(_candidate).exists():
        _CJK_FONT = _candidate
        fm.fontManager.addfont(_candidate)
        break

PROJECT = Path(__file__).resolve().parent
DICT_PATH = PROJECT / "artifacts" / "motion_gait_dictionary_v2.json"
OUT_PATH = PROJECT / "artifacts" / "motion_gait_dictionary_v2.pdf"

# Layout constants — landscape letter, larger to fit all columns
PAGE_W, PAGE_H = 14, 8.5
LEFT = 0.03
RIGHT = 0.97
FONT_SECTION = 10
FONT_BODY = 6.5
FONT_HEADER = 7
LINE_H = 0.017

# Column positions (fraction of page width) — tuned to actual content widths
# Content char widths at 6.5pt monospace on a 14" page: ~0.0033 per char
#   #: 2 chars, Word: 14 chars, Model: 16 chars, Lang: 2 chars,
#   Weights: 35 chars, Behavior: ~55 chars
COL = {
    "#":       LEFT,
    "Word":    LEFT + 0.025,
    "Model":   LEFT + 0.10,
    "Lang":    LEFT + 0.19,
    "Weights": LEFT + 0.22,
    "Behav":   LEFT + 0.38,
}

COL_LABELS = ["#", "Word", "Model", "Lang",
              "Weights  w03  w04  w13  w14  w23  w24",
              "DX     DY    yaw   spd   cv    work  eff     torso"]


def _has_cjk(text):
    return any(ord(c) > 0x2E80 for c in text)


def _font_kwargs(text, size=FONT_BODY):
    """Return font kwargs — use CJK font if text contains CJK chars."""
    if _CJK_FONT and _has_cjk(text):
        prop = fm.FontProperties(fname=_CJK_FONT, size=size)
        return {"fontproperties": prop}
    return {"family": "monospace", "fontsize": size}


def load_dictionary():
    with open(DICT_PATH) as f:
        return json.load(f)


def draw_title_page(pdf, data):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    meta = data["metadata"]
    y = 0.72
    ax.text(0.5, y, "Motion Gait Dictionary", ha="center", va="center",
            fontsize=22, fontweight="bold", family="monospace")
    y -= 0.06
    ax.text(0.5, y, "Semantic Motion Concepts \u2192 Neural Network Weights",
            ha="center", va="center", fontsize=13, family="monospace", color="#444")
    y -= 0.04
    ax.text(0.5, y, "for a 3-Link PyBullet Walking Robot",
            ha="center", va="center", fontsize=13, family="monospace", color="#444")

    y -= 0.08
    lines = [
        f"{meta['n_concepts']} concepts  \u00b7  {meta['n_total_entries']} entries",
        f"Robot: {meta['robot']}",
        f"Simulation: {meta['simulation']}",
        f"Generated: {meta['generated']}",
    ]
    for line in lines:
        ax.text(0.5, y, line, ha="center", va="center",
                fontsize=10, family="monospace", color="#666")
        y -= 0.035

    # Weight key legend
    y -= 0.04
    ax.text(0.5, y, "Weight Key", ha="center", va="center",
            fontsize=11, fontweight="bold", family="monospace")
    y -= 0.035
    for wk, desc in meta["weight_keys"].items():
        ax.text(0.5, y, f"{wk}: {desc}", ha="center", va="center",
                fontsize=9, family="monospace", color="#555")
        y -= 0.028

    # Sources
    y -= 0.03
    ax.text(0.5, y, "Sources", ha="center", va="center",
            fontsize=10, fontweight="bold", family="monospace")
    y -= 0.03
    for src in meta["sources"]:
        ax.text(0.5, y, src, ha="center", va="center",
                fontsize=9, family="monospace", color="#555")
        y -= 0.025

    pdf.savefig(fig)
    plt.close(fig)


def format_weights(w):
    parts = []
    for k in ["w03", "w04", "w13", "w14", "w23", "w24"]:
        v = w.get(k, 0)
        parts.append(f"{v:+.1f}")
    return "  ".join(parts)


def format_behavior(b):
    return (f"{b['dx']:+6.1f}  {b['dy']:+5.1f}  {b['yaw_rad']:+5.1f}"
            f"  {b['mean_speed']:5.2f}  {b['speed_cv']:4.2f}"
            f"  {b['work']:6.0f}  {b['efficiency']:.5f}"
            f"  {b['torso_duty']:.2f}")


def draw_concept_pages(pdf, data):
    concepts = data["concepts"]
    sorted_concepts = sorted(concepts.items(), key=lambda x: -x[1]["n_matches"])

    col_xs = [COL["#"], COL["Word"], COL["Model"], COL["Lang"],
              COL["Weights"], COL["Behav"]]

    def new_page():
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig, ax

    def draw_table_header(ax, y):
        for cx, label in zip(col_xs, COL_LABELS):
            ax.text(cx, y, label, fontsize=FONT_HEADER, fontweight="bold",
                    family="monospace", va="center", color="#333")
        y -= 0.004
        ax.plot([LEFT, RIGHT], [y, y], color="#999", linewidth=0.6)
        return y - LINE_H

    fig, ax = new_page()
    y = 1 - 0.04  # top margin

    for concept_id, entry in sorted_concepts:
        n = entry["n_matches"]
        models = entry["models"]
        langs = entry["languages"]
        words = entry["words"]

        # Estimate space needed
        needed = (3 + len(entry["synonyms"])) * LINE_H + 0.015

        if y - needed < 0.04:
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = new_page()
            y = 1 - 0.04

        # Concept header line
        header_text = f"{concept_id}"
        summary = f"{n} matches  |  {len(models)} models  |  {len(langs)} langs"
        ax.text(LEFT, y, header_text, fontsize=FONT_SECTION, fontweight="bold",
                family="monospace", va="center", color="#1a1a1a")
        ax.text(LEFT + 0.12, y, summary, fontsize=FONT_BODY,
                family="monospace", va="center", color="#666")

        # Words on same line, further right — CJK aware
        words_str = ", ".join(words)
        ax.text(LEFT + 0.32, y, words_str, va="center", color="#555",
                **_font_kwargs(words_str))
        y -= LINE_H * 1.1

        # Thin separator
        ax.plot([LEFT, RIGHT], [y + LINE_H * 0.35, y + LINE_H * 0.35],
                color="#ddd", linewidth=0.3)

        # Column headers
        y = draw_table_header(ax, y)

        # Entries
        for i, syn in enumerate(entry["synonyms"]):
            if y < 0.04 + LINE_H:
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = new_page()
                y = 1 - 0.04
                ax.text(LEFT, y, f"{concept_id} (cont.)", fontsize=FONT_SECTION,
                        fontweight="bold", family="monospace", va="center",
                        color="#1a1a1a")
                y -= LINE_H * 1.1
                y = draw_table_header(ax, y)

            # Alternating row background
            if i % 2 == 0:
                ax.fill_between([LEFT, RIGHT],
                                y - LINE_H * 0.4, y + LINE_H * 0.5,
                                color="#f5f5f5", linewidth=0)

            word_text = syn["word"][:14]
            # Render each column
            ax.text(COL["#"], y, str(i + 1), fontsize=FONT_BODY,
                    family="monospace", va="center", color="#222")
            ax.text(COL["Word"], y, word_text, va="center", color="#222",
                    **_font_kwargs(word_text))
            ax.text(COL["Model"], y, syn["model"][:16], fontsize=FONT_BODY,
                    family="monospace", va="center", color="#222")
            ax.text(COL["Lang"], y, syn["language"], fontsize=FONT_BODY,
                    family="monospace", va="center", color="#222")
            ax.text(COL["Weights"], y, format_weights(syn["weights"]),
                    fontsize=FONT_BODY, family="monospace", va="center",
                    color="#222")
            ax.text(COL["Behav"], y, format_behavior(syn["behavior"]),
                    fontsize=FONT_BODY, family="monospace", va="center",
                    color="#222")

            y -= LINE_H

        # Gap between concepts
        y -= LINE_H * 0.6

    pdf.savefig(fig)
    plt.close(fig)


def main():
    print(f"Loading dictionary from {DICT_PATH.name}...")
    data = load_dictionary()
    n_concepts = data["metadata"]["n_concepts"]
    n_entries = data["metadata"]["n_total_entries"]
    print(f"  {n_concepts} concepts, {n_entries} entries")

    print(f"Rendering PDF...")
    with PdfPages(str(OUT_PATH)) as pdf:
        draw_title_page(pdf, data)
        draw_concept_pages(pdf, data)

    print(f"Saved to {OUT_PATH}")
    print(f"  {OUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
