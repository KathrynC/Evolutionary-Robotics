#!/usr/bin/env python3
"""
generate_paper_pdf.py

Render the paper draft as a multi-page PDF with key figures embedded,
using matplotlib's PdfPages backend (no LaTeX required).
"""

import os
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

PROJECT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(PROJECT, "artifacts", "plots")
DRAFT = os.path.join(PROJECT, "artifacts", "paper_draft.md")
OUT_PDF = os.path.join(PROJECT, "artifacts", "paper_draft.pdf")

# Page dimensions (inches) — US letter
PW, PH = 8.5, 11.0
MARGIN = 0.75
TEXT_W = PW - 2 * MARGIN
TEXT_H = PH - 2 * MARGIN

# Fonts
TITLE_SIZE = 18
SUBTITLE_SIZE = 11
HEADING_SIZE = 14
SUBHEADING_SIZE = 11.5
BODY_SIZE = 9.5
CAPTION_SIZE = 8.5
TABLE_SIZE = 8.0


def new_text_page(pdf, bg="#ffffff"):
    """Create a new blank page and return (fig, ax)."""
    fig = plt.figure(figsize=(PW, PH), facecolor=bg)
    ax = fig.add_axes([MARGIN/PW, MARGIN/PH, TEXT_W/PW, TEXT_H/PH])
    ax.set_xlim(0, TEXT_W)
    ax.set_ylim(0, TEXT_H)
    ax.axis("off")
    ax.set_facecolor(bg)
    return fig, ax


def emit_text_page(pdf, fig):
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def add_figure_page(pdf, img_path, caption="", fig_num=""):
    """Add a full-page figure with caption."""
    if not os.path.exists(img_path):
        print(f"  WARNING: {img_path} not found, skipping")
        return

    img = Image.open(img_path)
    img_w, img_h = img.size
    aspect = img_h / img_w

    fig = plt.figure(figsize=(PW, PH), facecolor="white")

    # Reserve space for caption at bottom
    cap_h = 0.08 if caption else 0.02
    img_area_h = 1.0 - cap_h - 0.04  # top margin

    # Compute image axes to fit with correct aspect ratio
    avail_w = PW - 2 * MARGIN
    avail_h = (PH * img_area_h) - MARGIN
    display_w = avail_w
    display_h = display_w * aspect
    if display_h > avail_h:
        display_h = avail_h
        display_w = display_h / aspect

    left = (PW - display_w) / 2 / PW
    bottom = cap_h + (avail_h - display_h) / 2 / PH + MARGIN / PH
    width = display_w / PW
    height = display_h / PH

    ax = fig.add_axes([left, bottom, width, height])
    ax.imshow(np.array(img))
    ax.axis("off")

    if caption:
        fig.text(0.5, cap_h * 0.6, f"{fig_num}{caption}",
                 fontsize=CAPTION_SIZE, ha="center", va="center",
                 wrap=True, fontfamily="serif", color="#333333",
                 fontstyle="italic")

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    print(f"  Added figure: {os.path.basename(img_path)}")


def wrap_text(text, width=90):
    """Wrap text to specified character width."""
    lines = []
    for para in text.split("\n"):
        if para.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width))
    return lines


def render_section(pdf, title, body_text, level=2):
    """Render a section of body text across as many pages as needed."""
    lines = wrap_text(body_text, width=95)

    line_height = 0.155  # inches per line
    heading_height = 0.4
    max_lines_per_page = int((TEXT_H - heading_height) / line_height)

    first_page = True
    while lines:
        fig, ax = new_text_page(pdf)
        y = TEXT_H - 0.1

        if first_page and title:
            fs = HEADING_SIZE if level == 2 else SUBHEADING_SIZE
            fw = "bold"
            ax.text(0, y, title, fontsize=fs, fontweight=fw,
                    fontfamily="serif", color="#1a1a2e", va="top")
            y -= heading_height
            first_page = False

        page_lines = lines[:max_lines_per_page]
        lines = lines[max_lines_per_page:]

        for line in page_lines:
            if line.startswith("|") or line.startswith("-"):
                ax.text(0, y, line, fontsize=TABLE_SIZE,
                        fontfamily="monospace", color="#333333", va="top")
            elif line.startswith("```"):
                pass  # skip code fences
            elif line.strip() == "":
                y -= line_height * 0.5
                continue
            else:
                ax.text(0, y, line, fontsize=BODY_SIZE,
                        fontfamily="serif", color="#333333", va="top")
            y -= line_height

        emit_text_page(pdf, fig)


def parse_sections(md_text):
    """Parse markdown into sections: list of (level, title, body)."""
    sections = []
    current_title = ""
    current_level = 0
    current_body = []

    for line in md_text.split("\n"):
        if line.startswith("## ") and not line.startswith("### "):
            if current_title or current_body:
                sections.append((current_level, current_title, "\n".join(current_body)))
            current_title = line[3:].strip()
            current_level = 2
            current_body = []
        elif line.startswith("### "):
            if current_title or current_body:
                sections.append((current_level, current_title, "\n".join(current_body)))
            current_title = line[4:].strip()
            current_level = 3
            current_body = []
        elif line.startswith("# "):
            # Skip main title (handled separately)
            continue
        elif line.startswith("**Kathryn"):
            continue
        elif line.strip() == "---":
            continue
        else:
            current_body.append(line)

    if current_title or current_body:
        sections.append((current_level, current_title, "\n".join(current_body)))

    return sections


# Map section titles to figures that should follow them
FIGURE_PLACEMENTS = {
    "4.3 Core Results": [
        ("sr_fig01_dx_by_condition.png", "Displacement distribution by experimental condition (box plot). Baseline (gray) shows high variance; LLM conditions cluster near zero with occasional outliers.", "Figure 1. "),
        ("sr_fig06_diversity.png", "Behavioral diversity: PCA of Beer-framework metrics. LLM conditions (colored) occupy a small submanifold; baseline (gray) fills the space.", "Figure 2. "),
    ],
    "5.2 Extreme Collapse: Four Archetypes": [
        ("cel_fig03_weight_heatmap.png", "Synapse weights for all 132 celebrity names, sorted by domain then DX. Color: red = negative, blue = positive. Note the extreme collapse into 4 distinct patterns.", "Figure 3. "),
        ("cel_fig04_cluster_composition.png", "Top 15 weight clusters with domain composition. Clusters span multiple domains; domain boundaries dissolve.", "Figure 4. "),
        ("cel_fig01_dx_by_domain.png", "Displacement distribution by celebrity domain (box plot). Controversial figures show highest variance.", "Figure 5. "),
    ],
    "6.1 The Semantic-to-Weight Map (F: Sem → Wt)": [
        ("cs_fig01_weight_pca.png", "Weight-space PCA colored by experimental condition. LLM conditions cluster tightly; baseline fills the space.", "Figure 6. "),
        ("cel_fig06_faithfulness_by_domain.png", "Faithfulness ratio by celebrity domain. Sports and cultural have the lowest faithfulness; controversial the highest.", "Figure 7. "),
    ],
    "6.2 The Weight-to-Behavior Map (G: Wt → Beh)": [
        ("cs_fig04_wt_vs_beh.png", "Weight distance vs. behavioral distance. Strong positive correlation (Mantel r = 0.733) confirms local structure preservation.", "Figure 8. "),
    ],
    "6.4 Smooth Basins and Patch Structure": [
        ("cs_fig06_sheaf_patches.png", "Patch map: PCA of atlas points colored by connected component (smooth patch). LLM points (overlaid) concentrate in few patches.", "Figure 9. "),
    ],
    "6.5 Information Geometry": [
        ("cel_fig08_dimensionality.png", "Effective dimensionality (participation ratio) by condition. Celebrity and places are most collapsed; baseline nearly fills 6D space.", "Figure 10. "),
    ],
    "7.2 Results": [
        ("le_fig01_fitness_trajectories.png", "Evolution fitness trajectories: LLM-seeded (colored) vs. random baselines (gray). Revelation reaches 85.09m.", "Figure 11. "),
    ],
    "8.3 Direct Perturbation Probing (259 simulations)": [
        ("pp_fig02_cliffiness_by_condition.png", "Directly measured cliffiness at LLM-generated weight vectors, by condition. Dashed line = atlas median.", "Figure 12. "),
    ],
    "9.2 Behavioral Spectral Analysis": [
        ("hf_fig04_spectral_gaps.png", "Spectral gaps in behavioral covariance by condition. Places has the sharpest gap; baseline is nearly uniform.", "Figure 13. "),
    ],
}


def main():
    print("Generating paper PDF...")

    with open(DRAFT) as f:
        md_text = f.read()

    sections = parse_sections(md_text)

    with PdfPages(OUT_PDF) as pdf:
        # ── Title page ───────────────────────────────────────────────────
        fig, ax = new_text_page(pdf, bg="#faf9f6")
        y = TEXT_H * 0.75

        ax.text(TEXT_W / 2, y,
                "Reality Is What Doesn't Go Away\nWhen You Change the Physics Engine",
                fontsize=TITLE_SIZE, fontweight="bold", ha="center", va="top",
                fontfamily="serif", color="#1a1a2e", linespacing=1.4)

        y -= 1.2
        ax.text(TEXT_W / 2, y,
                "Structural Transfer from Language Models\nThrough Physical Substrates",
                fontsize=SUBTITLE_SIZE, ha="center", va="top",
                fontfamily="serif", color="#555555", fontstyle="italic",
                linespacing=1.3)

        y -= 1.0
        ax.text(TEXT_W / 2, y, "Kathryn Cramer",
                fontsize=12, ha="center", va="top",
                fontfamily="serif", color="#333333")
        y -= 0.3
        ax.text(TEXT_W / 2, y, "University of Vermont",
                fontsize=10, ha="center", va="top",
                fontfamily="serif", color="#666666")

        y -= 1.2
        ax.text(TEXT_W / 2, y, "Draft — February 2026",
                fontsize=9, ha="center", va="top",
                fontfamily="serif", color="#888888")

        # Key stats box
        y -= 1.0
        stats = [
            "706 LLM-mediated trials  •  ~25,000 supporting simulations",
            "7 semantic conditions  •  3 linked projects  •  132 celebrity names → 4 gaits",
            "LLM-seeded evolution: 85.09m (vs 48.41m random best)  •  Synonym convergence: 6/6",
        ]
        for line in stats:
            ax.text(TEXT_W / 2, y, line,
                    fontsize=8.5, ha="center", va="top",
                    fontfamily="monospace", color="#666666")
            y -= 0.22

        emit_text_page(pdf, fig)

        # ── Render all sections with figures ─────────────────────────────
        for level, title, body in sections:
            # Build section label for matching
            section_key = title

            render_section(pdf, title, body, level=level)

            # Insert figures after matching sections
            if section_key in FIGURE_PLACEMENTS:
                for fname, caption, fig_num in FIGURE_PLACEMENTS[section_key]:
                    fpath = os.path.join(PLOTS, fname)
                    add_figure_page(pdf, fpath, caption, fig_num)

        # ── Add remaining key figures at end ─────────────────────────────
        extra_figs = [
            ("atlas_fig04_slice_cliff.png", "2D slice through weight space showing cliffiness. Sharp boundaries separate smooth basins of attraction.", "Figure 14. "),
            ("cel_fig02_weight_pca.png", "Weight-space PCA of 132 celebrity names colored by domain. All 12 domains collapse onto the same 4 points; domain boundaries dissolve.", "Figure 15. "),
            ("cel_fig05_supercategory_scatter.png", "Super-category behavioral scatter: politics, pop culture, and intellectual figures overlap in weight-behavior space.", "Figure 16. "),
            ("cel_fig07_cross_condition_pca.png", "Cross-condition behavioral PCA: celebrities (diamonds) vs. all other LLM conditions. Celebrity points cluster within the LLM submanifold.", "Figure 17. "),
            ("cs_fig08_endtoend.png", "End-to-end validation: within/cross distances, triptych triangle, Mantel scatter.", "Figure 18. "),
            ("hf_fig01_gram_matrix.png", "L² Gram matrix of 121 zoo gait trajectories. Block structure reveals behavioral clusters.", "Figure 19. "),
        ]

        print("\n  Adding supplementary figures...")
        for fname, caption, fig_num in extra_figs:
            fpath = os.path.join(PLOTS, fname)
            add_figure_page(pdf, fpath, caption, fig_num)

    print(f"\n  WROTE {OUT_PDF}")
    file_size = os.path.getsize(OUT_PDF) / (1024 * 1024)
    print(f"  Size: {file_size:.1f} MB")


if __name__ == "__main__":
    main()
