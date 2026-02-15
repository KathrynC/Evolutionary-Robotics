#!/usr/bin/env python3
"""
render_report_pdf.py

Renders motion_seed_experiment_v2_report.md as a nicely formatted PDF
using matplotlib's PdfPages backend. Handles headers, body text, bold,
bullet points, markdown tables, and code spans.
"""

import re
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm

# Register CJK font
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
REPORT_PATH = PROJECT / "artifacts" / "motion_seed_experiment_v2_report.md"
OUT_PATH = PROJECT / "artifacts" / "motion_seed_experiment_v2_report.pdf"

# Layout — portrait letter
PAGE_W, PAGE_H = 8.5, 11
LEFT = 0.08
RIGHT = 0.92
TOP = 0.93
BOTTOM = 0.05
LINE_H = 0.016
BODY_SIZE = 8
SMALL_SIZE = 6.5


class PDFRenderer:
    def __init__(self, pdf):
        self.pdf = pdf
        self.fig = None
        self.ax = None
        self.y = TOP
        self._new_page()

    def _new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.y = TOP

    def _need(self, h):
        """Ensure h vertical space available, start new page if not."""
        if self.y - h < BOTTOM:
            self._new_page()

    def _text(self, x, y, text, **kwargs):
        self.ax.text(x, y, text, va="top", **kwargs)

    def blank(self, n=1):
        self.y -= LINE_H * n

    def h1(self, text):
        self._need(LINE_H * 3)
        self.y -= LINE_H * 0.5
        self._text(LEFT, self.y, text, fontsize=16, fontweight="bold",
                   family="serif", color="#111")
        self.y -= LINE_H * 2.2

    def h2(self, text):
        self._need(LINE_H * 3)
        self.y -= LINE_H * 0.8
        self.ax.plot([LEFT, RIGHT], [self.y + LINE_H * 0.3, self.y + LINE_H * 0.3],
                     color="#ccc", linewidth=0.5)
        self.y -= LINE_H * 0.2
        self._text(LEFT, self.y, text, fontsize=13, fontweight="bold",
                   family="serif", color="#222")
        self.y -= LINE_H * 1.8

    def h3(self, text):
        self._need(LINE_H * 2.5)
        self.y -= LINE_H * 0.5
        self._text(LEFT, self.y, text, fontsize=10, fontweight="bold",
                   family="serif", color="#333")
        self.y -= LINE_H * 1.5

    def para(self, text, indent=0):
        """Render a paragraph with word wrapping. Handles **bold** and `code`."""
        x_start = LEFT + indent
        max_width = RIGHT - x_start
        # Approximate chars per line at BODY_SIZE pt on PAGE_W
        # monospace char ~ 0.006 width units at 8pt on 8.5" page
        char_w = BODY_SIZE / 72 / PAGE_W
        chars_per_line = int(max_width / char_w)

        lines = self._wrap(text, chars_per_line)
        for line in lines:
            self._need(LINE_H * 1.2)
            self._render_rich_line(x_start, self.y, line, BODY_SIZE)
            self.y -= LINE_H

    def bullet(self, text):
        """Render a bullet point with wrapping."""
        self._need(LINE_H * 1.2)
        self._text(LEFT + 0.02, self.y, "\u2022", fontsize=BODY_SIZE,
                   family="serif", color="#333")

        x_start = LEFT + 0.04
        max_width = RIGHT - x_start
        char_w = BODY_SIZE / 72 / PAGE_W
        chars_per_line = int(max_width / char_w)

        lines = self._wrap(text, chars_per_line)
        for i, line in enumerate(lines):
            self._need(LINE_H * 1.2)
            self._render_rich_line(x_start, self.y, line, BODY_SIZE)
            self.y -= LINE_H

    def table(self, rows):
        """Render a markdown table. rows = list of lists of cell strings."""
        if not rows:
            return
        n_cols = len(rows[0])

        # Compute column widths based on max content
        col_maxlen = [0] * n_cols
        for row in rows:
            for j, cell in enumerate(row):
                if j < n_cols:
                    col_maxlen[j] = max(col_maxlen[j], len(cell))

        # Scale to fit page
        char_w = SMALL_SIZE / 72 / PAGE_W
        total_chars = sum(col_maxlen) + n_cols * 2  # 2 chars padding per col
        total_width = total_chars * char_w
        usable = RIGHT - LEFT

        if total_width > usable:
            # Scale down
            scale = usable / total_width
        else:
            scale = 1.0

        col_w = [(ml + 2) * char_w * scale for ml in col_maxlen]

        # Draw table
        row_h = LINE_H * 1.1
        self._need(row_h * min(len(rows), 4))

        for i, row in enumerate(rows):
            self._need(row_h)
            x = LEFT
            is_header = (i == 0)

            # Background for header
            if is_header:
                self.ax.fill_between([LEFT, LEFT + sum(col_w)],
                                     self.y - row_h * 0.3, self.y + row_h * 0.6,
                                     color="#e8e8e8", linewidth=0)
            elif i % 2 == 0:
                self.ax.fill_between([LEFT, LEFT + sum(col_w)],
                                     self.y - row_h * 0.3, self.y + row_h * 0.6,
                                     color="#f7f7f7", linewidth=0)

            for j, cell in enumerate(row):
                if j >= n_cols:
                    break
                weight = "bold" if is_header else "normal"
                # Strip markdown bold markers for display
                display = cell.replace("**", "")
                is_bold_cell = cell.startswith("**") and cell.endswith("**")
                if is_bold_cell:
                    weight = "bold"
                self._text(x + char_w * 0.5, self.y, display,
                           fontsize=SMALL_SIZE, family="monospace",
                           fontweight=weight,
                           color="#111" if (is_header or is_bold_cell) else "#333")
                x += col_w[j]

            # Line under header
            if is_header:
                self.ax.plot([LEFT, LEFT + sum(col_w)],
                             [self.y - row_h * 0.35, self.y - row_h * 0.35],
                             color="#999", linewidth=0.5)

            self.y -= row_h

        self.y -= LINE_H * 0.3

    def _wrap(self, text, width):
        """Simple word wrap."""
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip() if current else word
            # Use plain length (ignoring markdown) for wrapping
            plain = re.sub(r'\*\*|`', '', test)
            if len(plain) <= width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    def _render_rich_line(self, x, y, text, size):
        """Render a line with **bold** and `code` spans, with CJK support."""
        parts = re.split(r'(\*\*.*?\*\*|`[^`]+`)', text)
        char_w = size / 72 / PAGE_W
        for part in parts:
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                display = part[2:-2]
                x = self._render_cjk_aware(x, y, display, size,
                                           fontweight="bold", color="#111")
            elif part.startswith("`") and part.endswith("`"):
                display = part[1:-1]
                self._text(x, y, display, fontsize=size, family="monospace",
                           color="#8B0000",
                           bbox=dict(boxstyle="round,pad=0.15",
                                     facecolor="#f0f0f0", edgecolor="none"))
                x += (len(display) + 1) * char_w
            else:
                x = self._render_cjk_aware(x, y, part, size, color="#333")

    def _render_cjk_aware(self, x, y, text, size, fontweight="normal",
                          color="#333"):
        """Render text, switching to CJK font for CJK characters."""
        char_w = size / 72 / PAGE_W
        # Split into CJK and non-CJK runs
        segments = re.split(r'([\u2E80-\u9FFF\uF900-\uFAFF]+)', text)
        for seg in segments:
            if not seg:
                continue
            if _CJK_FONT and any(ord(c) > 0x2E80 for c in seg):
                prop = fm.FontProperties(fname=_CJK_FONT, size=size)
                self._text(x, y, seg, fontproperties=prop,
                           fontweight=fontweight, color=color)
            else:
                self._text(x, y, seg, fontsize=size, family="monospace",
                           fontweight=fontweight, color=color)
            x += len(seg) * char_w
        return x

    def finish(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)


def parse_markdown(text):
    """Parse markdown into a sequence of elements."""
    lines = text.split("\n")
    elements = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Blank line
        if not line.strip():
            elements.append(("blank",))
            i += 1
            continue

        # Headers
        if line.startswith("### "):
            elements.append(("h3", line[4:].strip()))
            i += 1
            continue
        if line.startswith("## "):
            elements.append(("h2", line[3:].strip()))
            i += 1
            continue
        if line.startswith("# "):
            elements.append(("h1", line[2:].strip()))
            i += 1
            continue

        # Table: collect all | lines, skip separator rows
        if "|" in line and line.strip().startswith("|"):
            table_rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                row_text = lines[i].strip()
                # Skip separator rows (|---|---|)
                if re.match(r'^\|[\s\-:|]+\|$', row_text):
                    i += 1
                    continue
                cells = [c.strip() for c in row_text.split("|")[1:-1]]
                table_rows.append(cells)
                i += 1
            elements.append(("table", table_rows))
            continue

        # Bullet
        if line.strip().startswith("- "):
            text = line.strip()[2:]
            elements.append(("bullet", text))
            i += 1
            continue

        # Regular paragraph — collect consecutive non-empty, non-special lines
        para_lines = []
        while i < len(lines):
            l = lines[i]
            if not l.strip():
                break
            if l.startswith("#") or l.startswith("| ") or l.strip().startswith("- "):
                break
            para_lines.append(l)
            i += 1
        elements.append(("para", " ".join(para_lines)))
        continue

    return elements


def main():
    print(f"Loading report from {REPORT_PATH.name}...")
    md_text = REPORT_PATH.read_text()

    elements = parse_markdown(md_text)
    print(f"  {len(elements)} elements parsed")

    print(f"Rendering PDF...")
    with PdfPages(str(OUT_PATH)) as pdf:
        r = PDFRenderer(pdf)

        for elem in elements:
            kind = elem[0]
            if kind == "blank":
                r.blank(0.5)
            elif kind == "h1":
                r.h1(elem[1])
            elif kind == "h2":
                r.h2(elem[1])
            elif kind == "h3":
                r.h3(elem[1])
            elif kind == "para":
                r.para(elem[1])
            elif kind == "bullet":
                r.bullet(elem[1])
            elif kind == "table":
                r.table(elem[1])

        r.finish()

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Saved to {OUT_PATH}")
    print(f"  {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
