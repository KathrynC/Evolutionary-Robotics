#!/usr/bin/env python3
"""
compile_story_video.py

Compile all character videos for a given story into a single video with:
  1. Opening title card (5s) — story name, character count, LLM count
  2. Per-character section divider (2s) + character video
  3. Closing credits (8s) — all characters, all models, project info

Usage:
    python compile_story_video.py "Romeo and Juliet"
    python compile_story_video.py "Breaking Bad"
"""

import subprocess
import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import date
from PIL import Image, ImageDraw, ImageFont
import numpy as np

PROJECT = Path(__file__).resolve().parent

# ── Video settings ───────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 30
OPENING_SECONDS = 5
SECTION_SECONDS = 2
CLOSING_SECONDS = 8

# ── Fonts ────────────────────────────────────────────────────────────────────

FONT_HELVETICA = "/System/Library/Fonts/Helvetica.ttc"
FONT_MENLO = "/System/Library/Fonts/Menlo.ttc"

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

FONT_HUGE = _load_font(FONT_HELVETICA, 56)
FONT_LARGE = _load_font(FONT_HELVETICA, 36)
FONT_MED = _load_font(FONT_HELVETICA, 28)
FONT_SMALL = _load_font(FONT_HELVETICA, 22)
FONT_TINY = _load_font(FONT_MENLO, 16)
FONT_SECTION = _load_font(FONT_HELVETICA, 64)
FONT_SECTION_SUB = _load_font(FONT_HELVETICA, 24)
FONT_CREDITS_HEAD = _load_font(FONT_HELVETICA, 28)
FONT_CREDITS_BODY = _load_font(FONT_HELVETICA, 20)
FONT_CREDITS_ROW = _load_font(FONT_MENLO, 18)

# ── Colors ───────────────────────────────────────────────────────────────────

BG_DARK = (24, 24, 32)
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (140, 140, 150)
TEXT_ACCENT = (100, 200, 255)
TEXT_GOLD = (255, 210, 100)
TEXT_STORY = (200, 160, 255)
RULE_COLOR = (80, 80, 90)


def centered(draw, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, y), text, fill=fill, font=font)


def card_to_mp4(img, duration_s, tmpdir, name):
    path = os.path.join(tmpdir, f"{name}.mp4")
    raw = np.array(img).tobytes()
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{WIDTH}x{HEIGHT}",
         "-framerate", str(FPS),
         "-i", "pipe:0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "fast", "-crf", "23",
         path],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(duration_s * FPS):
        ffmpeg.stdin.write(raw)
    ffmpeg.stdin.close()
    ffmpeg.wait(timeout=30)
    return path


# ── Card renderers ───────────────────────────────────────────────────────────

def render_opening_card(story_name, characters, n_models):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    centered(draw, 120, "Character Gait Experiment", FONT_LARGE, TEXT_DIM)
    centered(draw, 180, story_name, FONT_HUGE, TEXT_WHITE)

    draw.line([(340, 260), (940, 260)], fill=RULE_COLOR, width=1)

    centered(draw, 290, "Kathryn Cramer", FONT_LARGE, TEXT_GOLD)
    centered(draw, 340, "University of Vermont", FONT_MED, TEXT_DIM)

    centered(draw, 410, date.today().strftime("%B %d, %Y"), FONT_SMALL, TEXT_DIM)

    stats = f"{len(characters)} characters  \u00b7  {n_models} LLMs  \u00b7  Archetypometrics \u00d7 Synapse Gait Zoo"
    centered(draw, 480, stats, FONT_SMALL, TEXT_ACCENT)

    # Character list
    char_list = "  \u00b7  ".join(characters)
    # Wrap if too long
    if len(char_list) > 80:
        mid = len(characters) // 2
        line1 = "  \u00b7  ".join(characters[:mid])
        line2 = "  \u00b7  ".join(characters[mid:])
        centered(draw, 540, line1, FONT_SMALL, TEXT_STORY)
        centered(draw, 570, line2, FONT_SMALL, TEXT_STORY)
    else:
        centered(draw, 540, char_list, FONT_SMALL, TEXT_STORY)

    centered(draw, HEIGHT - 50, "3-link PyBullet robot  \u00b7  4000 steps @ 240 Hz",
             FONT_TINY, TEXT_DIM)

    return img


def render_section_card(character_name, story_name):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    centered(draw, HEIGHT // 2 - 60, character_name, FONT_SECTION, TEXT_WHITE)
    centered(draw, HEIGHT // 2 + 30, story_name, FONT_SECTION_SUB, TEXT_STORY)

    return img


def render_closing_credits(story_name, characters, char_data):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    y = 30
    centered(draw, y, story_name, FONT_LARGE, TEXT_WHITE)
    y += 50

    centered(draw, y, "Character Summary", FONT_CREDITS_HEAD, TEXT_ACCENT)
    y += 40

    # Per-character summary row
    header = f"{'Character':<22s}  {'Models':>6s}  {'Best |DX|':>9s}  {'Best Model':<20s}"
    centered(draw, y, header, FONT_CREDITS_ROW, TEXT_ACCENT)
    y += 26

    for char_name in characters:
        entries = [r for r in char_data if r["character"] == char_name
                   and r["success"] and r.get("analytics")]
        if not entries:
            continue
        best = max(entries, key=lambda r: abs(r["analytics"]["outcome"]["dx"]))
        best_dx = abs(best["analytics"]["outcome"]["dx"])
        row = f"{char_name:<22s}  {len(entries):>6d}  {best_dx:>9.2f}  {best['model']:<20s}"
        centered(draw, y, row, FONT_CREDITS_ROW, TEXT_WHITE)
        y += 24

    y += 20

    # LLMs
    centered(draw, y, "Language Models", FONT_CREDITS_HEAD, TEXT_ACCENT)
    y += 35
    llms = ["qwen3-coder:30b", "deepseek-r1:8b", "llama3.1:latest", "gpt-oss:20b"]
    centered(draw, y, "  \u00b7  ".join(llms), FONT_CREDITS_BODY, TEXT_WHITE)
    y += 35

    # Project info
    centered(draw, y, "Archetypometrics \u00d7 Synapse Gait Zoo", FONT_CREDITS_HEAD, TEXT_ACCENT)
    y += 35
    centered(draw, y, "2000 fictional characters from 341 stories",
             FONT_CREDITS_BODY, TEXT_WHITE)

    centered(draw, HEIGHT - 50,
             f"Character Gait Experiment  \u00b7  Kathryn Cramer  \u00b7  University of Vermont  \u00b7  "
             + date.today().strftime("%Y"),
             FONT_TINY, TEXT_DIM)

    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python compile_story_video.py \"Story Name\"")
        sys.exit(1)

    story_name = sys.argv[1]
    chars_dir = PROJECT / "videos" / "characters"

    # Load character data for credits
    final = PROJECT / "artifacts" / "character_seed_experiment.json"
    checkpoint = PROJECT / "artifacts" / "character_seed_experiment_checkpoint.json"
    data_path = final if final.exists() else checkpoint
    if not data_path.exists():
        print("No character experiment data found.")
        sys.exit(1)

    with open(data_path) as f:
        all_results = json.load(f)["results"]

    # Find characters from this story
    story_chars = sorted(set(
        r["character"] for r in all_results
        if r.get("story") == story_name and r["success"]
    ))

    if not story_chars:
        print(f"No characters found for story '{story_name}'")
        available_stories = sorted(set(r.get("story", "") for r in all_results if r["success"]))
        print(f"Available stories: {', '.join(available_stories[:20])}...")
        sys.exit(1)

    # Check which character videos exist
    available = []
    missing = []
    for char_name in story_chars:
        safe_name = char_name.replace(" ", "_").replace("/", "-")
        video_path = chars_dir / f"{safe_name}.mp4"
        if video_path.exists():
            available.append((char_name, video_path))
        else:
            missing.append(char_name)

    if missing:
        print(f"Missing videos for: {', '.join(missing)}")
        print(f"Run: python record_character_video.py --story \"{story_name}\"")
        if not available:
            sys.exit(1)

    print(f"Compiling '{story_name}': {len(available)} characters")
    n_models = len(set(r["model"] for r in all_results
                       if r.get("story") == story_name and r["success"]))

    safe_story = story_name.replace(" ", "_").replace(":", "").replace("/", "-")
    out_path = PROJECT / "videos" / "stories" / f"{safe_story}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        parts = []

        # 1. Opening
        print("Rendering opening title card...")
        char_names = [c[0] for c in available]
        opening_img = render_opening_card(story_name, char_names, n_models)
        opening_path = card_to_mp4(opening_img, OPENING_SECONDS, tmpdir, "00_opening")
        parts.append(opening_path)

        # 2. Per-character: section card + video
        for i, (char_name, video_path) in enumerate(available):
            print(f"  [{i+1}/{len(available)}] {char_name}")

            section_img = render_section_card(char_name, story_name)
            section_path = card_to_mp4(section_img, SECTION_SECONDS, tmpdir,
                                       f"sec_{i:03d}")
            parts.append(section_path)
            parts.append(str(video_path))

        # 3. Closing credits
        print("Rendering closing credits...")
        story_results = [r for r in all_results if r.get("story") == story_name]
        credits_img = render_closing_credits(story_name, char_names, story_results)
        credits_path = card_to_mp4(credits_img, CLOSING_SECONDS, tmpdir, "zz_credits")
        parts.append(credits_path)

        # 4. Concat
        concat_path = os.path.join(tmpdir, "concat.txt")
        with open(concat_path, "w") as f:
            for part in parts:
                escaped = part.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        print(f"Concatenating {len(parts)} segments...")
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-f", "concat", "-safe", "0",
             "-i", concat_path,
             "-c", "copy",
             str(out_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"ffmpeg error:\n{result.stderr[-500:]}")
            sys.exit(1)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(out_path)],
            capture_output=True, text=True,
        )
        duration_s = float(probe.stdout.strip()) if probe.stdout.strip() else 0

        print(f"\nDone!")
        print(f"  Output: {out_path}")
        print(f"  Duration: {duration_s/60:.1f} min ({duration_s:.0f}s)")
        print(f"  Size: {size_mb:.0f} MB")
        print(f"  Characters: {len(available)}")


if __name__ == "__main__":
    main()
