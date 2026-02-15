#!/usr/bin/env python3
"""
compile_concept_video.py

Concatenate all per-concept videos into a single compilation with:
  1. Opening title card (5s) — project name, author, date
  2. All concept videos in alphabetical order (with section title cards)
  3. Closing credits card (8s) — LLMs used, key papers, statistics

Usage:
    python compile_concept_video.py
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

# ── Video settings (must match record_concept_videos.py) ────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 30
OPENING_SECONDS = 5
SECTION_SECONDS = 2
CLOSING_SECONDS = 10

# ── Fonts ───────────────────────────────────────────────────────────────────

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
FONT_CREDITS_BODY = _load_font(FONT_HELVETICA, 20)
FONT_CREDITS_CITE = _load_font(FONT_HELVETICA, 17)
FONT_CREDITS_HEAD = _load_font(FONT_HELVETICA, 28)
FONT_SECTION = _load_font(FONT_HELVETICA, 64)
FONT_SECTION_SUB = _load_font(FONT_HELVETICA, 24)

# ── Colors ──────────────────────────────────────────────────────────────────

BG_DARK = (24, 24, 32)
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (140, 140, 150)
TEXT_ACCENT = (100, 200, 255)
TEXT_GOLD = (255, 210, 100)
RULE_COLOR = (80, 80, 90)


def centered(draw, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, y), text, fill=fill, font=font)


def card_to_mp4(img, duration_s, tmpdir, name):
    """Write a still image as a short MP4 clip."""
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


def render_opening_card():
    """Render the opening title card."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    centered(draw, 140, "Motion Gait Dictionary", FONT_HUGE, TEXT_WHITE)
    centered(draw, 220, "Semantic Motion Concepts for a Neural Walking Robot",
             FONT_SMALL, TEXT_DIM)

    # Rule
    draw.line([(340, 290), (940, 290)], fill=RULE_COLOR, width=1)

    centered(draw, 320, "Kathryn Cramer", FONT_LARGE, TEXT_GOLD)
    centered(draw, 370, "University of Vermont", FONT_MED, TEXT_DIM)

    centered(draw, 440, date.today().strftime("%B %d, %Y"), FONT_SMALL, TEXT_DIM)

    # Stats
    centered(draw, 520, "58 concepts  \u00b7  365 entries  \u00b7  5 LLMs  \u00b7  5 languages",
             FONT_SMALL, TEXT_ACCENT)

    # Footer
    centered(draw, HEIGHT - 50, "3-link PyBullet robot  \u00b7  4000 steps @ 240 Hz",
             FONT_TINY, TEXT_DIM)

    return img


def render_section_card(concept_id, n_entries, description=""):
    """Render a brief section divider card for each concept."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    centered(draw, HEIGHT // 2 - 60, concept_id, FONT_SECTION, TEXT_WHITE)
    if description:
        centered(draw, HEIGHT // 2 + 30, description, FONT_SECTION_SUB, TEXT_DIM)

    return img


def render_closing_credits():
    """Render the closing credits card."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    y = 40
    centered(draw, y, "Credits", FONT_LARGE, TEXT_WHITE)
    y += 60

    # LLMs
    centered(draw, y, "Language Models", FONT_CREDITS_HEAD, TEXT_ACCENT)
    y += 40
    llms = [
        ("gpt-4.1-mini", "OpenAI API"),
        ("qwen3-coder:30b", "Ollama (local)"),
        ("deepseek-r1:8b", "Ollama (local)"),
        ("llama3.1:latest", "Ollama (local)"),
        ("gpt-oss:20b", "Ollama (local)"),
    ]
    for name, source in llms:
        text = f"{name}  \u2014  {source}"
        centered(draw, y, text, FONT_CREDITS_BODY, TEXT_WHITE)
        y += 28

    # Prompt design consultants
    y += 15
    centered(draw, y, "Prompt Design Consultants", FONT_CREDITS_HEAD, TEXT_ACCENT)
    y += 40
    consultants = [
        "GPT-4.1-mini (via API)  \u00b7  DeepSeek (via web)  \u00b7  GPT-5.2 (via web)"
    ]
    for c in consultants:
        centered(draw, y, c, FONT_CREDITS_BODY, TEXT_WHITE)
        y += 28

    # Key papers
    y += 15
    centered(draw, y, "Key References", FONT_CREDITS_HEAD, TEXT_ACCENT)
    y += 38
    papers = [
        'Beer, R.D. (1996). "Toward the evolution of dynamical neural networks',
        '    for minimally cognitive behavior." Proc. Simulation of Adaptive Behavior.',
        '',
        'Sims, K. (1994). "Evolving virtual creatures."',
        '    Proc. SIGGRAPH \'94, pp. 15\u201322.',
        '',
        'Bongard, J. (2013\u20132024). Ludobots: An Introduction to',
        '    Evolutionary Robotics. University of Vermont / Reddit r/ludobots.',
    ]
    for line in papers:
        if line == '':
            y += 6
            continue
        centered(draw, y, line, FONT_CREDITS_CITE, TEXT_DIM)
        y += 22

    # Footer
    centered(draw, HEIGHT - 50,
             "Motion Gait Dictionary v2  \u00b7  Kathryn Cramer  \u00b7  University of Vermont  \u00b7  "
             + date.today().strftime("%Y"),
             FONT_TINY, TEXT_DIM)

    return img


# Concept descriptions (same as record_concept_videos.py)
CONCEPT_DESCRIPTIONS = {
    "freeze": "perfectly still, no motion",
    "drag": "slow effortful movement, lots of energy for little progress",
    "hop": "bouncing vertical motion with forward progress",
    "retreat": "backward movement, away from forward direction",
    "crawl": "slow ground-hugging locomotion",
    "stumble": "irregular, erratic movement",
    "patrol": "steady forward movement with moderate speed",
    "sway": "lateral rocking, side-to-side oscillation",
    "bounce": "repetitive vertical motion",
    "wobble": "unstable oscillating movement",
    "zigzag": "alternating lateral direction changes",
    "gallop": "fast asymmetric bounding gait",
    "stagger": "unsteady lurching with variable direction",
    "dash": "burst of fast forward motion",
    "sprint": "sustained high-speed forward locomotion",
    "stomp": "high-energy ground impacts with little travel",
    "march": "steady rhythmic forward locomotion",
    "drift": "slow passive lateral movement",
    "tiptoe": "cautious minimal-contact movement",
    "charge": "aggressive fast forward rush",
    "turn_left": "veering or rotating to the left",
    "turn_right": "veering or rotating to the right",
    "scurry": "fast small-step scrambling",
    "plod": "slow heavy trudging forward",
    "rock": "back-and-forth rocking in place",
    "slide": "smooth lateral gliding motion",
    "headstand": "inverted, torso on ground, legs in the air",
    "pivot": "turning in place with minimal translation",
    "twirl": "fast spinning rotation",
    "circle": "curved path, moving in a loop",
    "stand_still": "perfectly still, no motion",
    "lurch": "sudden irregular forward motion",
    "limp": "asymmetric gait favoring one side",
    "shuffle": "short sliding steps with little lift",
    "walk_and_spin": "forward motion with continuous rotation",
    "crab_walk": "lateral sideways locomotion",
    "backward_walk": "moving in reverse",
    "forward_walk": "basic forward locomotion",
    "spin": "rotation in place",
    "glide": "smooth effortless forward motion",
    "amble": "relaxed easy walking pace",
    "creep": "very slow cautious movement",
    "fall": "losing balance, toppling over",
    "lunge": "sudden forward thrust",
    "prance": "high-stepping showy gait",
    "prowl": "slow deliberate stalking movement",
    "roam": "wandering without fixed direction",
    "roll": "rotating along the ground",
    "rush": "hurried fast movement",
    "saunter": "casual unhurried stroll",
    "skid": "sliding loss of traction",
    "skip": "light bounding alternating steps",
    "stride": "long confident steps",
    "trot": "moderate-speed two-beat gait",
    "twist": "rotational contortion",
    "waddle": "side-to-side rocking walk",
    "wander": "aimless meandering",
    "weave": "sinuous side-to-side path",
}


def main():
    concepts_dir = PROJECT / "videos" / "concepts"
    dict_path = PROJECT / "artifacts" / "motion_gait_dictionary_v2.json"

    # Load dictionary for entry counts
    with open(dict_path) as f:
        data = json.load(f)
    concepts = data["concepts"]

    # Find all concept videos
    available = sorted([
        f.stem for f in concepts_dir.glob("*.mp4")
    ])
    print(f"Found {len(available)} concept videos")

    if not available:
        print("No videos found! Run record_concept_videos.py first.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        parts = []

        # 1. Opening title card
        print("Rendering opening title card...")
        opening_img = render_opening_card()
        opening_path = card_to_mp4(opening_img, OPENING_SECONDS, tmpdir, "00_opening")
        parts.append(opening_path)

        # 2. Each concept: section card + concept video
        for i, concept_id in enumerate(available):
            n_entries = concepts.get(concept_id, {}).get("n_matches", "?")
            desc = CONCEPT_DESCRIPTIONS.get(concept_id, "")
            print(f"  [{i+1}/{len(available)}] {concept_id} ({n_entries} entries)")

            # Section divider card
            section_img = render_section_card(concept_id, n_entries, desc)
            section_path = card_to_mp4(section_img, SECTION_SECONDS, tmpdir,
                                       f"sec_{i:03d}_{concept_id}")
            parts.append(section_path)

            # The concept video itself
            concept_video = concepts_dir / f"{concept_id}.mp4"
            parts.append(str(concept_video))

        # 3. Closing credits
        print("Rendering closing credits...")
        credits_img = render_closing_credits()
        credits_path = card_to_mp4(credits_img, CLOSING_SECONDS, tmpdir, "zz_credits")
        parts.append(credits_path)

        # 4. Write concat file for ffmpeg
        concat_path = os.path.join(tmpdir, "concat.txt")
        with open(concat_path, "w") as f:
            for part in parts:
                # ffmpeg concat demuxer needs escaped paths
                escaped = part.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        # 5. Concatenate with ffmpeg
        out_path = PROJECT / "videos" / "motion_gait_dictionary_compilation.mp4"
        print(f"\nConcatenating {len(parts)} segments...")
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-f", "concat", "-safe", "0",
             "-i", concat_path,
             "-c", "copy",
             str(out_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"ffmpeg concat error:\n{result.stderr[-500:]}")
            sys.exit(1)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(out_path)],
            capture_output=True, text=True,
        )
        duration_s = float(probe.stdout.strip()) if probe.stdout.strip() else 0
        duration_min = duration_s / 60

        print(f"\nDone!")
        print(f"  Output: {out_path}")
        print(f"  Duration: {duration_min:.1f} min ({duration_s:.0f}s)")
        print(f"  Size: {size_mb:.0f} MB")
        print(f"  Concepts: {len(available)}")


if __name__ == "__main__":
    main()
