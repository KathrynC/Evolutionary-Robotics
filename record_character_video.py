#!/usr/bin/env python3
"""
record_character_video.py

Record a video for one or more characters from the character seed experiment,
in the same format as the Motion Gait Dictionary concept videos:
  1. Title card (3s) — character name, story, entry count
  2. Clips (one per LLM model) — simulation with lower-third caption
  3. Credits card (4s) — roster of all entries with weights

Usage:
    python record_character_video.py "Juliet Capulet"
    python record_character_video.py "Walter White" "Jesse Pinkman"
    python record_character_video.py --all-romeo   # all Romeo and Juliet characters
    python record_character_video.py --story "Breaking Bad"
"""

import subprocess
import sys
import os
import json
import shutil
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
from pyrosim.neuralNetwork import NEURAL_NETWORK
import pyrosim.pyrosim as pyrosim

# ── Video settings (same as record_concept_videos.py) ────────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 30
FRAME_EVERY_N = 4
CLIP_STEPS = 4000
TITLE_SECONDS = 3
CREDITS_SECONDS = 4

CAMERA_DISTANCE = 3.0
CAMERA_YAW = 60.0
CAMERA_PITCH = -25.0
CAMERA_TARGET_Z_OFFSET = 0.5

# ── Fonts ────────────────────────────────────────────────────────────────────

FONT_HELVETICA = "/System/Library/Fonts/Helvetica.ttc"
FONT_MENLO = "/System/Library/Fonts/Menlo.ttc"

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

FONT_TITLE_LARGE = _load_font(FONT_HELVETICA, 64)
FONT_TITLE_MED = _load_font(FONT_HELVETICA, 32)
FONT_TITLE_SMALL = _load_font(FONT_HELVETICA, 24)
FONT_CAPTION = _load_font(FONT_MENLO, 28)
FONT_CREDITS_HEAD = _load_font(FONT_HELVETICA, 48)
FONT_CREDITS_ROW = _load_font(FONT_MENLO, 22)
FONT_CREDITS_FOOT = _load_font(FONT_MENLO, 16)
FONT_COUNTER = _load_font(FONT_MENLO, 22)

# ── Colors ───────────────────────────────────────────────────────────────────

BG_DARK = (24, 24, 32)
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (160, 160, 170)
TEXT_ACCENT = (100, 200, 255)
TEXT_STORY = (200, 160, 255)
CAPTION_BG = (0, 0, 0, 160)

# ── Card renderers ───────────────────────────────────────────────────────────

def render_title_card(character, story, n_entries):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    # Character name
    bbox = draw.textbbox((0, 0), character, font=FONT_TITLE_LARGE)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 - 120), character,
              fill=TEXT_WHITE, font=FONT_TITLE_LARGE)

    # Story name
    bbox = draw.textbbox((0, 0), story, font=FONT_TITLE_MED)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 - 30), story,
              fill=TEXT_STORY, font=FONT_TITLE_MED)

    # Entry count
    sub = f"{n_entries} LLM interpretations"
    bbox = draw.textbbox((0, 0), sub, font=FONT_TITLE_SMALL)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 + 30), sub,
              fill=TEXT_ACCENT, font=FONT_TITLE_SMALL)

    # Footer
    foot = "Character Gait Experiment  \u00b7  Archetypometrics \u00d7 Synapse Gait Zoo"
    bbox = draw.textbbox((0, 0), foot, font=FONT_TITLE_SMALL)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT - 60), foot,
              fill=TEXT_DIM, font=FONT_TITLE_SMALL)

    return np.array(img).tobytes()


def render_credits_card(character, story, entries):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    # Title
    title = f"{character}  \u00b7  {story}"
    bbox = draw.textbbox((0, 0), title, font=FONT_CREDITS_HEAD)
    tw = bbox[2] - bbox[0]
    x = max(40, (WIDTH - tw) // 2)
    draw.text((x, 30), title, fill=TEXT_WHITE, font=FONT_CREDITS_HEAD)

    # Column headers
    y = 100
    header = f"{'#':>3s}  {'Model':<20s}  {'Weights':>42s}  {'DX':>7s}  {'DY':>7s}  {'Speed':>6s}"
    draw.text((40, y), header, fill=TEXT_ACCENT, font=FONT_CREDITS_ROW)
    y += 32

    draw.line([(40, y), (WIDTH - 40, y)], fill=TEXT_DIM, width=1)
    y += 8

    for i, entry in enumerate(entries):
        w = entry["weights"]
        wstr = f"{w['w03']:+.1f} {w['w04']:+.1f} {w['w13']:+.1f} {w['w14']:+.1f} {w['w23']:+.1f} {w['w24']:+.1f}"
        o = entry["analytics"]["outcome"]
        row = (f"{i+1:3d}  {entry['model']:<20s}  {wstr:>42s}  "
               f"{o['dx']:+7.2f}  {o['dy']:+7.2f}  {o['mean_speed']:6.2f}")
        draw.text((40, y), row, fill=TEXT_WHITE, font=FONT_CREDITS_ROW)
        y += 28

    # Footer
    foot = "Character Gait Experiment  \u00b7  3-link PyBullet robot  \u00b7  4000 steps @ 240 Hz"
    bbox = draw.textbbox((0, 0), foot, font=FONT_CREDITS_FOOT)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT - 40), foot,
              fill=TEXT_DIM, font=FONT_CREDITS_FOOT)

    return np.array(img).tobytes()


def burn_caption(frame_bytes, character, model, dx, dy, clip_num, total_clips):
    img = Image.frombytes("RGB", (WIDTH, HEIGHT), frame_bytes)
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bar_h = 52
    bar_y = HEIGHT - bar_h
    draw.rectangle([(0, bar_y), (WIDTH, HEIGHT)], fill=CAPTION_BG)

    caption = f'{character}  \u00b7  {model}  \u00b7  DX={dx:+.1f} DY={dy:+.1f}'
    draw.text((20, bar_y + 10), caption, fill=TEXT_WHITE, font=FONT_CAPTION)

    counter = f"{clip_num}/{total_clips}"
    bbox = draw.textbbox((0, 0), counter, font=FONT_COUNTER)
    tw = bbox[2] - bbox[0]
    draw.text((WIDTH - tw - 20, bar_y + 14), counter,
              fill=TEXT_DIM, font=FONT_COUNTER)

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    return np.array(img).tobytes()


# ── Simulation ───────────────────────────────────────────────────────────────

def write_brain(weights):
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for src, tgt in [("0","3"),("1","3"),("2","3"),("0","4"),("1","4"),("2","4")]:
            key = f"w{src}{tgt}"
            w = weights.get(key, 0)
            f.write(f'    <synapse sourceNeuronName = "{src}" targetNeuronName = "{tgt}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def safe_get_base_pose(body_id):
    try:
        return p.getBasePositionAndOrientation(body_id)
    except Exception:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)


def render_sim_frame(robot_pos):
    target = [robot_pos[0], robot_pos[1], robot_pos[2] + CAMERA_TARGET_Z_OFFSET]
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=CAMERA_DISTANCE,
        yaw=CAMERA_YAW,
        pitch=CAMERA_PITCH,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=WIDTH / HEIGHT, nearVal=0.1, farVal=100)
    _, _, rgba, _, _ = p.getCameraImage(
        WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    rgb = np.array(rgba, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]
    return rgb.tobytes()


def simulate_clip(weights, character, model, dx, dy, clip_num, total_clips, ffmpeg):
    write_brain(weights)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(c.DT)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
    for link in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, link, lateralFriction=mu, restitution=0.0)

    nn = NEURAL_NETWORK("brain.nndf")

    for i in range(CLIP_STEPS):
        for neuronName in nn.neurons:
            n = nn.neurons[neuronName]
            if n.Is_Motor_Neuron():
                jn = n.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(
                        robotId, jn_bytes, n.Get_Value(), c.MAX_FORCE)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(
                        robotId, jn_bytes, p.POSITION_CONTROL,
                        n.Get_Value(), c.MAX_FORCE)

        p.stepSimulation()
        nn.Update()

        if i % FRAME_EVERY_N == 0:
            pos, _ = safe_get_base_pose(robotId)
            frame = render_sim_frame(pos)
            frame = burn_caption(frame, character, model, dx, dy,
                                 clip_num, total_clips)
            try:
                ffmpeg.stdin.write(frame)
            except BrokenPipeError:
                break

    p.disconnect()


# ── Main pipeline ────────────────────────────────────────────────────────────

def load_character_data():
    """Load character results from checkpoint or final output."""
    final = PROJECT / "artifacts" / "character_seed_experiment.json"
    checkpoint = PROJECT / "artifacts" / "character_seed_experiment_checkpoint.json"

    path = final if final.exists() else checkpoint
    if not path.exists():
        print("No character experiment data found. Run character_seed_experiment.py first.")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)
    return data["results"]


def record_character(character_name, results, out_dir):
    """Record a video for one character across all LLM models."""
    entries = [r for r in results
               if r["character"] == character_name and r["success"] and r.get("analytics")]

    if not entries:
        print(f"No successful results for '{character_name}'")
        return None

    story = entries[0]["story"]
    n = len(entries)

    # Sanitize filename
    safe_name = character_name.replace(" ", "_").replace("/", "-")
    video_path = out_dir / f"{safe_name}.mp4"

    print(f"\n{'='*60}")
    print(f"  {character_name} ({story}): {n} clips + title + credits")
    print(f"{'='*60}")

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo",
         "-pixel_format", "rgb24",
         "-video_size", f"{WIDTH}x{HEIGHT}",
         "-framerate", str(FPS),
         "-i", "pipe:0",
         "-c:v", "libx264",
         "-pix_fmt", "yuv420p",
         "-preset", "fast",
         "-crf", "23",
         str(video_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 1. Title card
    print(f"  Title card ({TITLE_SECONDS}s)...")
    title_frame = render_title_card(character_name, story, n)
    for _ in range(TITLE_SECONDS * FPS):
        ffmpeg.stdin.write(title_frame)

    # 2. Clips
    for i, entry in enumerate(entries):
        model = entry["model"]
        weights = entry["weights"]
        o = entry["analytics"]["outcome"]
        dx, dy = o["dx"], o["dy"]
        print(f"  [{i+1}/{n}] {model}  DX={dx:+.2f} DY={dy:+.2f}")
        simulate_clip(weights, character_name, model, dx, dy, i + 1, n, ffmpeg)

    # 3. Credits card
    print(f"  Credits card ({CREDITS_SECONDS}s)...")
    credits_frame = render_credits_card(character_name, story, entries)
    for _ in range(CREDITS_SECONDS * FPS):
        ffmpeg.stdin.write(credits_frame)

    ffmpeg.stdin.close()
    ffmpeg.wait(timeout=60)

    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {video_path} ({size_mb:.1f} MB)")
    return video_path


def main():
    results = load_character_data()

    # Parse args
    characters_to_record = []
    if "--story" in sys.argv:
        idx = sys.argv.index("--story")
        story_name = sys.argv[idx + 1]
        all_chars = set()
        for r in results:
            if r.get("story") == story_name and r["success"]:
                all_chars.add(r["character"])
        characters_to_record = sorted(all_chars)
        print(f"Recording all characters from '{story_name}': {len(characters_to_record)} characters")
    else:
        characters_to_record = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not characters_to_record:
        # List available characters
        chars = set()
        for r in results:
            if r["success"] and r.get("analytics"):
                chars.add(f"{r['character']} ({r['story']})")
        print(f"Available characters: {len(chars)}")
        print("Usage: python record_character_video.py \"Character Name\"")
        print("       python record_character_video.py --story \"Story Name\"")
        sys.exit(0)

    out_dir = PROJECT / "videos" / "characters"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Backup brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    try:
        for char_name in characters_to_record:
            record_character(char_name, results, out_dir)
    finally:
        if backup_path.exists():
            shutil.copy2(backup_path, brain_path)

    print(f"\nDone! {len(characters_to_record)} character videos in {out_dir}/")


if __name__ == "__main__":
    main()
