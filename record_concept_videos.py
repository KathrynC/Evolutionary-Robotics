#!/usr/bin/env python3
"""
record_concept_videos.py

Record one video per motion concept from the motion gait dictionary.
Each video contains:
  1. Title card (3s) — concept name + description
  2. Clips (5s each) — one per dictionary entry, with lower-third caption
  3. Credits card (4s) — roster of all entries shown

Usage:
    python record_concept_videos.py                  # all concepts
    python record_concept_videos.py drag hop freeze  # specific concepts
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

# ── Video settings ──────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 30
FRAME_EVERY_N = 4           # sim steps per video frame
CLIP_STEPS = 4000            # sim steps per clip (= c.SIM_STEPS)
TITLE_SECONDS = 3
CREDITS_SECONDS = 4
CLIP_SECONDS = CLIP_STEPS // FRAME_EVERY_N / FPS  # ~8.3s at defaults

CAMERA_DISTANCE = 3.0
CAMERA_YAW = 60.0
CAMERA_PITCH = -25.0
CAMERA_TARGET_Z_OFFSET = 0.5

# ── Fonts ───────────────────────────────────────────────────────────────────

FONT_HELVETICA = "/System/Library/Fonts/Helvetica.ttc"
FONT_MENLO = "/System/Library/Fonts/Menlo.ttc"
FONT_CJK = "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc"

# Preload fonts at various sizes
def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

FONT_TITLE_LARGE = _load_font(FONT_HELVETICA, 72)
FONT_TITLE_MED = _load_font(FONT_HELVETICA, 32)
FONT_TITLE_SMALL = _load_font(FONT_HELVETICA, 24)
FONT_CAPTION = _load_font(FONT_MENLO, 28)
FONT_CAPTION_CJK = _load_font(FONT_CJK, 28)
FONT_CREDITS_HEAD = _load_font(FONT_HELVETICA, 48)
FONT_CREDITS_ROW = _load_font(FONT_MENLO, 22)
FONT_CREDITS_ROW_CJK = _load_font(FONT_CJK, 22)
FONT_CREDITS_FOOT = _load_font(FONT_MENLO, 16)
FONT_COUNTER = _load_font(FONT_MENLO, 22)

# ── Colors ──────────────────────────────────────────────────────────────────

BG_DARK = (24, 24, 32)
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (160, 160, 170)
TEXT_ACCENT = (100, 200, 255)
CAPTION_BG = (0, 0, 0, 160)  # semi-transparent


def has_cjk(text):
    return any(ord(ch) > 0x2E80 for ch in text)


def pick_font(text, base_font, cjk_font):
    return cjk_font if has_cjk(text) else base_font


# ── Card renderers ──────────────────────────────────────────────────────────

def render_title_card(concept_id, n_entries, description=""):
    """Render a title card as raw RGB bytes."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    # Concept name — large, centered
    bbox = draw.textbbox((0, 0), concept_id, font=FONT_TITLE_LARGE)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 - 100), concept_id,
              fill=TEXT_WHITE, font=FONT_TITLE_LARGE)

    # Entry count
    sub = f"{n_entries} entries"
    bbox = draw.textbbox((0, 0), sub, font=FONT_TITLE_MED)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 + 10), sub,
              fill=TEXT_ACCENT, font=FONT_TITLE_MED)

    # Description
    if description:
        bbox = draw.textbbox((0, 0), description, font=FONT_TITLE_SMALL)
        tw = bbox[2] - bbox[0]
        x = max(40, (WIDTH - tw) // 2)
        draw.text((x, HEIGHT // 2 + 65), description,
                  fill=TEXT_DIM, font=FONT_TITLE_SMALL)

    # Bottom line
    foot = "Motion Gait Dictionary v2"
    bbox = draw.textbbox((0, 0), foot, font=FONT_TITLE_SMALL)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT - 60), foot,
              fill=TEXT_DIM, font=FONT_TITLE_SMALL)

    return np.array(img).tobytes()


def render_credits_card(concept_id, synonyms):
    """Render a credits card as raw RGB bytes."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    # Title
    bbox = draw.textbbox((0, 0), concept_id, font=FONT_CREDITS_HEAD)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, 30), concept_id,
              fill=TEXT_WHITE, font=FONT_CREDITS_HEAD)

    # Column headers
    y = 100
    header = f"{'#':>3s}  {'Word':<16s}  {'Lang':<5s}  {'Model':<18s}  {'Weights':>35s}"
    draw.text((60, y), header, fill=TEXT_ACCENT, font=FONT_CREDITS_ROW)
    y += 32

    # Draw a thin line
    draw.line([(60, y), (WIDTH - 60, y)], fill=TEXT_DIM, width=1)
    y += 8

    # Rows — fit as many as possible
    max_rows = (HEIGHT - y - 60) // 28
    for i, syn in enumerate(synonyms[:max_rows]):
        w = syn["weights"]
        wstr = f"{w['w03']:+.1f} {w['w04']:+.1f} {w['w13']:+.1f} {w['w14']:+.1f} {w['w23']:+.1f} {w['w24']:+.1f}"
        word = syn["word"][:14]
        row_text = f"{i+1:3d}  {word:<16s}  {syn['language']:<5s}  {syn['model']:<18s}  {wstr:>35s}"

        font = pick_font(word, FONT_CREDITS_ROW, FONT_CREDITS_ROW_CJK)
        draw.text((60, y), row_text, fill=TEXT_WHITE, font=font)
        y += 28

    if len(synonyms) > max_rows:
        draw.text((60, y + 4), f"  ... and {len(synonyms) - max_rows} more",
                  fill=TEXT_DIM, font=FONT_CREDITS_ROW)

    # Footer
    foot = "Motion Gait Dictionary v2  \u00b7  3-link PyBullet robot  \u00b7  4000 steps @ 240 Hz"
    bbox = draw.textbbox((0, 0), foot, font=FONT_CREDITS_FOOT)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT - 40), foot,
              fill=TEXT_DIM, font=FONT_CREDITS_FOOT)

    return np.array(img).tobytes()


def burn_caption(frame_bytes, word, language, model, clip_num, total_clips):
    """Burn a lower-third caption onto a raw RGB frame."""
    img = Image.frombytes("RGB", (WIDTH, HEIGHT), frame_bytes)
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Semi-transparent bar at bottom
    bar_h = 52
    bar_y = HEIGHT - bar_h
    draw.rectangle([(0, bar_y), (WIDTH, HEIGHT)], fill=CAPTION_BG)

    # Caption text: "word" · lang · model
    caption = f'"{word}"  \u00b7  {language}  \u00b7  {model}'
    font = pick_font(word, FONT_CAPTION, FONT_CAPTION_CJK)
    draw.text((20, bar_y + 10), caption, fill=TEXT_WHITE, font=font)

    # Clip counter on right
    counter = f"{clip_num}/{total_clips}"
    bbox = draw.textbbox((0, 0), counter, font=FONT_COUNTER)
    tw = bbox[2] - bbox[0]
    draw.text((WIDTH - tw - 20, bar_y + 14), counter,
              fill=TEXT_DIM, font=FONT_COUNTER)

    # Composite
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")

    return np.array(img).tobytes()


# ── Simulation ──────────────────────────────────────────────────────────────

def write_brain(weights):
    """Write brain.nndf for standard 6-synapse topology."""
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
    """Render one simulation frame with camera following robot."""
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


def simulate_clip(weights, word, language, model, clip_num, total_clips, ffmpeg):
    """Run one simulation clip and pipe captioned frames to ffmpeg."""
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
        # Act
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

        # Capture frame
        if i % FRAME_EVERY_N == 0:
            pos, _ = safe_get_base_pose(robotId)
            frame = render_sim_frame(pos)
            frame = burn_caption(frame, word, language, model,
                                 clip_num, total_clips)
            try:
                ffmpeg.stdin.write(frame)
            except BrokenPipeError:
                break

    p.disconnect()


# ── Main pipeline ───────────────────────────────────────────────────────────

CONCEPT_DESCRIPTIONS = {
    "freeze": "perfectly still, no motion",
    "drag": "slow effortful movement, lots of energy for little progress",
    "hop": "bouncing vertical motion with forward progress",
    "retreat": "backward movement, moving away from forward direction",
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
    "headstand": "inverted, torso on the ground, legs in the air",
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
}


def record_concept(concept_id, entry, out_dir):
    """Record a complete video for one motion concept."""
    synonyms = entry["synonyms"]
    n = len(synonyms)
    description = CONCEPT_DESCRIPTIONS.get(concept_id, "")

    video_path = out_dir / f"{concept_id}.mp4"

    print(f"\n{'='*60}")
    print(f"  {concept_id}: {n} clips + title + credits")
    print(f"{'='*60}")

    # Start ffmpeg
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
    title_frame = render_title_card(concept_id, n, description)
    for _ in range(TITLE_SECONDS * FPS):
        ffmpeg.stdin.write(title_frame)

    # 2. Clips
    for i, syn in enumerate(synonyms):
        word = syn["word"]
        lang = syn["language"]
        model = syn["model"]
        weights = syn["weights"]
        print(f"  [{i+1}/{n}] \"{word}\" · {lang} · {model}")
        simulate_clip(weights, word, lang, model, i + 1, n, ffmpeg)

    # 3. Credits card
    print(f"  Credits card ({CREDITS_SECONDS}s)...")
    credits_frame = render_credits_card(concept_id, synonyms)
    for _ in range(CREDITS_SECONDS * FPS):
        ffmpeg.stdin.write(credits_frame)

    # Close
    ffmpeg.stdin.close()
    ffmpeg.wait(timeout=60)

    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {video_path} ({size_mb:.1f} MB)")
    return video_path


def main():
    # Load dictionary
    dict_path = PROJECT / "artifacts" / "motion_gait_dictionary_v2.json"
    with open(dict_path) as f:
        data = json.load(f)
    concepts = data["concepts"]

    # Parse args — specific concepts or all
    if len(sys.argv) > 1:
        requested = sys.argv[1:]
    else:
        requested = sorted(concepts.keys())

    # Validate
    for r in requested:
        if r not in concepts:
            print(f"Unknown concept: {r}")
            print(f"Available: {', '.join(sorted(concepts.keys()))}")
            sys.exit(1)

    out_dir = PROJECT / "videos" / "concepts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Backup brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    try:
        for concept_id in requested:
            record_concept(concept_id, concepts[concept_id], out_dir)
    finally:
        # Restore brain.nndf
        if backup_path.exists():
            shutil.copy2(backup_path, brain_path)

    print(f"\nDone! {len(requested)} concept videos in {out_dir}/")


if __name__ == "__main__":
    main()
