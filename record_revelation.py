#!/usr/bin/env python3
"""
record_revelation.py

Record video of the Revelation 6:8 "pale horse" gait — the highest-displacement
LLM-generated gait (DX=29.17m) — with a 5-second title card at the end showing
the weights and provenance.

Output: videos/revelation_6_8.mp4
"""
import subprocess, sys, os, shutil
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

import constants as c
from pyrosim.neuralNetwork import NEURAL_NETWORK
import pyrosim.pyrosim as pyrosim

# ---------- video settings ----------
WIDTH, HEIGHT = 1280, 720
FPS = 30
FRAME_EVERY_N = 4
CAMERA_DISTANCE = 3.0
CAMERA_YAW = 60.0
CAMERA_PITCH = -25.0
CAMERA_TARGET_Z_OFFSET = 0.5
TITLE_SECONDS = 5

# ---------- Revelation 6:8 weights ----------
WEIGHTS = {
    "w03": -0.8,  "w04":  0.6,
    "w13":  0.2,  "w14": -0.9,
    "w23":  0.5,  "w24": -0.4,
}

SEED_TEXT = (
    "Revelation 6:8 — And I looked, and behold a pale horse:\n"
    "and his name that sat on him was Death."
)


def write_brain(weights):
    path = os.path.join(PROJECT, "brain.nndf")
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for syn, w in weights.items():
            src, tgt = syn[1], syn[2]
            f.write(f'    <synapse sourceNeuronName = "{src}" targetNeuronName = "{tgt}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def safe_get_base_pose(body_id):
    try:
        return p.getBasePositionAndOrientation(body_id)
    except Exception:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)


def render_frame(robot_pos):
    target = [robot_pos[0], robot_pos[1], robot_pos[2] + CAMERA_TARGET_Z_OFFSET]
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=CAMERA_DISTANCE,
        yaw=CAMERA_YAW,
        pitch=CAMERA_PITCH,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=WIDTH / HEIGHT, nearVal=0.1, farVal=100)
    _, _, rgba, _, _ = p.getCameraImage(
        WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    rgb = np.array(rgba, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]
    return rgb.tobytes()


def make_title_card(dx, dy, speed):
    """Render the title card as a WIDTH x HEIGHT RGB frame using matplotlib."""
    dpi = 100
    fig = plt.figure(figsize=(WIDTH / dpi, HEIGHT / dpi), dpi=dpi, facecolor="#1a1a2e")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    # Title
    ax.text(0.5, 0.92, "REVELATION 6:8", fontsize=32, fontweight="bold",
            color="#e6b800", ha="center", va="top", fontfamily="serif")

    # Seed text
    ax.text(0.5, 0.82, SEED_TEXT, fontsize=14, color="#cccccc",
            ha="center", va="top", fontfamily="serif", fontstyle="italic",
            linespacing=1.4)

    # Weights table
    y_start = 0.68
    ax.text(0.5, y_start, "SYNAPSE WEIGHTS", fontsize=16, fontweight="bold",
            color="#e6b800", ha="center", va="top", fontfamily="monospace")

    # Neuron labels
    labels = [
        ("Sensor", "Motor", "Synapse", "Weight"),
        ("Torso (0)", "BackLeg (3)", "w03", f"{WEIGHTS['w03']:+.1f}"),
        ("Torso (0)", "FrontLeg (4)", "w04", f"{WEIGHTS['w04']:+.1f}"),
        ("BackLeg (1)", "BackLeg (3)", "w13", f"{WEIGHTS['w13']:+.1f}"),
        ("BackLeg (1)", "FrontLeg (4)", "w14", f"{WEIGHTS['w14']:+.1f}"),
        ("FrontLeg (2)", "BackLeg (3)", "w23", f"{WEIGHTS['w23']:+.1f}"),
        ("FrontLeg (2)", "FrontLeg (4)", "w24", f"{WEIGHTS['w24']:+.1f}"),
    ]

    for i, (s, m, syn, w) in enumerate(labels):
        y = y_start - 0.045 * (i + 1)
        if i == 0:
            color = "#e6b800"
            fs = 11
            fw = "bold"
        else:
            # Color negative weights red, positive green
            wval = WEIGHTS[syn]
            color = "#ff6b6b" if wval < 0 else "#51cf66"
            fs = 12
            fw = "normal"
        ax.text(0.12, y, s, fontsize=fs, color=color, ha="left", fontfamily="monospace", fontweight=fw)
        ax.text(0.40, y, m, fontsize=fs, color=color, ha="left", fontfamily="monospace", fontweight=fw)
        ax.text(0.68, y, syn, fontsize=fs, color=color, ha="left", fontfamily="monospace", fontweight=fw)
        ax.text(0.82, y, w, fontsize=fs, color=color, ha="left", fontfamily="monospace", fontweight=fw)

    # Performance
    y_perf = y_start - 0.045 * 9
    ax.text(0.5, y_perf, "PERFORMANCE", fontsize=16, fontweight="bold",
            color="#e6b800", ha="center", va="top", fontfamily="monospace")
    perf_lines = [
        f"DX = {dx:+.2f}m    DY = {dy:+.2f}m    Speed = {speed:.2f} m/s",
        f"Evolved best: 85.09m (from 29.17m start, 500 evals)",
    ]
    for j, line in enumerate(perf_lines):
        ax.text(0.5, y_perf - 0.04 * (j + 1), line, fontsize=11,
                color="#cccccc", ha="center", va="top", fontfamily="monospace")

    # Provenance
    y_prov = y_perf - 0.04 * 4
    ax.text(0.5, y_prov,
            "LLM-generated (Ollama/llama3) from biblical verse prompt",
            fontsize=10, color="#888888", ha="center", va="top", fontfamily="serif")
    ax.text(0.5, y_prov - 0.035,
            "Synapse Gait Zoo  \u2022  Cramer 2024-2025  \u2022  PyBullet 3.25",
            fontsize=10, color="#888888", ha="center", va="top", fontfamily="serif")

    # Render to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3]  # drop alpha
    # Ensure correct size
    if arr.shape != (HEIGHT, WIDTH, 3):
        from PIL import Image
        img = Image.fromarray(arr)
        img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
        arr = np.array(img)
    plt.close(fig)
    return arr.tobytes()


def main():
    vid_dir = os.path.join(PROJECT, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    # Backup brain.nndf
    brain_path = os.path.join(PROJECT, "brain.nndf")
    backup_path = brain_path + ".backup"
    if os.path.exists(brain_path):
        shutil.copy2(brain_path, backup_path)

    write_brain(WEIGHTS)

    print("=" * 60)
    print("  Recording: Revelation 6:8 — Death on a pale horse")
    print(f"  Weights: {WEIGHTS}")
    print("=" * 60)

    # Connect headless
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

    video_path = os.path.join(vid_dir, "revelation_6_8.mp4")
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
         video_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    start_pos = safe_get_base_pose(robotId)[0]
    start_x = start_pos[0]
    frame_count = 0

    for i in range(c.SIM_STEPS):
        for neuronName in nn.neurons:
            n = nn.neurons[neuronName]
            if n.Is_Motor_Neuron():
                jn = n.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, n.Get_Value(), c.MAX_FORCE)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, p.POSITION_CONTROL, n.Get_Value(), c.MAX_FORCE)

        p.stepSimulation()
        nn.Update()

        if i % FRAME_EVERY_N == 0:
            pos, _ = safe_get_base_pose(robotId)
            frame = render_frame(pos)
            try:
                ffmpeg.stdin.write(frame)
                frame_count += 1
            except BrokenPipeError:
                break

    end_pos = safe_get_base_pose(robotId)[0]
    dx = end_pos[0] - start_x
    dy = end_pos[1] - start_pos[1]
    speed = np.sqrt(dx**2 + dy**2) / (c.SIM_STEPS * c.DT)

    p.disconnect()

    print(f"  Simulation: DX = {dx:+.2f}m, DY = {dy:+.2f}m")
    print(f"  Sim frames: {frame_count}")

    # Render title card (5 seconds)
    print("  Rendering title card...")
    title_frame = make_title_card(dx, dy, speed)
    title_frames = TITLE_SECONDS * FPS
    for _ in range(title_frames):
        try:
            ffmpeg.stdin.write(title_frame)
        except BrokenPipeError:
            break

    ffmpeg.stdin.close()
    ffmpeg.wait(timeout=60)

    # Restore brain.nndf
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, brain_path)

    total_frames = frame_count + title_frames
    total_sec = total_frames / FPS
    print(f"  Total: {total_frames} frames ({total_sec:.1f}s)")
    print(f"  Saved: {video_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
