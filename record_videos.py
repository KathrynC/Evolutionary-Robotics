#!/usr/bin/env python3
"""
record_videos.py

Role:
    Record gait videos for the Synapse Gait Zoo using PyBullet offscreen rendering
    (DIRECT mode + ER_TINY_RENDERER) piped to ffmpeg. No GUI or screen recording needed.

Supports:
    - Standard 6-synapse sensor-to-motor configs (via `configs` list)
    - Extended 10-synapse configs with motor-to-motor cross-wiring (w34, w43, w33, w44)
    - Arbitrary neuron/synapse configs including hidden neurons (via `hidden_configs` list)

Pipeline:
    1. Back up brain.nndf
    2. For each gait configuration, write a custom brain.nndf
    3. Run a headless PyBullet simulation (c.SIM_STEPS steps)
    4. Pipe raw RGB frames to ffmpeg to produce H.264 MP4
    5. Report DX displacement for each gait
    6. Restore the original brain.nndf

Outputs:
    videos/<gait_name>.mp4 for each configured gait

Notes:
    - FRAME_EVERY_N controls temporal downsampling (default 4: ~1000 frames from 4000 steps).
    - Camera follows the robot with fixed yaw/pitch/distance.
    - brain.nndf is a shared file: this script backs it up and restores it afterward.
    - Requires ffmpeg to be installed and available on PATH.

Ludobots role:
    - Video recording of NN-driven locomotion for visual evaluation.

Beyond Ludobots (this repo):
    - Supports cross-wired CPG topologies and hidden neuron architectures for the
      full gait zoo catalog. Musical time-signature metaphors map to synapse topology.
"""
import subprocess, sys, os, struct, shutil
import numpy as np
import pybullet as p
import pybullet_data

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

import constants as c
from pyrosim.neuralNetwork import NEURAL_NETWORK
import pyrosim.pyrosim as pyrosim

# ---------- video settings ----------
WIDTH, HEIGHT = 1280, 720
FPS = 30
FRAME_EVERY_N = 4          # capture 1 frame every N sim steps (4000/4 = 1000 frames ≈ 33s @ 30fps)
CAMERA_DISTANCE = 3.0
CAMERA_YAW = 60.0
CAMERA_PITCH = -25.0
CAMERA_TARGET_Z_OFFSET = 0.5

# ---------- hidden neuron configurations ----------
# (name, neurons, synapses, description)
# neurons: list of (id, type, ref) tuples
# synapses: list of (src, tgt, weight) tuples

BASE_NEURONS = [
    ("0", "sensor", "Torso"), ("1", "sensor", "BackLeg"),
    ("2", "sensor", "FrontLeg"), ("3", "motor", "Torso_BackLeg"),
    ("4", "motor", "Torso_FrontLeg"),
]

hidden_configs = [
    ("43_hidden_cpg_champion",
     BASE_NEURONS + [("5", "hidden", None), ("6", "hidden", None)],
     [("1","5",-0.6), ("2","6",-0.6),
      ("5","6", 0.7), ("6","5",-0.7),
      ("5","3",-0.8), ("6","4", 0.8),
      ("0","3",-0.3), ("0","4", 0.3)],
     "Hidden CPG — half-center oscillator, DX=+50.11, all-time champion"),
]

# ---------- musical time signature configurations ----------
# (name, w03, w13, w23, w04, w14, w24, w34, w43, w33, w44, description)
#
# Time signatures map to synapse topology:
#   Beat STRENGTH → weight magnitude
#   Beat GROUPING → which neurons couple
#   ACCENT PATTERN → motor asymmetry
#   Cross-wiring → internal metronome/groove
configs = [
    ("36_take_five",
     -0.8, -0.8, -0.5,  0.5,  0.8,  0.8,         # 3+2 grouping (3 strong, 2 weak)
      0.5, -0.5,  0.3, -0.3,                       # full CPG — the jazz pulse
     "5/4 Take Five — asymmetric 3+2 with CPG"),

    ("37_hemiola",
     -0.5, -1.0, -0.5,  1.0,  0.5,  1.0,         # 3/4↔6/8 ambiguity
      0.0,  0.0,  0.0,  0.0,                       # no cross-wiring — the ambiguity IS the rhythm
     "Hemiola — Bernstein's America, 3/4↔6/8 ambiguity"),

    ("38_heavy_waltz",
     -1.0, -0.2, -0.2,  0.2,  0.2,  1.0,         # ONE-two-three (beat 1 is 5× beats 2&3)
      0.5, -0.5,  0.2, -0.2,                       # CPG reinforces the downbeat
     "3/4 Heavy Waltz — strong ONE, ghosted 2&3, reversal ratio 1:380"),

    ("39_bulgarian",
     -0.57, -0.57, -0.86,  0.86,  0.57,  0.57,   # 2+2+3 ratio
      0.0,  0.0,  0.0,  0.0,                       # no cross-wiring — pure meter
     "7/8 Bulgarian — the limping meter of Balkan folk"),

    ("40_rubato",
     -0.1, -0.1, -0.1,  0.1,  0.1,  0.1,         # near-zero sensor drive
      0.7, -0.7,  0.5, -0.5,                       # strong internal CPG
     "Rubato (Chopin) — rhythm from within, reversal ratio 40:2192"),

    ("41_blues_shuffle",
     -0.67, -0.33, -1.0,  1.0,  0.33,  0.67,     # swing ratio 2:1 (long-short)
      0.4, -0.4,  0.0,  0.0,                       # cross-wired swing feel
     "12/8 Blues Shuffle — swung long-short-long"),

    ("42_polyrhythm",
     -0.75, -0.5, -0.75,  0.5,  0.5,  0.5,       # BackLeg in 4, FrontLeg in 3
      0.0,  0.0,  0.0,  0.0,                       # no cross-wiring — let the meters collide
     "4:3 Polyrhythm — two meters simultaneously"),
]


def write_brain_crosswired(w03, w13, w23, w04, w14, w24,
                            w34=0.0, w43=0.0, w33=0.0, w44=0.0):
    """Write brain.nndf with standard 6-synapse topology plus optional cross-wiring.

    Args:
        w03, w13, w23: Sensor-to-BackLeg-motor weights (neurons 0,1,2 -> 3).
        w04, w14, w24: Sensor-to-FrontLeg-motor weights (neurons 0,1,2 -> 4).
        w34: Motor 3 -> Motor 4 cross-wiring weight (default 0 = absent).
        w43: Motor 4 -> Motor 3 cross-wiring weight (default 0 = absent).
        w33: Motor 3 self-feedback weight (default 0 = absent).
        w44: Motor 4 self-feedback weight (default 0 = absent).

    Side effects:
        Overwrites brain.nndf in the project directory.
    """
    path = os.path.join(PROJECT, "brain.nndf")
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        # Standard sensor→motor (6 synapses)
        f.write(f'    <synapse sourceNeuronName = "0" targetNeuronName = "3" weight = "{w03}" />\n')
        f.write(f'    <synapse sourceNeuronName = "1" targetNeuronName = "3" weight = "{w13}" />\n')
        f.write(f'    <synapse sourceNeuronName = "2" targetNeuronName = "3" weight = "{w23}" />\n')
        f.write(f'    <synapse sourceNeuronName = "0" targetNeuronName = "4" weight = "{w04}" />\n')
        f.write(f'    <synapse sourceNeuronName = "1" targetNeuronName = "4" weight = "{w14}" />\n')
        f.write(f'    <synapse sourceNeuronName = "2" targetNeuronName = "4" weight = "{w24}" />\n')
        # Recurrent motor→motor cross-wiring (0-4 additional synapses)
        if w34 != 0.0:
            f.write(f'    <synapse sourceNeuronName = "3" targetNeuronName = "4" weight = "{w34}" />\n')
        if w43 != 0.0:
            f.write(f'    <synapse sourceNeuronName = "4" targetNeuronName = "3" weight = "{w43}" />\n')
        if w33 != 0.0:
            f.write(f'    <synapse sourceNeuronName = "3" targetNeuronName = "3" weight = "{w33}" />\n')
        if w44 != 0.0:
            f.write(f'    <synapse sourceNeuronName = "4" targetNeuronName = "4" weight = "{w44}" />\n')
        f.write('</neuralNetwork>\n')


def write_brain_full(neurons, synapses):
    """Write brain.nndf with arbitrary neurons (including hidden) and synapses.

    Args:
        neurons: List of (name, type, ref) tuples. type is "sensor", "motor",
            or "hidden". ref is the link/joint name (None for hidden neurons).
        synapses: List of (source_name, target_name, weight) tuples.
            Zero-weight synapses are omitted from the output.

    Side effects:
        Overwrites brain.nndf in the project directory.
    """
    path = os.path.join(PROJECT, "brain.nndf")
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        for name, ntype, ref in neurons:
            if ntype == "sensor":
                f.write(f'    <neuron name = "{name}" type = "sensor" linkName = "{ref}" />\n')
            elif ntype == "motor":
                f.write(f'    <neuron name = "{name}" type = "motor"  jointName = "{ref}" />\n')
            else:
                f.write(f'    <neuron name = "{name}" type = "hidden" />\n')
        for src, tgt, w in synapses:
            if w != 0.0:
                f.write(f'    <synapse sourceNeuronName = "{src}" targetNeuronName = "{tgt}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def safe_get_base_pose(body_id):
    """Return (position, orientation) for a body, with a safe fallback on error.

    Args:
        body_id: PyBullet body ID.

    Returns:
        Tuple of (position_xyz, quaternion_xyzw). Falls back to origin/identity
        if the query fails (e.g., body was already removed).
    """
    try:
        return p.getBasePositionAndOrientation(body_id)
    except Exception:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)


def render_frame(robot_pos):
    """Render one frame with camera following the robot.

    Args:
        robot_pos: (x, y, z) position of the robot base.

    Returns:
        Raw RGB bytes (HEIGHT x WIDTH x 3) suitable for piping to ffmpeg.
    """
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


def run_one(name, w03, w13, w23, w04, w14, w24, w34, w43, w33, w44, desc, vid_dir):
    """Run one simulation with cross-wired topology and produce a video.

    Args:
        name: Gait identifier (used as the video filename stem).
        w03..w24: Six standard sensor-to-motor synapse weights.
        w34, w43, w33, w44: Four optional motor cross-wiring weights.
        desc: Human-readable description (for console logging only).
        vid_dir: Directory to write the output MP4 file.

    Returns:
        Tuple of (dx, video_path) where dx is the X-axis displacement in meters
        and video_path is the absolute path to the saved MP4.

    Side effects:
        - Overwrites brain.nndf via write_brain_crosswired.
        - Creates/overwrites vid_dir/<name>.mp4.
        - Spawns an ffmpeg subprocess for video encoding.
    """
    write_brain_crosswired(w03, w13, w23, w04, w14, w24, w34, w43, w33, w44)

    # Connect headless
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(c.DT)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    # Friction
    mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
    for link in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, link, lateralFriction=mu, restitution=0.0)

    nn = NEURAL_NETWORK("brain.nndf")

    # Start ffmpeg pipe
    video_path = os.path.join(vid_dir, f"{name}.mp4")
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

    start_x = safe_get_base_pose(robotId)[0][0]

    for i in range(c.SIM_STEPS):
        # Act: send NN motor values to joints
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

        # Think: update NN from sensors
        nn.Update()

        # Capture frame
        if i % FRAME_EVERY_N == 0:
            pos, _ = safe_get_base_pose(robotId)
            frame = render_frame(pos)
            try:
                ffmpeg.stdin.write(frame)
            except BrokenPipeError:
                break

    end_x = safe_get_base_pose(robotId)[0][0]
    dx = end_x - start_x

    # Close ffmpeg
    ffmpeg.stdin.close()
    ffmpeg.wait(timeout=30)

    p.disconnect()
    return dx, video_path


def run_one_full(name, neurons, synapses, desc, vid_dir):
    """Run one simulation with arbitrary neuron/synapse topology and produce a video.

    Args:
        name: Gait identifier (used as the video filename stem).
        neurons: List of (name, type, ref) tuples defining the network architecture.
        synapses: List of (source, target, weight) tuples defining connectivity.
        desc: Human-readable description (for console logging only).
        vid_dir: Directory to write the output MP4 file.

    Returns:
        Tuple of (dx, video_path) where dx is the X-axis displacement in meters
        and video_path is the absolute path to the saved MP4.

    Side effects:
        - Overwrites brain.nndf via write_brain_full.
        - Creates/overwrites vid_dir/<name>.mp4.
        - Spawns an ffmpeg subprocess for video encoding.
    """
    write_brain_full(neurons, synapses)

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

    video_path = os.path.join(vid_dir, f"{name}.mp4")
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

    start_x = safe_get_base_pose(robotId)[0][0]

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
            except BrokenPipeError:
                break

    end_x = safe_get_base_pose(robotId)[0][0]
    dx = end_x - start_x

    ffmpeg.stdin.close()
    ffmpeg.wait(timeout=30)

    p.disconnect()
    return dx, video_path


def main():
    """Record videos for all configured gaits (standard + hidden topologies).

    Side effects:
        - Backs up and restores brain.nndf.
        - Writes MP4 files to videos/ directory.
        - Prints DX displacement and file paths to stdout.
    """
    vid_dir = os.path.join(PROJECT, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    # Backup brain.nndf
    shutil.copy2(os.path.join(PROJECT, "brain.nndf"),
                 os.path.join(PROJECT, "brain.nndf.backup"))

    for cfg in configs:
        name, w03, w13, w23, w04, w14, w24, w34, w43, w33, w44, desc = cfg
        cross_str = f"[{w34:+.1f}, {w43:+.1f}, {w33:+.1f}, {w44:+.1f}]"
        print(f"\n{'='*60}")
        print(f"  Recording: {name} — {desc}")
        print(f"  Sensor→Motor BackLeg:  [{w03:+.1f}, {w13:+.1f}, {w23:+.1f}]")
        print(f"  Sensor→Motor FrontLeg: [{w04:+.1f}, {w14:+.1f}, {w24:+.1f}]")
        print(f"  Cross-wiring [3→4, 4→3, 3→3, 4→4]: {cross_str}")
        print(f"{'='*60}")

        dx, path = run_one(name, w03, w13, w23, w04, w14, w24,
                           w34, w43, w33, w44, desc, vid_dir)
        print(f"  DX = {dx:+.2f}")
        print(f"  Saved: {path}")

    for cfg in hidden_configs:
        name, neurons, synapses, desc = cfg
        n_hidden = len([n for n in neurons if n[1] == "hidden"])
        n_syn = len([s for s in synapses if s[2] != 0])
        print(f"\n{'='*60}")
        print(f"  Recording: {name} — {desc}")
        print(f"  Neurons: {len(neurons)} ({n_hidden} hidden)")
        print(f"  Synapses ({n_syn}):")
        for s, t, w in synapses:
            if w != 0:
                print(f"    {s}→{t}: {w:+.4f}")
        print(f"{'='*60}")

        dx, path = run_one_full(name, neurons, synapses, desc, vid_dir)
        print(f"  DX = {dx:+.2f}")
        print(f"  Saved: {path}")

    # Restore original brain.nndf
    backup = os.path.join(PROJECT, "brain.nndf.backup")
    if os.path.exists(backup):
        shutil.copy2(backup, os.path.join(PROJECT, "brain.nndf"))

    total = len(configs) + len(hidden_configs)
    print(f"\n{'='*60}")
    print(f"  All {total} videos recorded!")
    print(f"  Files in: {vid_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
