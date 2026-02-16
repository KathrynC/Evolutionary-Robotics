"""
Record Module K: Random Search video
5 robots with different random synaptic weights, title card + weight captions
"""
import os
import sys
import shutil
import tempfile
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont
import pyrosim.pyrosim as pyrosim
from solution import SOLUTION
import constants as c

# --- Video settings ---
WIDTH, HEIGHT = 1280, 720
FPS = 60
SIM_STEPS = 1000
CAPTURE_EVERY = 2       # capture every Nth physics step → 500 frames per robot
TITLE_SECONDS = 4
CAPTION_SECONDS = 3
NUM_ROBOTS = 5

# --- Camera ---
CAM_DISTANCE = 5.0
CAM_YAW = 30
CAM_PITCH = -25
CAM_TARGET = [0, 0, 0.5]

# --- Fonts ---
def load_font(size, bold=False):
    paths = [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Avenir Next.ttc",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size, index=1 if bold else 0)
        except (OSError, IndexError):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()

FONT_TITLE = load_font(52, bold=True)
FONT_SUBTITLE = load_font(32)
FONT_NAME = load_font(36, bold=True)
FONT_CAPTION = load_font(22)
FONT_WEIGHT = load_font(18)

# --- Colors ---
BG_DARK = (18, 18, 28)
ACCENT = (80, 180, 220)
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (160, 160, 170)
WEIGHT_POS = (100, 210, 130)
WEIGHT_NEG = (210, 100, 110)
CAPTION_BG = (18, 18, 28, 200)

def make_title_card():
    """Title card: module name, student, university."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    # Decorative line
    draw.rectangle([100, 200, WIDTH - 100, 203], fill=ACCENT)

    # Module title
    draw.text((WIDTH // 2, 260), "K. Random Search",
              fill=TEXT_WHITE, font=FONT_TITLE, anchor="mm")

    # Subtitle
    draw.text((WIDTH // 2, 330),
              "5 robots with different random synaptic weights",
              fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")

    # Decorative line
    draw.rectangle([100, 380, WIDTH - 100, 383], fill=ACCENT)

    # Name and university
    draw.text((WIDTH // 2, 440), "Kathryn Cramer",
              fill=TEXT_WHITE, font=FONT_NAME, anchor="mm")
    draw.text((WIDTH // 2, 490), "University of Vermont",
              fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")

    # Course info
    draw.text((WIDTH // 2, 560), "Evolutionary Robotics  \u2022  r/ludobots",
              fill=(120, 120, 130), font=FONT_CAPTION, anchor="mm")

    return img


def format_weight(w):
    """Format a weight value with sign."""
    return f"{w:+.3f}"


def make_weight_card(robot_num, weights):
    """Caption card showing weight matrix for one robot."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    # Robot header
    draw.text((WIDTH // 2, 140),
              f"Robot {robot_num + 1} of {NUM_ROBOTS}",
              fill=ACCENT, font=FONT_TITLE, anchor="mm")

    draw.rectangle([200, 190, WIDTH - 200, 193], fill=(60, 60, 70))

    # Column headers
    motor_names = ["BackLeg Motor", "FrontLeg Motor"]
    sensor_names = ["Torso Sensor", "BackLeg Sensor", "FrontLeg Sensor"]

    table_x = 340
    col_width = 220
    row_height = 55
    top_y = 240

    # Header row
    draw.text((WIDTH // 2, top_y - 30), "Synapse Weight Matrix",
              fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")

    for j, mname in enumerate(motor_names):
        x = table_x + j * col_width + col_width // 2
        draw.text((x, top_y + 15), mname,
                  fill=ACCENT, font=FONT_CAPTION, anchor="mm")

    # Data rows
    for i, sname in enumerate(sensor_names):
        y = top_y + 55 + i * row_height
        draw.text((table_x - 20, y + row_height // 2), sname,
                  fill=TEXT_DIM, font=FONT_CAPTION, anchor="rm")
        for j in range(c.numMotorNeurons):
            x = table_x + j * col_width + col_width // 2
            w = weights[i][j]
            color = WEIGHT_POS if w >= 0 else WEIGHT_NEG
            draw.text((x, y + row_height // 2), format_weight(w),
                      fill=color, font=FONT_NAME, anchor="mm")

    # Fitness note
    draw.text((WIDTH // 2, HEIGHT - 120),
              "Fitness = x-position of torso (more negative = further left = better)",
              fill=(100, 100, 110), font=FONT_CAPTION, anchor="mm")

    return img


def make_overlay(frame_img, robot_num, weights, fitness_so_far, step, total_steps):
    """Overlay weight caption bar on a simulation frame."""
    # Create RGBA overlay
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Top bar: robot number
    draw.rectangle([0, 0, WIDTH, 45], fill=(18, 18, 28, 180))
    draw.text((20, 22), f"Robot {robot_num + 1}/{NUM_ROBOTS}",
              fill=ACCENT, font=FONT_CAPTION, anchor="lm")

    # Progress bar
    progress = step / total_steps
    bar_x, bar_y, bar_w, bar_h = WIDTH - 220, 12, 200, 20
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                   outline=(80, 80, 90), width=1)
    draw.rectangle([bar_x + 1, bar_y + 1, bar_x + 1 + int((bar_w - 2) * progress), bar_y + bar_h - 1],
                   fill=ACCENT)

    # Bottom bar: weights + fitness
    bar_top = HEIGHT - 80
    draw.rectangle([0, bar_top, WIDTH, HEIGHT], fill=(18, 18, 28, 200))

    # Weights in a row
    weight_strs = []
    pairs = [("T\u2192B", weights[0][0]), ("T\u2192F", weights[0][1]),
             ("B\u2192B", weights[1][0]), ("B\u2192F", weights[1][1]),
             ("F\u2192B", weights[2][0]), ("F\u2192F", weights[2][1])]
    x_start = 30
    for label, w in pairs:
        color = WEIGHT_POS if w >= 0 else WEIGHT_NEG
        draw.text((x_start, bar_top + 20), label,
                  fill=TEXT_DIM, font=FONT_WEIGHT, anchor="lm")
        draw.text((x_start, bar_top + 48), format_weight(w),
                  fill=color, font=FONT_WEIGHT, anchor="lm")
        x_start += 130

    # Fitness on right
    if fitness_so_far is not None:
        fit_color = WEIGHT_POS if fitness_so_far < 0 else WEIGHT_NEG
        draw.text((WIDTH - 30, bar_top + 35),
                  f"x = {fitness_so_far:+.2f}",
                  fill=fit_color, font=FONT_NAME, anchor="rm")

    # Composite
    frame_rgba = frame_img.convert("RGBA")
    composited = Image.alpha_composite(frame_rgba, overlay)
    return composited.convert("RGB")


def make_results_card(fitnesses):
    """Final results card showing all 5 fitness values."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2, 120), "Results: Random Search",
              fill=TEXT_WHITE, font=FONT_TITLE, anchor="mm")
    draw.rectangle([200, 170, WIDTH - 200, 173], fill=ACCENT)

    best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])

    for i, fit in enumerate(fitnesses):
        y = 220 + i * 70
        color = ACCENT if i == best_idx else TEXT_DIM
        marker = "  \u2190 best" if i == best_idx else ""
        draw.text((WIDTH // 2, y + 20),
                  f"Robot {i + 1}:  fitness = {fit:+.4f}{marker}",
                  fill=color, font=FONT_NAME, anchor="mm")

    draw.rectangle([200, HEIGHT - 160, WIDTH - 200, HEIGHT - 157], fill=(60, 60, 70))
    draw.text((WIDTH // 2, HEIGHT - 110),
              "Fitness = torso x-position after 1000 timesteps",
              fill=(100, 100, 110), font=FONT_CAPTION, anchor="mm")

    return img


def render_pybullet_frame(phys_client, robot_id):
    """Capture one frame from PyBullet."""
    # Track robot position for camera
    pos, _ = p.getBasePositionAndOrientation(robot_id)

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[pos[0], pos[1], 0.5],
        distance=CAM_DISTANCE,
        yaw=CAM_YAW,
        pitch=CAM_PITCH,
        roll=0,
        upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=WIDTH / HEIGHT, nearVal=0.1, farVal=100)
    _, _, rgba, _, _ = p.getCameraImage(
        WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER)
    img = Image.frombytes("RGBA", (WIDTH, HEIGHT), bytes(rgba))
    return img.convert("RGB")


def save_frame(img, frame_dir, frame_num):
    """Save a frame as PNG."""
    img.save(os.path.join(frame_dir, f"frame_{frame_num:06d}.png"))


def simulate_robot(solution, frame_dir, frame_counter, robot_num):
    """Run one robot simulation, capturing frames."""
    # Connect PyBullet
    phys_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load world + robot
    p.loadURDF("plane.urdf")
    p.loadSDF("world.sdf")
    robot_id = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robot_id)

    # Load neural network
    from pyrosim.neuralNetwork import NEURAL_NETWORK
    from sensor import SENSOR
    from motor import MOTOR

    nn = NEURAL_NETWORK("brain" + str(solution.myID) + ".nndf")

    sensors = {}
    for nname in nn.neurons:
        if nn.neurons[nname].Is_Sensor_Neuron():
            sensors[nname] = SENSOR(nn.neurons[nname].Get_Link_Name())
    motors = {}
    for nname in nn.neurons:
        if nn.neurons[nname].Is_Motor_Neuron():
            motors[nname] = MOTOR(nn.neurons[nname].Get_Joint_Name())

    fitness_val = None
    captured = 0
    total_captures = SIM_STEPS // CAPTURE_EVERY

    for t in range(SIM_STEPS):
        p.stepSimulation()

        # Sense
        for nname in sensors:
            nn.neurons[nname].Set_Value(sensors[nname].Get_Value())

        # Think
        for nname in nn.neurons:
            if nn.neurons[nname].Is_Motor_Neuron():
                nn.neurons[nname].Set_Value(0.0)
                for sname in nn.synapses:
                    if nn.synapses[sname].Get_Target_Neuron_Name() == nname:
                        src = nn.synapses[sname].Get_Source_Neuron_Name()
                        nn.neurons[nname].Add_To_Value(
                            nn.neurons[src].Get_Value() * nn.synapses[sname].Get_Weight())
                nn.neurons[nname].Threshold()

        # Act
        for nname in motors:
            angle = nn.neurons[nname].Get_Value() * c.motorJointRange
            motors[nname].Set_Value(robot_id, angle)

        # Capture frame
        if t % CAPTURE_EVERY == 0:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            fitness_val = pos[0]

            frame = render_pybullet_frame(phys_client, robot_id)
            frame = make_overlay(frame, robot_num, solution.weights,
                                fitness_val, captured, total_captures)
            save_frame(frame, frame_dir, frame_counter)
            frame_counter += 1
            captured += 1

    # Final fitness
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    fitness_val = pos[0]
    p.disconnect()

    return frame_counter, fitness_val


def main():
    frame_dir = tempfile.mkdtemp(prefix="random_search_video_")
    print(f"Frames dir: {frame_dir}")
    frame_counter = 0

    # --- Title card ---
    print("Rendering title card...")
    title = make_title_card()
    for _ in range(TITLE_SECONDS * FPS):
        save_frame(title, frame_dir, frame_counter)
        frame_counter += 1

    fitnesses = []

    for i in range(NUM_ROBOTS):
        print(f"\n--- Robot {i + 1}/{NUM_ROBOTS} ---")

        # Create solution with random weights
        sol = SOLUTION(i)
        sol.Create_World()
        sol.Create_Body()
        sol.Create_Brain()
        print(f"Weights:\n{sol.weights}")

        # Weight caption card
        print("Rendering weight card...")
        wcard = make_weight_card(i, sol.weights)
        for _ in range(CAPTION_SECONDS * FPS):
            save_frame(wcard, frame_dir, frame_counter)
            frame_counter += 1

        # Simulate and capture
        print("Simulating...")
        frame_counter, fitness = simulate_robot(sol, frame_dir, frame_counter, i)
        fitnesses.append(fitness)
        print(f"Fitness: {fitness:+.4f}")

    # --- Results card ---
    print("\nRendering results card...")
    results = make_results_card(fitnesses)
    for _ in range(TITLE_SECONDS * FPS):
        save_frame(results, frame_dir, frame_counter)
        frame_counter += 1

    print(f"\nTotal frames: {frame_counter}")

    # --- Encode with ffmpeg ---
    output_file = "random_search_video.mp4"
    ffmpeg_bin = os.path.expanduser("~/miniforge3/bin/ffmpeg")
    print(f"Encoding {output_file}...")
    cmd = (f'{ffmpeg_bin} -y -framerate {FPS} -i "{frame_dir}/frame_%06d.png" '
           f'-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p '
           f'-vf "scale={WIDTH}:{HEIGHT}" "{output_file}"')
    ret = os.system(cmd)

    # Cleanup frames only if encoding succeeded
    if ret == 0:
        shutil.rmtree(frame_dir)
    else:
        print(f"WARNING: ffmpeg failed (exit {ret}). Frames kept at {frame_dir}")
    print(f"\nDone! Video saved to {output_file}")

    # Cleanup brain/fitness files
    for fname in os.listdir("."):
        if fname.startswith("brain") and fname.endswith(".nndf"):
            os.remove(fname)
        if fname.startswith("fitness") and fname.endswith(".txt"):
            os.remove(fname)

    # Print results
    print("\nResults:")
    for i, f in enumerate(fitnesses):
        marker = " <-- best" if f == min(fitnesses) else ""
        print(f"  Robot {i + 1}: fitness = {f:+.4f}{marker}")


if __name__ == "__main__":
    main()
