"""
Record Module L: The Hill Climber video
Initial simulation → evolution data → final evolved simulation
"""
import os
import copy
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
CAPTURE_EVERY = 2
TITLE_SECONDS = 4
CAPTION_SECONDS = 3
NUM_GENERATIONS = 20

# --- Camera ---
CAM_DISTANCE = 5.0
CAM_YAW = 30
CAM_PITCH = -25

# --- Colors ---
BG_DARK = (18, 18, 28)
ACCENT = (80, 180, 220)
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (160, 160, 170)
WEIGHT_POS = (100, 210, 130)
WEIGHT_NEG = (210, 100, 110)
IMPROVED_COLOR = (80, 220, 120)
REJECTED_COLOR = (120, 120, 130)

# --- Fonts ---
def load_font(size, bold=False):
    paths = [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
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
FONT_DATA = load_font(20)
FONT_DATA_BOLD = load_font(20, bold=True)
FONT_BIG_NUM = load_font(64, bold=True)


def make_title_card():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 200, WIDTH - 100, 203], fill=ACCENT)
    draw.text((WIDTH // 2, 260), "L. The Hill Climber",
              fill=TEXT_WHITE, font=FONT_TITLE, anchor="mm")
    draw.text((WIDTH // 2, 330),
              "Evolving a 3-link robot through iterative mutation and selection",
              fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")
    draw.rectangle([100, 380, WIDTH - 100, 383], fill=ACCENT)
    draw.text((WIDTH // 2, 440), "Kathryn Cramer",
              fill=TEXT_WHITE, font=FONT_NAME, anchor="mm")
    draw.text((WIDTH // 2, 490), "University of Vermont",
              fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")
    draw.text((WIDTH // 2, 560), "Evolutionary Robotics  \u2022  r/ludobots",
              fill=(120, 120, 130), font=FONT_CAPTION, anchor="mm")
    return img


def format_weight(w):
    return f"{w:+.3f}"


def make_phase_card(phase_text, sub_text=""):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)
    draw.text((WIDTH // 2, HEIGHT // 2 - 30), phase_text,
              fill=ACCENT, font=FONT_TITLE, anchor="mm")
    if sub_text:
        draw.text((WIDTH // 2, HEIGHT // 2 + 40), sub_text,
                  fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")
    return img


def make_weight_card(label, weights, fitness=None):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2, 100), label,
              fill=ACCENT, font=FONT_TITLE, anchor="mm")
    draw.rectangle([200, 150, WIDTH - 200, 153], fill=(60, 60, 70))

    motor_names = ["BackLeg Motor", "FrontLeg Motor"]
    sensor_names = ["Torso Sensor", "BackLeg Sensor", "FrontLeg Sensor"]
    table_x = 340
    col_width = 220
    row_height = 55
    top_y = 200

    draw.text((WIDTH // 2, top_y - 20), "Synapse Weight Matrix",
              fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")
    for j, mname in enumerate(motor_names):
        x = table_x + j * col_width + col_width // 2
        draw.text((x, top_y + 20), mname, fill=ACCENT, font=FONT_CAPTION, anchor="mm")
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

    if fitness is not None:
        fit_color = WEIGHT_POS if fitness < 0 else WEIGHT_NEG
        draw.text((WIDTH // 2, HEIGHT - 140), f"Fitness: {fitness:+.4f}",
                  fill=fit_color, font=FONT_NAME, anchor="mm")
        draw.text((WIDTH // 2, HEIGHT - 100),
                  "(x-position of torso: more negative = further left = better)",
                  fill=(100, 100, 110), font=FONT_CAPTION, anchor="mm")
    return img


def make_evolution_card(gen_data, up_to_gen):
    """Show evolution log building up line by line."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2, 40), "Hill Climber Evolution",
              fill=TEXT_WHITE, font=FONT_NAME, anchor="mm")
    draw.rectangle([100, 70, WIDTH - 100, 73], fill=ACCENT)

    # Column headers
    y = 90
    draw.text((80, y), "Gen", fill=ACCENT, font=FONT_DATA_BOLD, anchor="lm")
    draw.text((160, y), "Parent", fill=ACCENT, font=FONT_DATA_BOLD, anchor="lm")
    draw.text((370, y), "Child", fill=ACCENT, font=FONT_DATA_BOLD, anchor="lm")
    draw.text((570, y), "Result", fill=ACCENT, font=FONT_DATA_BOLD, anchor="lm")
    draw.text((800, y), "Best So Far", fill=ACCENT, font=FONT_DATA_BOLD, anchor="lm")

    # Determine how many lines fit — show a scrolling window
    line_height = 26
    max_lines = 20
    start = max(0, up_to_gen - max_lines)

    for idx in range(start, min(up_to_gen, len(gen_data))):
        g = gen_data[idx]
        row_y = 120 + (idx - start) * line_height
        if row_y > HEIGHT - 80:
            break

        gen_num = g["gen"]
        parent_fit = g["parent_fitness"]
        child_fit = g["child_fitness"]
        improved = g["improved"]
        best = g["best_so_far"]

        draw.text((80, row_y), f"{gen_num:2d}",
                  fill=TEXT_DIM, font=FONT_DATA, anchor="lm")
        draw.text((160, row_y), f"{parent_fit:+.4f}",
                  fill=TEXT_WHITE, font=FONT_DATA, anchor="lm")

        # Arrow
        draw.text((310, row_y), "\u2192", fill=TEXT_DIM, font=FONT_DATA, anchor="lm")

        child_color = IMPROVED_COLOR if improved else REJECTED_COLOR
        draw.text((370, row_y), f"{child_fit:+.4f}",
                  fill=child_color, font=FONT_DATA, anchor="lm")

        result_text = "\u2714 improved" if improved else "\u2718 rejected"
        result_color = IMPROVED_COLOR if improved else REJECTED_COLOR
        draw.text((570, row_y), result_text,
                  fill=result_color, font=FONT_DATA, anchor="lm")

        best_color = WEIGHT_POS if best < 0 else WEIGHT_NEG
        draw.text((800, row_y), f"{best:+.4f}",
                  fill=best_color, font=FONT_DATA_BOLD, anchor="lm")

    # Bottom summary
    if up_to_gen > 0 and up_to_gen <= len(gen_data):
        current = gen_data[up_to_gen - 1]
        draw.rectangle([0, HEIGHT - 60, WIDTH, HEIGHT], fill=(25, 25, 38))
        improvements = sum(1 for g in gen_data[:up_to_gen] if g["improved"])
        draw.text((WIDTH // 2, HEIGHT - 30),
                  f"Generation {current['gen']}/{NUM_GENERATIONS}  |  "
                  f"{improvements} improvements  |  "
                  f"Best: {current['best_so_far']:+.4f}",
                  fill=ACCENT, font=FONT_CAPTION, anchor="mm")

    return img


def make_sim_overlay(frame_img, label, weights, fitness_val, step, total_steps):
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Top bar
    draw.rectangle([0, 0, WIDTH, 45], fill=(18, 18, 28, 180))
    draw.text((20, 22), label, fill=ACCENT, font=FONT_CAPTION, anchor="lm")

    # Progress bar
    progress = step / total_steps
    bar_x, bar_y, bar_w, bar_h = WIDTH - 220, 12, 200, 20
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                   outline=(80, 80, 90), width=1)
    draw.rectangle([bar_x + 1, bar_y + 1,
                    bar_x + 1 + int((bar_w - 2) * progress), bar_y + bar_h - 1],
                   fill=ACCENT)

    # Bottom bar
    bar_top = HEIGHT - 80
    draw.rectangle([0, bar_top, WIDTH, HEIGHT], fill=(18, 18, 28, 200))

    pairs = [("T\u2192B", weights[0][0]), ("T\u2192F", weights[0][1]),
             ("B\u2192B", weights[1][0]), ("B\u2192F", weights[1][1]),
             ("F\u2192B", weights[2][0]), ("F\u2192F", weights[2][1])]
    x_start = 30
    for wlabel, w in pairs:
        color = WEIGHT_POS if w >= 0 else WEIGHT_NEG
        draw.text((x_start, bar_top + 20), wlabel,
                  fill=TEXT_DIM, font=FONT_WEIGHT, anchor="lm")
        draw.text((x_start, bar_top + 48), format_weight(w),
                  fill=color, font=FONT_WEIGHT, anchor="lm")
        x_start += 130

    if fitness_val is not None:
        fit_color = WEIGHT_POS if fitness_val < 0 else WEIGHT_NEG
        draw.text((WIDTH - 30, bar_top + 35), f"x = {fitness_val:+.2f}",
                  fill=fit_color, font=FONT_NAME, anchor="rm")

    frame_rgba = frame_img.convert("RGBA")
    composited = Image.alpha_composite(frame_rgba, overlay)
    return composited.convert("RGB")


def make_comparison_card(initial_fitness, final_fitness, initial_weights, final_weights):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2, 60), "Hill Climber Results",
              fill=TEXT_WHITE, font=FONT_TITLE, anchor="mm")
    draw.rectangle([100, 100, WIDTH - 100, 103], fill=ACCENT)

    # Side by side comparison
    left_x = WIDTH // 4
    right_x = 3 * WIDTH // 4

    draw.text((left_x, 150), "Initial (Random)",
              fill=TEXT_DIM, font=FONT_NAME, anchor="mm")
    draw.text((right_x, 150), f"Evolved (Gen {NUM_GENERATIONS})",
              fill=ACCENT, font=FONT_NAME, anchor="mm")

    # Big fitness numbers
    init_color = WEIGHT_NEG
    final_color = WEIGHT_POS if final_fitness < initial_fitness else WEIGHT_NEG
    draw.text((left_x, 230), f"{initial_fitness:+.2f}",
              fill=init_color, font=FONT_BIG_NUM, anchor="mm")
    draw.text((right_x, 230), f"{final_fitness:+.2f}",
              fill=final_color, font=FONT_BIG_NUM, anchor="mm")

    # Arrow
    draw.text((WIDTH // 2, 230), "\u2192", fill=ACCENT, font=FONT_BIG_NUM, anchor="mm")

    # Improvement
    improvement = initial_fitness - final_fitness
    draw.text((WIDTH // 2, 320),
              f"Improvement: {improvement:+.2f} (moved {abs(improvement):.1f} units further left)",
              fill=IMPROVED_COLOR, font=FONT_NAME, anchor="mm")

    draw.rectangle([100, 370, WIDTH - 100, 373], fill=(60, 60, 70))

    # Weight matrices side by side
    draw.text((left_x, 400), "Weights:", fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")
    draw.text((right_x, 400), "Weights:", fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")

    sensor_labels = ["T", "B", "F"]
    for i in range(3):
        y = 435 + i * 35
        # Initial weights
        for j in range(2):
            w = initial_weights[i][j]
            color = WEIGHT_POS if w >= 0 else WEIGHT_NEG
            x = left_x - 60 + j * 120
            draw.text((x, y), f"{sensor_labels[i]}: {format_weight(w)}",
                      fill=color, font=FONT_DATA, anchor="lm")
        # Final weights
        for j in range(2):
            w = final_weights[i][j]
            changed = abs(w - initial_weights[i][j]) > 0.001
            color = WEIGHT_POS if w >= 0 else WEIGHT_NEG
            x = right_x - 60 + j * 120
            text = f"{sensor_labels[i]}: {format_weight(w)}"
            draw.text((x, y), text, fill=color, font=FONT_DATA, anchor="lm")
            if changed:
                draw.text((x + 130, y), "\u2190", fill=ACCENT, font=FONT_DATA, anchor="lm")

    draw.text((WIDTH // 2, HEIGHT - 60),
              "Fitness = torso x-position after 1000 timesteps  \u2022  more negative = better",
              fill=(100, 100, 110), font=FONT_CAPTION, anchor="mm")

    return img


def render_pybullet_frame(robot_id):
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[pos[0], pos[1], 0.5],
        distance=CAM_DISTANCE, yaw=CAM_YAW, pitch=CAM_PITCH,
        roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=WIDTH / HEIGHT, nearVal=0.1, farVal=100)
    _, _, rgba, _, _ = p.getCameraImage(
        WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER)
    img = Image.frombytes("RGBA", (WIDTH, HEIGHT), bytes(rgba))
    return img.convert("RGB")


def save_frame(img, frame_dir, frame_num):
    img.save(os.path.join(frame_dir, f"frame_{frame_num:06d}.png"))


def simulate_and_capture(solution, frame_dir, frame_counter, overlay_label):
    """Run simulation, capture frames, return (frame_counter, fitness)."""
    phys_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    p.loadSDF("world.sdf")
    robot_id = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robot_id)

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
        for nname in sensors:
            nn.neurons[nname].Set_Value(sensors[nname].Get_Value())
        for nname in nn.neurons:
            if nn.neurons[nname].Is_Motor_Neuron():
                nn.neurons[nname].Set_Value(0.0)
                for sname in nn.synapses:
                    if nn.synapses[sname].Get_Target_Neuron_Name() == nname:
                        src = nn.synapses[sname].Get_Source_Neuron_Name()
                        nn.neurons[nname].Add_To_Value(
                            nn.neurons[src].Get_Value() * nn.synapses[sname].Get_Weight())
                nn.neurons[nname].Threshold()
        for nname in motors:
            angle = nn.neurons[nname].Get_Value() * c.motorJointRange
            motors[nname].Set_Value(robot_id, angle)

        if t % CAPTURE_EVERY == 0:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            fitness_val = pos[0]
            frame = render_pybullet_frame(robot_id)
            frame = make_sim_overlay(frame, overlay_label, solution.weights,
                                     fitness_val, captured, total_captures)
            save_frame(frame, frame_dir, frame_counter)
            frame_counter += 1
            captured += 1

    pos, _ = p.getBasePositionAndOrientation(robot_id)
    fitness_val = pos[0]
    p.disconnect()
    return frame_counter, fitness_val


def run_evolution(initial_solution):
    """Run HC evolution headless, return (final_parent, gen_data)."""
    parent = copy.deepcopy(initial_solution)
    parent.Set_ID(0)

    # Evaluate parent
    parent.Create_World()
    parent.Create_Body()
    parent.Create_Brain()
    os.system("python3 simulate.py DIRECT 0 2>/dev/null")
    with open("fitness0.txt", "r") as f:
        parent.fitness = float(f.read())

    gen_data = []
    best_so_far = parent.fitness

    for g in range(NUM_GENERATIONS):
        child = copy.deepcopy(parent)
        child.Set_ID(1)
        child.Mutate()
        child.Create_World()
        child.Create_Body()
        child.Create_Brain()
        os.system("python3 simulate.py DIRECT 1 2>/dev/null")
        with open("fitness1.txt", "r") as f:
            child.fitness = float(f.read())

        improved = child.fitness < parent.fitness
        if improved:
            parent = child
            parent.Set_ID(0)
            best_so_far = parent.fitness

        gen_data.append({
            "gen": g + 1,
            "parent_fitness": parent.fitness if not improved else gen_data[-1]["best_so_far"] if gen_data else initial_solution.fitness_initial,
            "child_fitness": child.fitness,
            "improved": improved,
            "best_so_far": best_so_far,
        })
        print(f"  Gen {g+1:2d}: parent {parent.fitness:+.4f} child {child.fitness:+.4f} {'IMPROVED' if improved else 'rejected'}")

    # Fix parent_fitness in gen_data (need to track properly)
    # Re-derive from evolution trace
    current_parent_fit = initial_solution.fitness_initial
    for gd in gen_data:
        gd["parent_fitness"] = current_parent_fit
        if gd["improved"]:
            current_parent_fit = gd["child_fitness"]

    return parent, gen_data


def main():
    frame_dir = tempfile.mkdtemp(prefix="hill_climber_video_")
    print(f"Frames dir: {frame_dir}")
    frame_counter = 0

    # --- Title card ---
    print("Rendering title card...")
    title = make_title_card()
    for _ in range(TITLE_SECONDS * FPS):
        save_frame(title, frame_dir, frame_counter)
        frame_counter += 1

    # --- Create initial solution ---
    initial = SOLUTION(0)
    initial.Create_World()
    initial.Create_Body()
    initial.Create_Brain()
    print(f"Initial weights:\n{initial.weights}")

    # --- Phase 1: Initial simulation ---
    print("\n=== Phase 1: Initial Simulation ===")
    phase1 = make_phase_card("Phase 1: Initial Random Robot",
                             "Before any evolution")
    for _ in range(2 * FPS):
        save_frame(phase1, frame_dir, frame_counter)
        frame_counter += 1

    wcard = make_weight_card("Initial Random Weights", initial.weights)
    for _ in range(CAPTION_SECONDS * FPS):
        save_frame(wcard, frame_dir, frame_counter)
        frame_counter += 1

    print("Simulating initial robot...")
    frame_counter, initial_fitness = simulate_and_capture(
        initial, frame_dir, frame_counter, "Initial Random Robot")
    initial.fitness_initial = initial_fitness
    print(f"Initial fitness: {initial_fitness:+.4f}")

    # --- Phase 2: Evolution ---
    print("\n=== Phase 2: Hill Climber Evolution ===")
    phase2 = make_phase_card("Phase 2: Hill Climber Evolution",
                             f"{NUM_GENERATIONS} generations of mutation and selection")
    for _ in range(2 * FPS):
        save_frame(phase2, frame_dir, frame_counter)
        frame_counter += 1

    print("Running evolution...")
    evolved_parent, gen_data = run_evolution(initial)

    # Animate the evolution log building up
    frames_per_gen = int(1.5 * FPS)  # 1.5 seconds per line appearing
    for g in range(1, len(gen_data) + 1):
        evo_card = make_evolution_card(gen_data, g)
        for _ in range(frames_per_gen):
            save_frame(evo_card, frame_dir, frame_counter)
            frame_counter += 1

    # Hold final evolution state
    final_evo = make_evolution_card(gen_data, len(gen_data))
    for _ in range(2 * FPS):
        save_frame(final_evo, frame_dir, frame_counter)
        frame_counter += 1

    # --- Phase 3: Final evolved simulation ---
    print("\n=== Phase 3: Final Evolved Simulation ===")
    phase3 = make_phase_card("Phase 3: Evolved Robot",
                             f"After {NUM_GENERATIONS} generations of hill climbing")
    for _ in range(2 * FPS):
        save_frame(phase3, frame_dir, frame_counter)
        frame_counter += 1

    evolved_parent.Create_World()
    evolved_parent.Create_Body()
    evolved_parent.Create_Brain()

    wcard2 = make_weight_card(f"Evolved Weights (Gen {NUM_GENERATIONS})",
                               evolved_parent.weights, evolved_parent.fitness)
    for _ in range(CAPTION_SECONDS * FPS):
        save_frame(wcard2, frame_dir, frame_counter)
        frame_counter += 1

    print("Simulating evolved robot...")
    frame_counter, final_fitness = simulate_and_capture(
        evolved_parent, frame_dir, frame_counter,
        f"Evolved Robot (Gen {NUM_GENERATIONS})")
    print(f"Final fitness: {final_fitness:+.4f}")

    # --- Comparison card ---
    print("\nRendering comparison card...")
    comparison = make_comparison_card(
        initial_fitness, final_fitness,
        initial.weights, evolved_parent.weights)
    for _ in range(TITLE_SECONDS * FPS):
        save_frame(comparison, frame_dir, frame_counter)
        frame_counter += 1

    print(f"\nTotal frames: {frame_counter}")

    # --- Encode ---
    output_file = "hill_climber_video.mp4"
    ffmpeg_bin = os.path.expanduser("~/miniforge3/bin/ffmpeg")
    print(f"Encoding {output_file}...")
    cmd = (f'{ffmpeg_bin} -y -framerate {FPS} -i "{frame_dir}/frame_%06d.png" '
           f'-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p '
           f'-vf "scale={WIDTH}:{HEIGHT}" "{output_file}"')
    ret = os.system(cmd)

    if ret == 0:
        shutil.rmtree(frame_dir)
    else:
        print(f"WARNING: ffmpeg failed. Frames kept at {frame_dir}")

    # Cleanup sim files
    for fname in os.listdir("."):
        if fname.startswith("brain") and fname.endswith(".nndf"):
            os.remove(fname)
        if fname.startswith("fitness") and fname.endswith(".txt"):
            os.remove(fname)

    print(f"\nDone! Video saved to {output_file}")
    print(f"Initial fitness: {initial_fitness:+.4f}")
    print(f"Final fitness:   {final_fitness:+.4f}")
    print(f"Improvement:     {initial_fitness - final_fitness:+.4f}")


if __name__ == "__main__":
    main()
