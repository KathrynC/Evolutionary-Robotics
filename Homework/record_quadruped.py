"""
Record Module N: The Quadruped video
9-link quadruped evolved with parallel hill climber
"""
import os
import sys
import copy
import shutil
import tempfile
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont

# We need to use the quadruped branch's constants, so override before importing
# This script should be run from the quadruped branch
import constants as c
import pyrosim.pyrosim as pyrosim
from solution import SOLUTION

# --- Override for video (fewer gens for reasonable time) ---
EVOLVE_GENERATIONS = 100

# --- Video settings ---
WIDTH, HEIGHT = 1280, 720
FPS = 60
SIM_STEPS = c.numTimeSteps  # 1000
CAPTURE_EVERY = 2
TITLE_SECONDS = 4
CAPTION_SECONDS = 3
POP_SIZE = c.populationSize  # 10

# --- Camera for quadruped ---
CAM_DISTANCE = 6.0
CAM_YAW = 45
CAM_PITCH = -25

# --- Colors ---
BG_DARK = (18, 18, 28)
ACCENT = (80, 180, 220)
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (160, 160, 170)
WEIGHT_POS = (100, 210, 130)
WEIGHT_NEG = (210, 100, 110)
IMPROVED_COLOR = (80, 220, 120)
BEST_GOLD = (240, 200, 80)
LEG_COLORS = [(200, 120, 80), (80, 180, 200), (180, 80, 200), (80, 200, 120)]

# --- Fonts ---
def load_font(size, bold=False):
    paths = ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"]
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
FONT_SMALL = load_font(16)


def make_title_card():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 180, WIDTH - 100, 183], fill=ACCENT)
    draw.text((WIDTH // 2, 240), "N. The Quadruped",
              fill=TEXT_WHITE, font=FONT_TITLE, anchor="mm")
    draw.text((WIDTH // 2, 310),
              "A 4-legged robot with 9 links and 8 joints",
              fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")
    draw.text((WIDTH // 2, 355),
              f"Evolved with parallel hill climber ({POP_SIZE} pop \u00d7 {EVOLVE_GENERATIONS} gen)",
              fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")
    draw.rectangle([100, 400, WIDTH - 100, 403], fill=ACCENT)
    draw.text((WIDTH // 2, 450), "Kathryn Cramer",
              fill=TEXT_WHITE, font=FONT_NAME, anchor="mm")
    draw.text((WIDTH // 2, 500), "University of Vermont",
              fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")
    draw.text((WIDTH // 2, 570), "Evolutionary Robotics  \u2022  r/ludobots",
              fill=(120, 120, 130), font=FONT_CAPTION, anchor="mm")
    return img


def make_body_diagram():
    """Show the quadruped body structure."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2, 50), "Quadruped Body Plan",
              fill=TEXT_WHITE, font=FONT_NAME, anchor="mm")
    draw.rectangle([100, 80, WIDTH - 100, 83], fill=ACCENT)

    # Body diagram - schematic top-down view
    cx, cy = WIDTH // 2, 300
    torso_w, torso_h = 120, 60

    # Torso
    draw.rectangle([cx - torso_w//2, cy - torso_h//2, cx + torso_w//2, cy + torso_h//2],
                   outline=TEXT_WHITE, width=2)
    draw.text((cx, cy), "Torso", fill=TEXT_WHITE, font=FONT_SMALL, anchor="mm")

    # Legs
    leg_data = [
        ("Front", 0, (cx + torso_w//2 + 10, cy), (cx + torso_w//2 + 80, cy)),
        ("Back", 1, (cx - torso_w//2 - 10, cy), (cx - torso_w//2 - 80, cy)),
        ("Left", 2, (cx, cy - torso_h//2 - 10), (cx, cy - torso_h//2 - 80)),
        ("Right", 3, (cx, cy + torso_h//2 + 10), (cx, cy + torso_h//2 + 80)),
    ]

    for name, idx, start, end in leg_data:
        color = LEG_COLORS[idx]
        # Upper leg
        draw.line([start, end], fill=color, width=4)
        # Lower leg (extension)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        lower_end = (end[0] + dx, end[1] + dy)
        draw.line([end, lower_end], fill=color, width=3)
        # Foot sensor circle
        draw.ellipse([lower_end[0] - 8, lower_end[1] - 8, lower_end[0] + 8, lower_end[1] + 8],
                     fill=color)
        # Joint dots
        draw.ellipse([start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4], fill=TEXT_WHITE)
        draw.ellipse([end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4], fill=TEXT_WHITE)

    # Legend
    y_legend = 470
    draw.text((WIDTH // 2, y_legend - 20), "Body Structure",
              fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")

    specs = [
        "9 links: 1 torso + 4 upper legs + 4 lower legs",
        "8 revolute joints: all rotating around x-axis (1,0,0)",
        "4 touch sensors: on lower leg feet (filled circles)",
        "8 motor neurons: one per joint",
        "32 synapses: fully connected (4 sensors \u00d7 8 motors)",
        f"Motor range: \u00b1{c.motorJointRange} radians  \u2022  Joint axis: (0,1,0)",
    ]
    for i, spec in enumerate(specs):
        draw.text((WIDTH // 2, y_legend + 15 + i * 28), spec,
                  fill=TEXT_DIM, font=FONT_SMALL, anchor="mm")

    return img


def make_phase_card(phase_text, sub_text=""):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)
    draw.text((WIDTH // 2, HEIGHT // 2 - 30), phase_text,
              fill=ACCENT, font=FONT_TITLE, anchor="mm")
    if sub_text:
        draw.text((WIDTH // 2, HEIGHT // 2 + 40), sub_text,
                  fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")
    return img


def make_weight_card(label, weights):
    """Compact weight display for 4x8 matrix."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)
    draw.text((WIDTH // 2, 50), label,
              fill=ACCENT, font=FONT_NAME, anchor="mm")
    draw.rectangle([100, 80, WIDTH - 100, 83], fill=(60, 60, 70))

    draw.text((WIDTH // 2, 110), "4\u00d78 Synapse Weight Matrix",
              fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")

    motor_labels = ["F\u2191", "F\u2193", "B\u2191", "B\u2193", "L\u2191", "L\u2193", "R\u2191", "R\u2193"]
    sensor_labels = ["Front foot", "Back foot", "Left foot", "Right foot"]

    cell_w = 100
    cell_h = 50
    table_x = (WIDTH - 8 * cell_w) // 2 + 80
    table_y = 160

    # Motor headers
    for j, ml in enumerate(motor_labels):
        x = table_x + j * cell_w + cell_w // 2
        draw.text((x, table_y - 15), ml, fill=ACCENT, font=FONT_SMALL, anchor="mm")

    # Rows
    for i in range(c.numSensorNeurons):
        y = table_y + i * cell_h
        draw.text((table_x - 10, y + cell_h // 2), sensor_labels[i],
                  fill=TEXT_DIM, font=FONT_SMALL, anchor="rm")
        for j in range(c.numMotorNeurons):
            x = table_x + j * cell_w + cell_w // 2
            w = weights[i][j]
            # Color intensity based on weight magnitude
            intensity = min(abs(w), 1.0)
            if w >= 0:
                color = (int(60 + 140 * intensity), int(180 + 30 * intensity), int(100 + 30 * intensity))
            else:
                color = (int(180 + 30 * intensity), int(60 + 40 * intensity), int(60 + 50 * intensity))
            draw.text((x, y + cell_h // 2), f"{w:+.2f}",
                      fill=color, font=FONT_SMALL, anchor="mm")

    return img


def make_evolution_graph(gen_bests, gen_avgs, up_to_gen):
    """Line chart of best and average fitness over generations."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2, 35), "Evolution Progress",
              fill=TEXT_WHITE, font=FONT_NAME, anchor="mm")
    draw.rectangle([100, 60, WIDTH - 100, 63], fill=ACCENT)

    # Chart area
    chart_left = 140
    chart_right = WIDTH - 60
    chart_top = 100
    chart_bottom = HEIGHT - 120
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top

    # Axes
    draw.rectangle([chart_left, chart_top, chart_right, chart_bottom],
                   outline=(60, 60, 70), width=1)

    n = min(up_to_gen, len(gen_bests))
    if n < 2:
        return img

    all_vals = gen_bests[:n] + gen_avgs[:n]
    y_min = min(all_vals) - 0.5
    y_max = max(all_vals) + 0.5
    y_range = y_max - y_min if y_max != y_min else 1

    def to_px(gen, val):
        px_x = chart_left + int((gen / max(EVOLVE_GENERATIONS, 1)) * chart_w)
        px_y = chart_bottom - int(((val - y_min) / y_range) * chart_h)
        return (px_x, px_y)

    # Zero line if visible
    if y_min < 0 < y_max:
        zero_y = chart_bottom - int(((0 - y_min) / y_range) * chart_h)
        draw.line([chart_left, zero_y, chart_right, zero_y], fill=(50, 50, 60), width=1)
        draw.text((chart_left - 5, zero_y), "0", fill=TEXT_DIM, font=FONT_SMALL, anchor="rm")

    # Draw average line
    avg_points = [to_px(i, gen_avgs[i]) for i in range(n)]
    for i in range(len(avg_points) - 1):
        draw.line([avg_points[i], avg_points[i + 1]], fill=(100, 100, 140), width=2)

    # Draw best line
    best_points = [to_px(i, gen_bests[i]) for i in range(n)]
    for i in range(len(best_points) - 1):
        draw.line([best_points[i], best_points[i + 1]], fill=BEST_GOLD, width=2)

    # Current point
    if n > 0:
        bx, by = best_points[-1]
        draw.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=BEST_GOLD)

    # Labels
    draw.text((chart_left + 10, chart_top + 10), "\u2014 Best",
              fill=BEST_GOLD, font=FONT_SMALL, anchor="lm")
    draw.text((chart_left + 10, chart_top + 30), "\u2014 Average",
              fill=(100, 100, 140), font=FONT_SMALL, anchor="lm")

    # Axis labels
    draw.text((WIDTH // 2, chart_bottom + 30), "Generation",
              fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")
    # Y-axis tick labels
    for tick_val in [y_min, (y_min + y_max) / 2, y_max]:
        tick_y = chart_bottom - int(((tick_val - y_min) / y_range) * chart_h)
        draw.text((chart_left - 8, tick_y), f"{tick_val:+.1f}",
                  fill=TEXT_DIM, font=FONT_SMALL, anchor="rm")

    # Bottom stats
    draw.rectangle([0, HEIGHT - 60, WIDTH, HEIGHT], fill=(25, 25, 38))
    draw.text((WIDTH // 2, HEIGHT - 30),
              f"Generation {n}/{EVOLVE_GENERATIONS}  |  "
              f"Best: {gen_bests[n-1]:+.4f}  |  Avg: {gen_avgs[n-1]:+.4f}",
              fill=ACCENT, font=FONT_CAPTION, anchor="mm")

    return img


def make_sim_overlay_quad(frame_img, label, fitness_val, step, total_steps):
    """Simpler overlay for quadruped (no weight details — too many)."""
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    draw.rectangle([0, 0, WIDTH, 45], fill=(18, 18, 28, 180))
    draw.text((20, 22), label, fill=ACCENT, font=FONT_CAPTION, anchor="lm")

    progress = step / total_steps
    bar_x, bar_y, bar_w, bar_h = WIDTH - 220, 12, 200, 20
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                   outline=(80, 80, 90), width=1)
    draw.rectangle([bar_x + 1, bar_y + 1,
                    bar_x + 1 + int((bar_w - 2) * progress), bar_y + bar_h - 1],
                   fill=ACCENT)

    bar_top = HEIGHT - 50
    draw.rectangle([0, bar_top, WIDTH, HEIGHT], fill=(18, 18, 28, 200))
    if fitness_val is not None:
        fit_color = WEIGHT_POS if fitness_val < 0 else WEIGHT_NEG
        draw.text((WIDTH // 2, bar_top + 25), f"x-position: {fitness_val:+.3f}",
                  fill=fit_color, font=FONT_NAME, anchor="mm")

    frame_rgba = frame_img.convert("RGBA")
    return Image.alpha_composite(frame_rgba, overlay).convert("RGB")


def make_results_card(initial_best, final_best, final_fitnesses):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2, 60), "Quadruped Evolution Results",
              fill=TEXT_WHITE, font=FONT_TITLE, anchor="mm")
    draw.rectangle([100, 105, WIDTH - 100, 108], fill=ACCENT)

    left_x = WIDTH // 4
    right_x = 3 * WIDTH // 4

    draw.text((left_x, 150), "Initial Best", fill=TEXT_DIM, font=FONT_NAME, anchor="mm")
    draw.text((right_x, 150), f"Evolved (Gen {EVOLVE_GENERATIONS})", fill=ACCENT, font=FONT_NAME, anchor="mm")

    draw.text((left_x, 220), f"{initial_best:+.2f}", fill=WEIGHT_NEG, font=FONT_BIG_NUM, anchor="mm")
    draw.text((right_x, 220), f"{final_best:+.2f}",
              fill=WEIGHT_POS if final_best < initial_best else WEIGHT_NEG,
              font=FONT_BIG_NUM, anchor="mm")
    draw.text((WIDTH // 2, 220), "\u2192", fill=ACCENT, font=FONT_BIG_NUM, anchor="mm")

    improvement = initial_best - final_best
    draw.text((WIDTH // 2, 300),
              f"Improvement: {improvement:+.2f} ({abs(improvement):.1f} units further left)",
              fill=IMPROVED_COLOR, font=FONT_NAME, anchor="mm")

    draw.rectangle([100, 340, WIDTH - 100, 343], fill=(60, 60, 70))

    # Body stats
    specs = [
        "9 links  \u2022  8 joints  \u2022  4 sensors  \u2022  8 motors  \u2022  32 synapses",
        f"Population: {POP_SIZE}  \u2022  Generations: {EVOLVE_GENERATIONS}  \u2022  Motor range: \u00b1{c.motorJointRange}",
        f"Total simulations: ~{POP_SIZE * EVOLVE_GENERATIONS + POP_SIZE}",
    ]
    for i, s in enumerate(specs):
        draw.text((WIDTH // 2, 380 + i * 35), s,
                  fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")

    # Final population
    draw.text((WIDTH // 2, 500), "Final Population Fitnesses:",
              fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")
    best_idx = min(range(len(final_fitnesses)), key=lambda i: final_fitnesses[i])
    row_text = "  ".join(
        f"{'[' if i == best_idx else ''}{final_fitnesses[i]:+.2f}{']*' if i == best_idx else ''}"
        for i in range(len(final_fitnesses)))
    draw.text((WIDTH // 2, 535), row_text,
              fill=TEXT_WHITE, font=FONT_DATA, anchor="mm")

    draw.text((WIDTH // 2, HEIGHT - 40),
              "Fitness = torso x-position  \u2022  more negative = better  \u2022  * = champion",
              fill=(100, 100, 110), font=FONT_CAPTION, anchor="mm")
    return img


def render_pybullet_frame(robot_id):
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[pos[0], pos[1], 1.0],
        distance=CAM_DISTANCE, yaw=CAM_YAW, pitch=CAM_PITCH,
        roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=WIDTH / HEIGHT, nearVal=0.1, farVal=100)
    _, _, rgba, _, _ = p.getCameraImage(WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj,
                                         renderer=p.ER_TINY_RENDERER)
    return Image.frombytes("RGBA", (WIDTH, HEIGHT), bytes(rgba)).convert("RGB")


def save_frame(img, frame_dir, frame_num):
    img.save(os.path.join(frame_dir, f"frame_{frame_num:06d}.png"))


def simulate_and_capture(solution, frame_dir, frame_counter, overlay_label):
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
    sensors = {n: SENSOR(nn.neurons[n].Get_Link_Name())
               for n in nn.neurons if nn.neurons[n].Is_Sensor_Neuron()}
    motors = {n: MOTOR(nn.neurons[n].Get_Joint_Name())
              for n in nn.neurons if nn.neurons[n].Is_Motor_Neuron()}

    fitness_val = None
    captured = 0
    total_captures = SIM_STEPS // CAPTURE_EVERY

    for t in range(SIM_STEPS):
        p.stepSimulation()
        for n in sensors:
            nn.neurons[n].Set_Value(sensors[n].Get_Value())
        for n in nn.neurons:
            if nn.neurons[n].Is_Motor_Neuron():
                nn.neurons[n].Set_Value(0.0)
                for s in nn.synapses:
                    if nn.synapses[s].Get_Target_Neuron_Name() == n:
                        src = nn.synapses[s].Get_Source_Neuron_Name()
                        nn.neurons[n].Add_To_Value(
                            nn.neurons[src].Get_Value() * nn.synapses[s].Get_Weight())
                nn.neurons[n].Threshold()
        for n in motors:
            motors[n].Set_Value(robot_id, nn.neurons[n].Get_Value() * c.motorJointRange)

        if t % CAPTURE_EVERY == 0:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            fitness_val = pos[0]
            frame = render_pybullet_frame(robot_id)
            frame = make_sim_overlay_quad(frame, overlay_label, fitness_val, captured, total_captures)
            save_frame(frame, frame_dir, frame_counter)
            frame_counter += 1
            captured += 1

    pos, _ = p.getBasePositionAndOrientation(robot_id)
    fitness_val = pos[0]
    p.disconnect()
    return frame_counter, fitness_val


def evaluate_solution(solution):
    solution.Create_World()
    solution.Create_Body()
    solution.Create_Brain()
    os.system(f"python3 simulate.py DIRECT {solution.myID} 2>/dev/null")
    with open(f"fitness{solution.myID}.txt", "r") as f:
        solution.fitness = float(f.read())
    return solution.fitness


def main():
    frame_dir = tempfile.mkdtemp(prefix="quadruped_video_")
    print(f"Frames dir: {frame_dir}")
    frame_counter = 0
    next_id = 0

    # --- Title card ---
    print("Rendering title card...")
    for _ in range(TITLE_SECONDS * FPS):
        save_frame(make_title_card(), frame_dir, frame_counter)
        frame_counter += 1

    # --- Body diagram ---
    print("Rendering body diagram...")
    for _ in range(CAPTION_SECONDS * FPS):
        save_frame(make_body_diagram(), frame_dir, frame_counter)
        frame_counter += 1

    # --- Create + evaluate initial population ---
    print("\nCreating initial population...")
    parents = {}
    for i in range(POP_SIZE):
        parents[i] = SOLUTION(next_id)
        next_id += 1

    print("Evaluating initial population...")
    for i in parents:
        evaluate_solution(parents[i])
        print(f"  Parent {i}: {parents[i].fitness:+.4f}")

    initial_fitnesses = [parents[i].fitness for i in range(POP_SIZE)]
    initial_best = min(initial_fitnesses)

    # --- Brief initial best simulation ---
    print("\n=== Initial Best Simulation ===")
    phase1 = make_phase_card("Initial Quadruped (Random Weights)",
                             f"Best of {POP_SIZE} random robots")
    for _ in range(2 * FPS):
        save_frame(phase1, frame_dir, frame_counter)
        frame_counter += 1

    best_init_idx = min(range(POP_SIZE), key=lambda i: initial_fitnesses[i])
    init_best = parents[best_init_idx]
    init_best.Create_World(); init_best.Create_Body(); init_best.Create_Brain()

    wcard = make_weight_card("Initial Best Weights", init_best.weights)
    for _ in range(2 * FPS):
        save_frame(wcard, frame_dir, frame_counter)
        frame_counter += 1

    print(f"Simulating initial best (fitness {init_best.fitness:+.4f})...")
    frame_counter, _ = simulate_and_capture(
        init_best, frame_dir, frame_counter,
        f"Initial Best (fitness {init_best.fitness:+.2f})")

    # --- Evolution ---
    print(f"\n=== Evolving ({EVOLVE_GENERATIONS} generations) ===")
    phase2 = make_phase_card("Evolution in Progress",
                             f"{POP_SIZE} parallel hill climbers \u00d7 {EVOLVE_GENERATIONS} generations")
    for _ in range(2 * FPS):
        save_frame(phase2, frame_dir, frame_counter)
        frame_counter += 1

    gen_bests = [initial_best]
    gen_avgs = [sum(initial_fitnesses) / len(initial_fitnesses)]

    for g in range(EVOLVE_GENERATIONS):
        children = {}
        for i in range(POP_SIZE):
            children[i] = copy.deepcopy(parents[i])
            children[i].Set_ID(next_id)
            next_id += 1
            children[i].Mutate()
            evaluate_solution(children[i])

        n_improved = 0
        for i in range(POP_SIZE):
            if children[i].fitness < parents[i].fitness:
                parents[i] = children[i]
                n_improved += 1

        fitnesses = [parents[i].fitness for i in range(POP_SIZE)]
        gen_bests.append(min(fitnesses))
        gen_avgs.append(sum(fitnesses) / len(fitnesses))

        if (g + 1) % 10 == 0 or g == 0:
            print(f"  Gen {g+1:3d}: best={min(fitnesses):+.4f} avg={sum(fitnesses)/len(fitnesses):+.4f} improved={n_improved}/{POP_SIZE}")

    # Animate evolution graph — show in chunks
    graph_gens = list(range(0, EVOLVE_GENERATIONS + 1, 5))  # every 5th gen
    if EVOLVE_GENERATIONS not in graph_gens:
        graph_gens.append(EVOLVE_GENERATIONS)

    frames_per_step = FPS // 2  # 0.5 sec per step
    for g in graph_gens:
        chart = make_evolution_graph(gen_bests, gen_avgs, g + 1)
        for _ in range(frames_per_step):
            save_frame(chart, frame_dir, frame_counter)
            frame_counter += 1

    # Hold final graph
    final_chart = make_evolution_graph(gen_bests, gen_avgs, EVOLVE_GENERATIONS + 1)
    for _ in range(3 * FPS):
        save_frame(final_chart, frame_dir, frame_counter)
        frame_counter += 1

    # --- Final best simulation ---
    print("\n=== Final Evolved Quadruped ===")
    final_fitnesses = [parents[i].fitness for i in range(POP_SIZE)]
    final_best = min(final_fitnesses)
    final_best_idx = min(range(POP_SIZE), key=lambda i: final_fitnesses[i])

    phase3 = make_phase_card("Evolved Quadruped",
                             f"Champion after {EVOLVE_GENERATIONS} generations")
    for _ in range(2 * FPS):
        save_frame(phase3, frame_dir, frame_counter)
        frame_counter += 1

    champion = parents[final_best_idx]
    champion.Create_World(); champion.Create_Body(); champion.Create_Brain()

    wcard2 = make_weight_card(f"Champion Weights (Gen {EVOLVE_GENERATIONS})", champion.weights)
    for _ in range(2 * FPS):
        save_frame(wcard2, frame_dir, frame_counter)
        frame_counter += 1

    print(f"Simulating champion (fitness {champion.fitness:+.4f})...")
    frame_counter, champion_fitness = simulate_and_capture(
        champion, frame_dir, frame_counter,
        f"Evolved Champion (fitness {champion.fitness:+.2f})")

    # --- Results ---
    print("\nRendering results card...")
    results = make_results_card(initial_best, final_best, final_fitnesses)
    for _ in range(TITLE_SECONDS * FPS):
        save_frame(results, frame_dir, frame_counter)
        frame_counter += 1

    print(f"\nTotal frames: {frame_counter}")

    # --- Encode ---
    output_file = "quadruped_video.mp4"
    ffmpeg_bin = os.path.expanduser("~/miniforge3/bin/ffmpeg")
    print(f"Encoding {output_file}...")
    cmd = (f'{ffmpeg_bin} -y -framerate {FPS} -i "{frame_dir}/frame_%06d.png" '
           f'-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p '
           f'-vf "scale={WIDTH}:{HEIGHT}" "{output_file}"')
    ret = os.system(cmd)
    if ret == 0:
        shutil.rmtree(frame_dir)
    else:
        print(f"WARNING: ffmpeg failed. Frames at {frame_dir}")

    for fname in os.listdir("."):
        if fname.startswith("brain") and fname.endswith(".nndf"):
            os.remove(fname)
        if fname.startswith("fitness") and fname.endswith(".txt"):
            os.remove(fname)

    print(f"\nDone! Video saved to {output_file}")
    print(f"Initial best: {initial_best:+.4f}  Final best: {final_best:+.4f}")
    print(f"Improvement: {initial_best - final_best:+.4f}")


if __name__ == "__main__":
    main()
