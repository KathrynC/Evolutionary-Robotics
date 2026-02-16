"""
Record Module M: The Parallel Hill Climber video
Population of hill climbers evolving in parallel
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
POP_SIZE = c.populationSize       # 10
NUM_GENERATIONS = 10

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
BAR_BLUE = (60, 140, 200)
BAR_GREEN = (60, 200, 120)
BEST_GOLD = (240, 200, 80)

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
    draw.rectangle([100, 200, WIDTH - 100, 203], fill=ACCENT)
    draw.text((WIDTH // 2, 260), "M. The Parallel Hill Climber",
              fill=TEXT_WHITE, font=FONT_TITLE, anchor="mm")
    draw.text((WIDTH // 2, 330),
              f"Population of {POP_SIZE} robots evolving independently in parallel",
              fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")
    draw.rectangle([100, 380, WIDTH - 100, 383], fill=ACCENT)
    draw.text((WIDTH // 2, 440), "Kathryn Cramer",
              fill=TEXT_WHITE, font=FONT_NAME, anchor="mm")
    draw.text((WIDTH // 2, 490), "University of Vermont",
              fill=TEXT_DIM, font=FONT_SUBTITLE, anchor="mm")
    draw.text((WIDTH // 2, 560), "Evolutionary Robotics  \u2022  r/ludobots",
              fill=(120, 120, 130), font=FONT_CAPTION, anchor="mm")
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


def draw_fitness_bars(draw, fitnesses, y_top, bar_height, label, highlight_best=True):
    """Draw horizontal bar chart of population fitnesses."""
    draw.text((WIDTH // 2, y_top - 25), label,
              fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")

    if not fitnesses:
        return

    min_f = min(fitnesses)
    max_f = max(fitnesses)
    f_range = max(abs(min_f), abs(max_f), 0.1)
    best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])

    bar_left = 180
    bar_right = WIDTH - 80
    bar_width = bar_right - bar_left
    zero_x = bar_left + bar_width // 2

    for i, fit in enumerate(fitnesses):
        y = y_top + i * (bar_height + 4)

        # Label
        draw.text((bar_left - 10, y + bar_height // 2),
                  f"P{i+1:2d}", fill=TEXT_DIM, font=FONT_SMALL, anchor="rm")

        # Bar background
        draw.rectangle([bar_left, y, bar_right, y + bar_height],
                       fill=(30, 30, 42))

        # Zero line
        draw.line([zero_x, y, zero_x, y + bar_height], fill=(60, 60, 70), width=1)

        # Bar
        px_per_unit = (bar_width // 2) / f_range if f_range > 0 else 1
        bar_end = zero_x + int(fit * px_per_unit)
        bar_end = max(bar_left, min(bar_right, bar_end))

        color = BAR_GREEN if fit < 0 else BAR_BLUE
        if highlight_best and i == best_idx:
            color = BEST_GOLD

        if fit < 0:
            draw.rectangle([bar_end, y + 1, zero_x, y + bar_height - 1], fill=color)
        else:
            draw.rectangle([zero_x, y + 1, bar_end, y + bar_height - 1], fill=color)

        # Value
        fit_color = WEIGHT_POS if fit < 0 else WEIGHT_NEG
        if highlight_best and i == best_idx:
            fit_color = BEST_GOLD
        draw.text((bar_right + 10, y + bar_height // 2),
                  f"{fit:+.2f}", fill=fit_color, font=FONT_SMALL, anchor="lm")


def make_population_card(fitnesses, title_text, sub_text=""):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)
    draw.text((WIDTH // 2, 50), title_text,
              fill=TEXT_WHITE, font=FONT_NAME, anchor="mm")
    if sub_text:
        draw.text((WIDTH // 2, 85), sub_text,
                  fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")
    draw.rectangle([100, 110, WIDTH - 100, 113], fill=ACCENT)

    bar_height = 22
    draw_fitness_bars(draw, fitnesses, 140, bar_height, "Population Fitness")

    best = min(fitnesses)
    avg = sum(fitnesses) / len(fitnesses)
    draw.rectangle([100, HEIGHT - 80, WIDTH - 100, HEIGHT - 77], fill=(60, 60, 70))
    draw.text((WIDTH // 2, HEIGHT - 50),
              f"Best: {best:+.4f}   Average: {avg:+.4f}   Spread: {max(fitnesses) - min(fitnesses):.4f}",
              fill=ACCENT, font=FONT_CAPTION, anchor="mm")
    return img


def make_evolution_summary(gen_history, up_to_gen):
    """Show evolution as a generation-by-generation summary with bar charts."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2, 35), "Parallel Hill Climber Evolution",
              fill=TEXT_WHITE, font=FONT_NAME, anchor="mm")
    draw.rectangle([100, 60, WIDTH - 100, 63], fill=ACCENT)

    # Table: Gen | Best | Avg | Worst | Improvements
    headers = ["Gen", "Best", "Average", "Worst", "Improved"]
    col_xs = [100, 220, 420, 620, 820]

    y = 85
    for i, (h, x) in enumerate(zip(headers, col_xs)):
        draw.text((x, y), h, fill=ACCENT, font=FONT_DATA_BOLD, anchor="lm")

    # Show initial row
    row_height = 30
    y_start = 120

    if up_to_gen >= 0 and len(gen_history) > 0:
        # Initial population row
        init = gen_history[0]["pre_fitnesses"]
        y_row = y_start
        draw.text((col_xs[0], y_row), "Init", fill=TEXT_DIM, font=FONT_DATA, anchor="lm")
        best_init = min(init)
        draw.text((col_xs[1], y_row), f"{best_init:+.4f}",
                  fill=WEIGHT_POS if best_init < 0 else WEIGHT_NEG,
                  font=FONT_DATA, anchor="lm")
        avg_init = sum(init) / len(init)
        draw.text((col_xs[2], y_row), f"{avg_init:+.4f}",
                  fill=TEXT_DIM, font=FONT_DATA, anchor="lm")
        worst_init = max(init)
        draw.text((col_xs[3], y_row), f"{worst_init:+.4f}",
                  fill=TEXT_DIM, font=FONT_DATA, anchor="lm")

    for g_idx in range(min(up_to_gen, len(gen_history))):
        g = gen_history[g_idx]
        y_row = y_start + (g_idx + 1) * row_height
        if y_row > HEIGHT - 120:
            break

        gen_num = g["gen"]
        best = g["best"]
        avg = g["avg"]
        worst = g["worst"]
        n_improved = g["n_improved"]

        is_latest = (g_idx == up_to_gen - 1)
        base_color = TEXT_WHITE if is_latest else TEXT_DIM

        draw.text((col_xs[0], y_row), f"{gen_num:3d}",
                  fill=base_color, font=FONT_DATA, anchor="lm")

        best_color = BEST_GOLD if is_latest else (WEIGHT_POS if best < 0 else WEIGHT_NEG)
        draw.text((col_xs[1], y_row), f"{best:+.4f}",
                  fill=best_color, font=FONT_DATA_BOLD if is_latest else FONT_DATA, anchor="lm")

        draw.text((col_xs[2], y_row), f"{avg:+.4f}",
                  fill=base_color, font=FONT_DATA, anchor="lm")
        draw.text((col_xs[3], y_row), f"{worst:+.4f}",
                  fill=base_color, font=FONT_DATA, anchor="lm")

        imp_color = IMPROVED_COLOR if n_improved > 0 else REJECTED_COLOR
        draw.text((col_xs[4], y_row), f"{n_improved}/{POP_SIZE}",
                  fill=imp_color, font=FONT_DATA, anchor="lm")

    # Bottom bar
    if up_to_gen > 0 and up_to_gen <= len(gen_history):
        g = gen_history[up_to_gen - 1]
        draw.rectangle([0, HEIGHT - 60, WIDTH, HEIGHT], fill=(25, 25, 38))
        total_imps = sum(h["n_improved"] for h in gen_history[:up_to_gen])
        draw.text((WIDTH // 2, HEIGHT - 30),
                  f"Generation {g['gen']}/{NUM_GENERATIONS}  |  "
                  f"Total improvements: {total_imps}  |  "
                  f"Best: {g['best']:+.4f}",
                  fill=ACCENT, font=FONT_CAPTION, anchor="mm")

    return img


def make_sim_overlay(frame_img, label, weights, fitness_val, step, total_steps):
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

    bar_top = HEIGHT - 80
    draw.rectangle([0, bar_top, WIDTH, HEIGHT], fill=(18, 18, 28, 200))

    pairs = [("T\u2192B", weights[0][0]), ("T\u2192F", weights[0][1]),
             ("B\u2192B", weights[1][0]), ("B\u2192F", weights[1][1]),
             ("F\u2192B", weights[2][0]), ("F\u2192F", weights[2][1])]
    x_start = 30
    for wlabel, w in pairs:
        color = WEIGHT_POS if w >= 0 else WEIGHT_NEG
        draw.text((x_start, bar_top + 20), wlabel, fill=TEXT_DIM, font=FONT_WEIGHT, anchor="lm")
        draw.text((x_start, bar_top + 48), f"{w:+.3f}", fill=color, font=FONT_WEIGHT, anchor="lm")
        x_start += 130

    if fitness_val is not None:
        fit_color = WEIGHT_POS if fitness_val < 0 else WEIGHT_NEG
        draw.text((WIDTH - 30, bar_top + 35), f"x = {fitness_val:+.2f}",
                  fill=fit_color, font=FONT_NAME, anchor="rm")

    frame_rgba = frame_img.convert("RGBA")
    return Image.alpha_composite(frame_rgba, overlay).convert("RGB")


def make_results_card(initial_best, initial_avg, final_best, final_avg, final_fitnesses):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2, 50), "Parallel Hill Climber Results",
              fill=TEXT_WHITE, font=FONT_TITLE, anchor="mm")
    draw.rectangle([100, 95, WIDTH - 100, 98], fill=ACCENT)

    left_x = WIDTH // 4
    right_x = 3 * WIDTH // 4

    draw.text((left_x, 130), "Initial Population", fill=TEXT_DIM, font=FONT_NAME, anchor="mm")
    draw.text((right_x, 130), f"After {NUM_GENERATIONS} Generations", fill=ACCENT, font=FONT_NAME, anchor="mm")

    draw.text((left_x, 195), f"{initial_best:+.2f}", fill=WEIGHT_NEG, font=FONT_BIG_NUM, anchor="mm")
    draw.text((right_x, 195), f"{final_best:+.2f}",
              fill=WEIGHT_POS if final_best < initial_best else WEIGHT_NEG,
              font=FONT_BIG_NUM, anchor="mm")

    draw.text((left_x, 240), "best", fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")
    draw.text((right_x, 240), "best", fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")

    draw.text((WIDTH // 2, 195), "\u2192", fill=ACCENT, font=FONT_BIG_NUM, anchor="mm")

    improvement = initial_best - final_best
    draw.text((WIDTH // 2, 290),
              f"Improvement: {improvement:+.2f} ({abs(improvement):.1f} units further left)",
              fill=IMPROVED_COLOR, font=FONT_NAME, anchor="mm")

    draw.text((left_x, 330), f"avg: {initial_avg:+.2f}", fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")
    draw.text((right_x, 330), f"avg: {final_avg:+.2f}", fill=TEXT_DIM, font=FONT_CAPTION, anchor="mm")

    draw.rectangle([100, 360, WIDTH - 100, 363], fill=(60, 60, 70))

    # Final population bars
    bar_height = 18
    draw_fitness_bars(draw, final_fitnesses, 390, bar_height, "Final Population")

    draw.text((WIDTH // 2, HEIGHT - 40),
              "Fitness = torso x-position  \u2022  more negative = better  \u2022  gold = best",
              fill=(100, 100, 110), font=FONT_CAPTION, anchor="mm")
    return img


def render_pybullet_frame(robot_id):
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[pos[0], pos[1], 0.5],
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
            frame = make_sim_overlay(frame, overlay_label, solution.weights,
                                     fitness_val, captured, total_captures)
            save_frame(frame, frame_dir, frame_counter)
            frame_counter += 1
            captured += 1

    pos, _ = p.getBasePositionAndOrientation(robot_id)
    fitness_val = pos[0]
    p.disconnect()
    return frame_counter, fitness_val


def evaluate_solution(solution):
    """Evaluate a solution headless, return fitness."""
    solution.Create_World()
    solution.Create_Body()
    solution.Create_Brain()
    os.system(f"python3 simulate.py DIRECT {solution.myID} 2>/dev/null")
    with open(f"fitness{solution.myID}.txt", "r") as f:
        solution.fitness = float(f.read())
    return solution.fitness


def main():
    frame_dir = tempfile.mkdtemp(prefix="parallel_hc_video_")
    print(f"Frames dir: {frame_dir}")
    frame_counter = 0
    next_id = 0

    # --- Title card ---
    print("Rendering title card...")
    title = make_title_card()
    for _ in range(TITLE_SECONDS * FPS):
        save_frame(title, frame_dir, frame_counter)
        frame_counter += 1

    # --- Create initial population ---
    print("\nCreating initial population...")
    parents = {}
    for i in range(POP_SIZE):
        parents[i] = SOLUTION(next_id)
        next_id += 1

    # Evaluate initial population
    print("Evaluating initial population...")
    for i in parents:
        evaluate_solution(parents[i])
        print(f"  Parent {i}: fitness = {parents[i].fitness:+.4f}")

    initial_fitnesses = [parents[i].fitness for i in range(POP_SIZE)]
    initial_best = min(initial_fitnesses)
    initial_avg = sum(initial_fitnesses) / len(initial_fitnesses)

    # --- Phase 1: Initial population ---
    print("\n=== Phase 1: Initial Population ===")
    phase1 = make_phase_card("Phase 1: Initial Population",
                             f"{POP_SIZE} random robots evaluated")
    for _ in range(2 * FPS):
        save_frame(phase1, frame_dir, frame_counter)
        frame_counter += 1

    pop_card = make_population_card(initial_fitnesses,
                                     "Initial Population (Random Weights)",
                                     f"{POP_SIZE} independently generated robots")
    for _ in range(3 * FPS):
        save_frame(pop_card, frame_dir, frame_counter)
        frame_counter += 1

    # Simulate the initial best robot
    best_idx = min(range(POP_SIZE), key=lambda i: initial_fitnesses[i])
    best_parent = parents[best_idx]
    best_parent.Create_World()
    best_parent.Create_Body()
    best_parent.Create_Brain()
    print(f"Simulating initial best (Parent {best_idx}, fitness {best_parent.fitness:+.4f})...")
    frame_counter, _ = simulate_and_capture(
        best_parent, frame_dir, frame_counter,
        f"Initial Best: Parent {best_idx + 1} (fitness {best_parent.fitness:+.2f})")

    # --- Phase 2: Evolution ---
    print("\n=== Phase 2: Parallel Evolution ===")
    phase2 = make_phase_card("Phase 2: Parallel Evolution",
                             f"{POP_SIZE} hill climbers \u00d7 {NUM_GENERATIONS} generations")
    for _ in range(2 * FPS):
        save_frame(phase2, frame_dir, frame_counter)
        frame_counter += 1

    gen_history = []
    for g in range(NUM_GENERATIONS):
        pre_fitnesses = [parents[i].fitness for i in range(POP_SIZE)]

        # Spawn + mutate + evaluate children
        children = {}
        n_improved = 0
        for i in range(POP_SIZE):
            children[i] = copy.deepcopy(parents[i])
            children[i].Set_ID(next_id)
            next_id += 1
            children[i].Mutate()
            evaluate_solution(children[i])

        # Select
        for i in range(POP_SIZE):
            if children[i].fitness < parents[i].fitness:
                parents[i] = children[i]
                n_improved += 1

        post_fitnesses = [parents[i].fitness for i in range(POP_SIZE)]
        gen_history.append({
            "gen": g + 1,
            "pre_fitnesses": pre_fitnesses,
            "best": min(post_fitnesses),
            "avg": sum(post_fitnesses) / len(post_fitnesses),
            "worst": max(post_fitnesses),
            "n_improved": n_improved,
        })
        print(f"  Gen {g+1:2d}: best={min(post_fitnesses):+.4f} avg={sum(post_fitnesses)/len(post_fitnesses):+.4f} improved={n_improved}/{POP_SIZE}")

    # Animate evolution summary
    frames_per_gen = int(1.5 * FPS)
    for g in range(1, len(gen_history) + 1):
        evo_card = make_evolution_summary(gen_history, g)
        for _ in range(frames_per_gen):
            save_frame(evo_card, frame_dir, frame_counter)
            frame_counter += 1

    # Hold final
    final_evo = make_evolution_summary(gen_history, len(gen_history))
    for _ in range(2 * FPS):
        save_frame(final_evo, frame_dir, frame_counter)
        frame_counter += 1

    # --- Phase 3: Final best ---
    print("\n=== Phase 3: Evolved Best Robot ===")
    final_fitnesses = [parents[i].fitness for i in range(POP_SIZE)]
    final_best = min(final_fitnesses)
    final_avg = sum(final_fitnesses) / len(final_fitnesses)
    final_best_idx = min(range(POP_SIZE), key=lambda i: final_fitnesses[i])

    phase3 = make_phase_card("Phase 3: Best Evolved Robot",
                             f"Champion from population of {POP_SIZE}")
    for _ in range(2 * FPS):
        save_frame(phase3, frame_dir, frame_counter)
        frame_counter += 1

    # Show final population
    pop_card2 = make_population_card(final_fitnesses,
                                      f"Final Population (Gen {NUM_GENERATIONS})",
                                      f"Gold = champion (Parent {final_best_idx + 1})")
    for _ in range(3 * FPS):
        save_frame(pop_card2, frame_dir, frame_counter)
        frame_counter += 1

    # Simulate champion
    champion = parents[final_best_idx]
    champion.Create_World()
    champion.Create_Body()
    champion.Create_Brain()
    print(f"Simulating champion (Parent {final_best_idx}, fitness {champion.fitness:+.4f})...")
    frame_counter, champion_fitness = simulate_and_capture(
        champion, frame_dir, frame_counter,
        f"Champion: Parent {final_best_idx + 1} (fitness {champion.fitness:+.2f})")

    # --- Results card ---
    print("\nRendering results card...")
    results = make_results_card(initial_best, initial_avg, final_best, final_avg, final_fitnesses)
    for _ in range(TITLE_SECONDS * FPS):
        save_frame(results, frame_dir, frame_counter)
        frame_counter += 1

    print(f"\nTotal frames: {frame_counter}")

    # --- Encode ---
    output_file = "parallel_hc_video.mp4"
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

    # Cleanup
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
