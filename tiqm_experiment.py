#!/usr/bin/env python3
"""
tiqm_experiment.py

Transactional Interpretation of LLM-Mediated Locomotion.

Implements Cramer's TIQM as a framework for semantic-to-physical translation:
  - OFFER WAVE: LLM generates weights + physics parameters from a semantic seed
  - MEDIUM: PyBullet simulates with alternate physics
  - CONFIRMATION WAVE: A second LLM observes the behavior and rates semantic match
  - TRANSACTION: Completes when offer and confirmation resonate

Protocols implemented:
  1. Extended Offer (Phase 1): LLM chooses 6 weights + 6 physics params = 12D channel
  2. Single-Round TIQM (Phase 2): Offer + one confirmation + resonance score
  3. Iterative TIQM (Phase 2+): Multiple offer-confirmation rounds until convergence

Usage:
    # Phase 1: Extended offer only (no confirmation wave)
    python tiqm_experiment.py --phase 1 --seeds romeo_juliet

    # Phase 2: Full single-round TIQM
    python tiqm_experiment.py --phase 2 --seeds romeo_juliet

    # Phase 2+: Iterative TIQM (up to 5 rounds)
    python tiqm_experiment.py --phase 2 --iterative --max-rounds 5 --seeds romeo_juliet

    # Multi-offer (K=4 competing transactions)
    python tiqm_experiment.py --phase 2 --multi-offer 4 --seeds romeo_juliet
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pybullet as p
import pybullet_data

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from structured_random_common import (
    parse_weights, write_brain, WEIGHT_NAMES, OLLAMA_URL
)
from compute_beer_analytics import compute_all, DT, NumpyEncoder

ARTIFACTS = PROJECT / "artifacts"

# ── Models ───────────────────────────────────────────────────────────────────
# Offer wave: the model that generates weights + physics
# Confirmation wave: a DIFFERENT model that evaluates the behavior
# Using different models prevents self-confirmation bias.

OFFER_MODELS = [
    "qwen3-coder:30b",
    "deepseek-r1:8b",
    "llama3.1:latest",
    "gpt-oss:20b",
]

CONFIRMATION_MODEL = "llama3.1:latest"  # lightweight, fast for scoring

REASONING_MODELS = {"deepseek-r1:8b", "gpt-oss:20b"}

# ── Physics parameter space ──────────────────────────────────────────────────

# Defaults (our standard physics)
DEFAULT_PHYSICS = {
    "gravity": 9.81,         # magnitude (always downward)
    "friction": 2.5,         # lateral friction
    "restitution": 0.0,      # bounciness
    "damping": 0.0,          # joint damping
    "max_force": 150.0,      # motor strength (N)
    "mass_ratio": 1.0,       # torso mass multiplier
}

# Allowed ranges for LLM-chosen physics
PHYSICS_RANGES = {
    "gravity":     (0.0, 20.0),
    "friction":    (0.1, 5.0),
    "restitution": (0.0, 1.0),
    "damping":     (0.0, 2.0),
    "max_force":   (50.0, 300.0),
    "mass_ratio":  (0.3, 3.0),
}

PHYSICS_GRID = {
    "gravity":     [0.0, 2.0, 5.0, 9.81, 15.0, 20.0],
    "friction":    [0.1, 0.5, 1.0, 2.5, 5.0],
    "restitution": [0.0, 0.25, 0.5, 0.75, 1.0],
    "damping":     [0.0, 0.5, 1.0, 2.0],
    "max_force":   [50.0, 100.0, 150.0, 200.0, 300.0],
    "mass_ratio":  [0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
}

WEIGHT_GRID = [-1.0, -0.7, -0.4, -0.1, 0.0, 0.1, 0.4, 0.7, 1.0]


def snap_weight(v):
    return min(WEIGHT_GRID, key=lambda g: abs(g - v))


def snap_physics(key, v):
    lo, hi = PHYSICS_RANGES[key]
    v = max(lo, min(hi, v))
    return min(PHYSICS_GRID[key], key=lambda g: abs(g - v))


def parse_physics(response):
    """Parse physics parameters from LLM response."""
    text = response.strip()
    # Strip think tags
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # Strip code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None, None
    try:
        raw = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None, None

    # Parse weights
    weights = {}
    for wn in WEIGHT_NAMES:
        if wn not in raw:
            return None, None
        weights[wn] = snap_weight(float(raw[wn]))

    # Parse physics (optional — use defaults if missing)
    physics = dict(DEFAULT_PHYSICS)
    for pk in PHYSICS_RANGES:
        if pk in raw:
            try:
                physics[pk] = snap_physics(pk, float(raw[pk]))
            except (ValueError, TypeError):
                pass

    return weights, physics


# ── LLM interface ────────────────────────────────────────────────────────────

def query_ollama(model, prompt, temperature=0.8, max_tokens=800, timeout=120):
    effective_max = 2000 if model in REASONING_MODELS else max_tokens
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": effective_max}
    })
    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=timeout
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        if "error" in data:
            return None
        return data["response"]
    except Exception:
        return None


# ── Offer wave prompts ───────────────────────────────────────────────────────

OFFER_PROMPT_CHARACTER = """\
You are designing BOTH a neural controller AND the physics of the world for a \
3-link walking robot (Torso, BackLeg, FrontLeg) with two hinge joints.

NEURAL CONTROLLER (6 weights):
  Sensors: s0=torso_contact, s1=back_foot_contact, s2=front_foot_contact
  Motors: m3=back_joint, m4=front_joint
  Control: m3 = tanh(w03*s0 + w13*s1 + w23*s2), m4 = tanh(w04*s0 + w14*s1 + w24*s2)
  Weight roles:
    w03, w04: torso touch → motors (balance response)
    w13, w24: foot → same leg (local reflex)
    w14, w23: foot → opposite leg (cross-coupling)

PHYSICS PARAMETERS (6 params):
  gravity: strength of gravity (0=weightless, 9.81=Earth, 20=heavy planet)
  friction: ground friction (0.1=ice, 2.5=normal, 5.0=tar)
  restitution: bounciness (0=dead, 1=superball)
  damping: joint resistance (0=free, 2=moving through honey)
  max_force: muscle strength in Newtons (50=weak, 150=normal, 300=powerful)
  mass_ratio: torso heaviness multiplier (0.3=light, 1=normal, 3=heavy)

CHARACTER: {character} from {story}
{extra_context}

Think about this character's physical presence, their energy, their fate in the \
story. What kind of WORLD do they inhabit? What kind of BODY do they have?

A doomed character might live in a world with crushing gravity. A comedic character \
might bounce (high restitution). A graceful character might have low friction and \
high muscle strength. A heavy, imposing character might have high mass_ratio.

Choose weights from [-1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0].
Choose physics values from the ranges above.

Output a single JSON object with ALL 12 keys:
{{"w03":_, "w04":_, "w13":_, "w14":_, "w23":_, "w24":_, \
"gravity":_, "friction":_, "restitution":_, "damping":_, "max_force":_, "mass_ratio":_}}"""

OFFER_PROMPT_SEQUENCE = """\
You are designing BOTH a neural controller AND the physics of the world for a \
3-link walking robot (Torso, BackLeg, FrontLeg) with two hinge joints.

NEURAL CONTROLLER (6 weights):
  Sensors: s0=torso_contact, s1=back_foot_contact, s2=front_foot_contact
  Motors: m3=back_joint, m4=front_joint
  Control: m3 = tanh(w03*s0 + w13*s1 + w23*s2), m4 = tanh(w04*s0 + w14*s1 + w24*s2)
  Weight roles:
    w03, w04: torso touch → motors (balance response)
    w13, w24: foot → same leg (local reflex)
    w14, w23: foot → opposite leg (cross-coupling)

PHYSICS PARAMETERS (6 params):
  gravity: strength of gravity (0=weightless, 9.81=Earth, 20=heavy planet)
  friction: ground friction (0.1=ice, 2.5=normal, 5.0=tar)
  restitution: bounciness (0=dead, 1=superball)
  damping: joint resistance (0=free, 2=moving through honey)
  max_force: muscle strength in Newtons (50=weak, 150=normal, 300=powerful)
  mass_ratio: torso heaviness multiplier (0.3=light, 1=normal, 3=heavy)

SEQUENCE: {seq_id} — {seq_name}
First terms: {terms}

Think about this sequence's mathematical character — its growth, rhythm, regularity \
or chaos. What kind of WORLD does this sequence describe? What physics does it imply?

An exponentially growing sequence might need a world with escalating forces. A \
periodic sequence might live in a stable, rhythmic world. A chaotic sequence might \
have extreme physics. The zero sequence might have zero gravity.

Choose weights from [-1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0].
Choose physics values from the ranges above.

Output a single JSON object with ALL 12 keys:
{{"w03":_, "w04":_, "w13":_, "w14":_, "w23":_, "w24":_, \
"gravity":_, "friction":_, "restitution":_, "damping":_, "max_force":_, "mass_ratio":_}}"""

OFFER_PROMPT_GENERIC = """\
You are designing BOTH a neural controller AND the physics of the world for a \
3-link walking robot (Torso, BackLeg, FrontLeg) with two hinge joints.

NEURAL CONTROLLER (6 weights):
  Sensors: s0=torso_contact, s1=back_foot_contact, s2=front_foot_contact
  Motors: m3=back_joint, m4=front_joint
  Control: m3 = tanh(w03*s0 + w13*s1 + w23*s2), m4 = tanh(w04*s0 + w14*s1 + w24*s2)

PHYSICS PARAMETERS (6 params):
  gravity (0-20), friction (0.1-5), restitution (0-1), damping (0-2),
  max_force (50-300 N), mass_ratio (0.3-3)

SEED: {seed}
{extra_context}

What kind of world does "{seed}" inhabit? What movement does it produce?

Choose weights from [-1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0].

Output a single JSON object with ALL 12 keys:
{{"w03":_, "w04":_, "w13":_, "w14":_, "w23":_, "w24":_, \
"gravity":_, "friction":_, "restitution":_, "damping":_, "max_force":_, "mass_ratio":_}}"""


# ── Confirmation wave ────────────────────────────────────────────────────────

CONFIRMATION_PROMPT = """\
A robot was simulated with these results:

  Displacement: DX={dx:+.2f}m forward, DY={dy:+.2f}m sideways
  Speed: mean={speed:.2f} m/s, max={max_speed:.2f} m/s
  Torso on ground: {torso_duty:.0%} of the time (>30% suggests falling/crawling)
  Contact entropy: {entropy:.2f} bits (higher = more varied foot patterns)
  Phase lock: {phase_lock:.2f} (1.0 = perfectly coordinated legs)
  Net yaw: {yaw:.1f} degrees (0 = straight, ±180 = turned around)
  Dominant frequency: {freq:.2f} Hz (leg cycling speed)

  Physics of this world:
    Gravity: {gravity} m/s² (Earth=9.81)
    Friction: {friction} (ice=0.1, normal=2.5, tar=5.0)
    Bounciness: {restitution} (dead=0, superball=1)
    Joint damping: {damping} (free=0, honey=2)
    Muscle strength: {max_force}N (weak=50, normal=150, powerful=300)
    Body heaviness: {mass_ratio}x (light=0.3, normal=1, heavy=3)

The semantic seed for this simulation was: "{seed}"

Questions:
1. Describe the movement you see in 1-2 sentences (as if watching a creature move).
2. Does this movement MATCH the seed "{seed}"? Rate the semantic resonance from 0.0 \
(no connection) to 1.0 (perfect embodiment).
3. Does the PHYSICS chosen match the seed? Rate physics resonance 0.0 to 1.0.
4. What would you change to better embody "{seed}"?

Output a JSON object:
{{"movement_description": "...", "resonance_behavior": 0.X, "resonance_physics": 0.X, \
"suggestion": "..."}}"""


def parse_confirmation(response):
    """Parse confirmation wave response."""
    text = response.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if "{" in part:
                text = part
                break
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


# ── Simulation with alternate physics ────────────────────────────────────────

def run_trial_alternate_physics(weights, physics_params):
    """Run simulation with custom physics parameters.

    Like run_trial_inmemory but applies the LLM-chosen physics before simulating.
    """
    write_brain(weights)

    grav = physics_params.get("gravity", 9.81)
    fric = physics_params.get("friction", 2.5)
    rest = physics_params.get("restitution", 0.0)
    damp = physics_params.get("damping", 0.0)
    mf = physics_params.get("max_force", 150.0)
    mr = physics_params.get("mass_ratio", 1.0)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -grav)  # always downward, variable magnitude
    p.setTimeStep(c.DT)

    planeId = p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    # Apply physics to ground plane
    p.changeDynamics(planeId, -1, lateralFriction=fric, restitution=rest)

    # Apply physics to all robot links
    for link in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, link,
                         lateralFriction=fric,
                         restitution=rest,
                         jointDamping=damp)

    # Apply mass ratio to torso (link -1 = base)
    if mr != 1.0:
        dyn = p.getDynamicsInfo(robotId, -1)
        base_mass = dyn[0]
        p.changeDynamics(robotId, -1, mass=base_mass * mr)

    nn = NEURAL_NETWORK("brain.nndf")
    n_steps = c.SIM_STEPS

    # Pre-allocate telemetry arrays
    t_arr = np.empty(n_steps)
    x = np.empty(n_steps); y = np.empty(n_steps); z = np.empty(n_steps)
    vx = np.empty(n_steps); vy = np.empty(n_steps); vz = np.empty(n_steps)
    wx = np.empty(n_steps); wy = np.empty(n_steps); wz = np.empty(n_steps)
    roll_a = np.empty(n_steps); pitch_a = np.empty(n_steps); yaw_a = np.empty(n_steps)
    contact_torso = np.empty(n_steps, dtype=bool)
    contact_back = np.empty(n_steps, dtype=bool)
    contact_front = np.empty(n_steps, dtype=bool)
    j0_pos = np.empty(n_steps); j0_vel = np.empty(n_steps); j0_tau = np.empty(n_steps)
    j1_pos = np.empty(n_steps); j1_vel = np.empty(n_steps); j1_tau = np.empty(n_steps)

    link_names = ["Torso", "BackLeg", "FrontLeg"]
    link_indices = {}
    joint_indices = {}
    for i_j in range(p.getNumJoints(robotId)):
        info = p.getJointInfo(robotId, i_j)
        jname = info[1].decode("utf-8") if isinstance(info[1], bytes) else info[1]
        lname = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
        if lname in link_names:
            link_indices[lname] = i_j
        joint_indices[jname] = i_j

    back_link_idx = link_indices.get("BackLeg", -1)
    front_link_idx = link_indices.get("FrontLeg", -1)
    j0_idx = joint_indices.get("Torso_BackLeg", 0)
    j1_idx = joint_indices.get("Torso_FrontLeg", 1)

    for i in range(n_steps):
        for neuronName in nn.neurons:
            n_obj = nn.neurons[neuronName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, n_obj.Get_Value(), mf)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, p.POSITION_CONTROL,
                                                n_obj.Get_Value(), mf)
        p.stepSimulation()
        nn.Update()

        t_arr[i] = i * c.DT
        pos, orn = p.getBasePositionAndOrientation(robotId)
        vel_lin, vel_ang = p.getBaseVelocity(robotId)
        rpy_vals = p.getEulerFromQuaternion(orn)

        x[i] = pos[0]; y[i] = pos[1]; z[i] = pos[2]
        vx[i] = vel_lin[0]; vy[i] = vel_lin[1]; vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]; wy[i] = vel_ang[1]; wz[i] = vel_ang[2]
        roll_a[i] = rpy_vals[0]; pitch_a[i] = rpy_vals[1]; yaw_a[i] = rpy_vals[2]

        contact_pts = p.getContactPoints(robotId)
        tc = False; bc = False; fc = False
        for cp in contact_pts:
            li = cp[3]
            if li == -1: tc = True
            elif li == back_link_idx: bc = True
            elif li == front_link_idx: fc = True
        contact_torso[i] = tc
        contact_back[i] = bc
        contact_front[i] = fc

        js0 = p.getJointState(robotId, j0_idx)
        js1 = p.getJointState(robotId, j1_idx)
        j0_pos[i] = js0[0]; j0_vel[i] = js0[1]; j0_tau[i] = js0[3]
        j1_pos[i] = js1[0]; j1_vel[i] = js1[1]; j1_tau[i] = js1[3]

    p.disconnect()

    data = {
        "t": t_arr, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll_a, "pitch": pitch_a, "yaw": yaw_a,
        "contact_torso": contact_torso,
        "contact_back": contact_back,
        "contact_front": contact_front,
        "j0_pos": j0_pos, "j0_vel": j0_vel, "j0_tau": j0_tau,
        "j1_pos": j1_pos, "j1_vel": j1_vel, "j1_tau": j1_tau,
    }
    return compute_all(data, DT)


# ── Seed sets ────────────────────────────────────────────────────────────────

ROMEO_JULIET_SEEDS = [
    ("Juliet", "Romeo and Juliet"),
    ("Romeo", "Romeo and Juliet"),
    ("Mercutio", "Romeo and Juliet"),
    ("Tybalt", "Romeo and Juliet"),
    ("Friar Laurence", "Romeo and Juliet"),
    ("Nurse", "Romeo and Juliet"),
    ("Benvolio", "Romeo and Juliet"),
    ("Paris", "Romeo and Juliet"),
    ("Lady Capulet", "Romeo and Juliet"),
    ("Lord Capulet", "Romeo and Juliet"),
]

DEATH_TEST_SEEDS = [
    # Characters who die
    ("Hamlet", "Hamlet"),
    ("Ophelia", "Hamlet"),
    ("Juliet", "Romeo and Juliet"),
    ("Gatsby", "The Great Gatsby"),
    ("Boromir", "The Lord of the Rings"),
    ("Dumbledore", "Harry Potter"),
    # Characters who survive
    ("Friar Laurence", "Romeo and Juliet"),
    ("Horatio", "Hamlet"),
    ("Nick Carraway", "The Great Gatsby"),
    ("Samwise Gamgee", "The Lord of the Rings"),
    ("Harry Potter", "Harry Potter"),
    ("Hermione Granger", "Harry Potter"),
]


def get_seed_set(name):
    if name == "romeo_juliet":
        return ROMEO_JULIET_SEEDS, "character"
    elif name == "death_test":
        return DEATH_TEST_SEEDS, "character"
    else:
        return ROMEO_JULIET_SEEDS, "character"


# ── Main experiment loop ─────────────────────────────────────────────────────

def run_tiqm_experiment(seed_set_name="romeo_juliet", phase=2,
                        iterative=False, max_rounds=3,
                        multi_offer=1, models=None,
                        temperature=0.8):

    seeds, seed_type = get_seed_set(seed_set_name)
    if models is None:
        models = OFFER_MODELS

    n_trials = len(seeds) * len(models) * multi_offer
    print(f"TIQM Experiment: Phase {phase}, {len(seeds)} seeds × "
          f"{len(models)} models × {multi_offer} offers = {n_trials} trials")
    if iterative:
        print(f"  Iterative mode: up to {max_rounds} rounds per trial")
    print()

    results = []
    t0 = time.time()
    trial_num = 0

    for seed_data in seeds:
        if seed_type == "character":
            character, story = seed_data
            seed_label = f"{character} ({story})"
        else:
            seed_label = str(seed_data)

        for model in models:
            for offer_idx in range(multi_offer):
                trial_num += 1
                prefix = f"[{trial_num}/{n_trials}] {model[:15]} | {seed_label[:40]}"

                # ── OFFER WAVE ────────────────────────────────────
                if seed_type == "character":
                    prompt = OFFER_PROMPT_CHARACTER.format(
                        character=character, story=story, extra_context="")
                else:
                    prompt = OFFER_PROMPT_GENERIC.format(
                        seed=seed_label, extra_context="")

                resp = query_ollama(model, prompt, temperature=temperature)
                if resp is None:
                    print(f"  {prefix} -> OFFER FAIL (LLM error)")
                    results.append({
                        "seed": seed_label, "model": model, "offer_idx": offer_idx,
                        "status": "offer_fail", "phase": phase,
                    })
                    continue

                weights, physics = parse_physics(resp)
                if weights is None:
                    print(f"  {prefix} -> OFFER FAIL (parse)")
                    results.append({
                        "seed": seed_label, "model": model, "offer_idx": offer_idx,
                        "status": "parse_fail", "phase": phase,
                    })
                    continue

                # ── MEDIUM (simulation) ───────────────────────────
                try:
                    analytics = run_trial_alternate_physics(weights, physics)
                except Exception as e:
                    print(f"  {prefix} -> SIM FAIL ({e})")
                    results.append({
                        "seed": seed_label, "model": model, "offer_idx": offer_idx,
                        "status": "sim_fail", "weights": weights, "physics": physics,
                        "phase": phase,
                    })
                    continue

                dx = analytics["outcome"]["dx"]
                dy = analytics["outcome"]["dy"]
                spd = analytics["outcome"]["mean_speed"]
                max_spd = spd * (1 + analytics["outcome"].get("speed_cv", 0))
                torso_duty = analytics["contact"]["duty_torso"]
                entropy = analytics["contact"]["contact_entropy_bits"]
                phase_lock = analytics["coordination"]["phase_lock_score"]
                yaw_net = np.degrees(analytics["outcome"]["yaw_net_rad"])
                freq = analytics["coordination"]["joint_0"]["dominant_freq_hz"]

                result_entry = {
                    "seed": seed_label, "model": model, "offer_idx": offer_idx,
                    "status": "ok", "phase": phase,
                    "weights": weights, "physics": physics,
                    "dx": dx, "dy": dy, "speed": spd,
                    "torso_duty": torso_duty, "entropy": entropy,
                    "phase_lock": phase_lock,
                    "analytics": analytics,
                }

                line = (f"  {prefix} -> DX={dx:+.2f} DY={dy:+.2f} spd={spd:.2f} "
                        f"| g={physics['gravity']} f={physics['friction']} "
                        f"r={physics['restitution']} mf={physics['max_force']}")

                # ── CONFIRMATION WAVE (Phase 2+) ──────────────────
                if phase >= 2:
                    conf_prompt = CONFIRMATION_PROMPT.format(
                        dx=dx, dy=dy, speed=spd, max_speed=max_spd,
                        torso_duty=torso_duty, entropy=entropy,
                        phase_lock=phase_lock, yaw=yaw_net, freq=freq,
                        gravity=physics["gravity"], friction=physics["friction"],
                        restitution=physics["restitution"], damping=physics["damping"],
                        max_force=physics["max_force"], mass_ratio=physics["mass_ratio"],
                        seed=seed_label,
                    )

                    conf_resp = query_ollama(CONFIRMATION_MODEL, conf_prompt,
                                             temperature=0.3, max_tokens=500)
                    if conf_resp:
                        conf = parse_confirmation(conf_resp)
                        if conf:
                            result_entry["confirmation"] = conf
                            rb = conf.get("resonance_behavior", 0)
                            rp = conf.get("resonance_physics", 0)
                            desc = conf.get("movement_description", "")[:60]
                            result_entry["resonance_behavior"] = rb
                            result_entry["resonance_physics"] = rp
                            result_entry["resonance_combined"] = (rb + rp) / 2
                            line += f" | res={rb:.2f}/{rp:.2f} \"{desc}\""
                        else:
                            line += " | CONF PARSE FAIL"
                    else:
                        line += " | CONF LLM FAIL"

                    # ── ITERATIVE TIQM ────────────────────────────
                    if iterative and conf and max_rounds > 1:
                        for round_n in range(1, max_rounds):
                            suggestion = conf.get("suggestion", "")
                            if not suggestion:
                                break
                            # Build iterative prompt
                            iter_prompt = (
                                f"Previous attempt for \"{seed_label}\":\n"
                                f"  Weights: {weights}\n"
                                f"  Physics: {physics}\n"
                                f"  Result: DX={dx:+.2f}, speed={spd:.2f}, "
                                f"torso_duty={torso_duty:.2f}\n"
                                f"  Observer feedback: \"{suggestion}\"\n\n"
                                f"Adjust your weights and physics to better embody "
                                f"\"{seed_label}\".\n\n"
                            )
                            if seed_type == "character":
                                iter_prompt += OFFER_PROMPT_CHARACTER.format(
                                    character=character, story=story,
                                    extra_context=f"\nObserver says: {suggestion}")
                            else:
                                iter_prompt += OFFER_PROMPT_GENERIC.format(
                                    seed=seed_label,
                                    extra_context=f"\nObserver says: {suggestion}")

                            resp2 = query_ollama(model, iter_prompt,
                                                 temperature=temperature)
                            if not resp2:
                                break
                            w2, ph2 = parse_physics(resp2)
                            if not w2:
                                break

                            try:
                                a2 = run_trial_alternate_physics(w2, ph2)
                            except Exception:
                                break

                            dx2 = a2["outcome"]["dx"]
                            spd2 = a2["outcome"]["mean_speed"]
                            td2 = a2["contact"]["duty_torso"]

                            # New confirmation
                            conf_prompt2 = CONFIRMATION_PROMPT.format(
                                dx=dx2,
                                dy=a2["outcome"]["dy"],
                                speed=spd2,
                                max_speed=spd2 * (1 + a2["outcome"].get("speed_cv", 0)),
                                torso_duty=td2,
                                entropy=a2["contact"]["contact_entropy_bits"],
                                phase_lock=a2["coordination"]["phase_lock_score"],
                                yaw=np.degrees(a2["outcome"]["yaw_net_rad"]),
                                freq=a2["coordination"]["joint_0"]["dominant_freq_hz"],
                                gravity=ph2["gravity"], friction=ph2["friction"],
                                restitution=ph2["restitution"], damping=ph2["damping"],
                                max_force=ph2["max_force"], mass_ratio=ph2["mass_ratio"],
                                seed=seed_label,
                            )
                            conf_resp2 = query_ollama(CONFIRMATION_MODEL, conf_prompt2,
                                                      temperature=0.3, max_tokens=500)
                            if not conf_resp2:
                                break
                            conf2 = parse_confirmation(conf_resp2)
                            if not conf2:
                                break

                            rb2 = conf2.get("resonance_behavior", 0)
                            rp2 = conf2.get("resonance_physics", 0)

                            result_entry.setdefault("iterations", []).append({
                                "round": round_n,
                                "weights": w2, "physics": ph2,
                                "dx": dx2, "speed": spd2, "torso_duty": td2,
                                "resonance_behavior": rb2,
                                "resonance_physics": rp2,
                                "suggestion": conf2.get("suggestion", ""),
                            })

                            # Update if improved
                            if (rb2 + rp2) / 2 > result_entry.get("resonance_combined", 0):
                                result_entry["resonance_behavior"] = rb2
                                result_entry["resonance_physics"] = rp2
                                result_entry["resonance_combined"] = (rb2 + rp2) / 2
                                result_entry["best_round"] = round_n
                                weights, physics = w2, ph2
                                dx, spd, torso_duty = dx2, spd2, td2
                                conf = conf2

                            # Check convergence
                            prev_r = result_entry["iterations"][-1]["resonance_behavior"] if len(result_entry.get("iterations", [])) > 1 else rb
                            if abs(rb2 - prev_r) < 0.05:
                                break  # Converged

                        n_rounds = len(result_entry.get("iterations", [])) + 1
                        result_entry["total_rounds"] = n_rounds

                print(line)
                results.append(result_entry)

                # Checkpoint
                if trial_num % 20 == 0:
                    elapsed = time.time() - t0
                    rate = trial_num / elapsed
                    remaining = (n_trials - trial_num) / rate if rate > 0 else 0
                    print(f"  [checkpoint] {trial_num}/{n_trials}, "
                          f"{elapsed:.0f}s, ~{remaining/60:.0f}min remaining")

    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("TIQM EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Trials: {len(results)}, Time: {elapsed:.0f}s")

    ok = [r for r in results if r["status"] == "ok"]
    fails = [r for r in results if r["status"] != "ok"]
    print(f"Successful: {len(ok)}, Failed: {len(fails)}")

    if ok and phase >= 2:
        resonances = [r.get("resonance_combined", 0) for r in ok if "resonance_combined" in r]
        if resonances:
            print(f"Resonance: mean={np.mean(resonances):.2f}, "
                  f"min={min(resonances):.2f}, max={max(resonances):.2f}")

    # Physics diversity
    if ok:
        physics_vals = defaultdict(list)
        for r in ok:
            if r.get("physics"):
                for pk, pv in r["physics"].items():
                    physics_vals[pk].append(pv)
        print("\nPhysics parameter diversity:")
        for pk, vals in sorted(physics_vals.items()):
            print(f"  {pk}: mean={np.mean(vals):.2f}, "
                  f"std={np.std(vals):.2f}, "
                  f"range=[{min(vals):.1f}, {max(vals):.1f}]")

    # Weight collapse check
    from collections import Counter
    wt_counter = Counter()
    for r in ok:
        if r.get("weights"):
            wt = tuple(r["weights"][k] for k in sorted(r["weights"].keys()))
            wt_counter[wt] += 1
    collapsed = sum(c for c in wt_counter.values() if c > 1)
    print(f"\nWeight collapse: {len(wt_counter)} unique vectors, "
          f"{collapsed}/{len(ok)} in collapsed clusters")

    # Physics+weight collapse (12D)
    full_counter = Counter()
    for r in ok:
        if r.get("weights") and r.get("physics"):
            wt = tuple(r["weights"][k] for k in sorted(r["weights"].keys()))
            ph = tuple(r["physics"][k] for k in sorted(r["physics"].keys()))
            full_counter[(wt, ph)] += 1
    full_collapsed = sum(c for c in full_counter.values() if c > 1)
    print(f"12D collapse (weights+physics): {len(full_counter)} unique vectors, "
          f"{full_collapsed}/{len(ok)} in collapsed clusters")

    # Save
    out_path = ARTIFACTS / f"tiqm_{seed_set_name}_phase{phase}.json"
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed_set": seed_set_name,
        "phase": phase,
        "iterative": iterative,
        "max_rounds": max_rounds if iterative else 1,
        "multi_offer": multi_offer,
        "models": models,
        "confirmation_model": CONFIRMATION_MODEL if phase >= 2 else None,
        "temperature": temperature,
        "n_trials": len(results),
        "n_ok": len(ok),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TIQM Locomotion Experiment")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2],
                        help="Phase 1=offer only, Phase 2=offer+confirmation")
    parser.add_argument("--seeds", default="romeo_juliet",
                        help="Seed set: romeo_juliet, death_test")
    parser.add_argument("--iterative", action="store_true",
                        help="Enable iterative TIQM (multiple rounds)")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max rounds for iterative TIQM")
    parser.add_argument("--multi-offer", type=int, default=1,
                        help="Number of competing offers per seed per model")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model list")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    run_tiqm_experiment(
        seed_set_name=args.seeds,
        phase=args.phase,
        iterative=args.iterative,
        max_rounds=args.max_rounds,
        multi_offer=args.multi_offer,
        models=args.models,
        temperature=args.temperature,
    )
