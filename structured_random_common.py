#!/usr/bin/env python3
"""
structured_random_common.py

Shared infrastructure for the structured random search experiment.

EXPERIMENT OVERVIEW
===================
This experiment tests whether an LLM can serve as a structured sampler of
neural network weight space. Instead of drawing 6 synapse weights uniformly
at random from [-1, 1]^6, we give a local LLM (Ollama, qwen3-coder:30b) a
"seed" — a verb, a theorem, a Bible verse, or a place name — and ask it to
translate the seed's character into 6 specific synapse weights. We then run
a full headless PyBullet simulation and compute Beer-framework analytics on
the resulting gait.

The hypothesis is that LLM-mediated weight generation is a *structured*
random search: the LLM's internal representations impose correlations on
the weight vectors that align with dynamically meaningful submanifolds of
the 6D weight space. This module provides the shared plumbing that all
five experimental conditions use.

PIPELINE (per trial)
====================
1. Condition script selects a seed (e.g. "Pythagorean Theorem")
2. Condition script builds a prompt via its make_prompt() function
3. This module sends the prompt to Ollama → gets back a JSON weight vector
4. This module writes brain.nndf with those weights
5. This module runs a headless PyBullet simulation (4000 steps @ 240 Hz)
6. All telemetry is captured in pre-allocated numpy arrays (no disk I/O)
7. compute_all() from compute_beer_analytics.py computes the full 4-pillar
   Beer-framework analysis: Outcome, Contact, Coordination, Rotation Axis
8. Key scalar metrics are extracted and returned to the condition script

THE ROBOT
=========
3-link body (Torso + BackLeg + FrontLeg), 2 hinge joints, 3 touch sensors
(neurons 0-2), 2 motors (neurons 3-4). Standard 6-synapse topology:
  w03: Torso sensor    → BackLeg motor
  w04: Torso sensor    → FrontLeg motor
  w13: BackLeg sensor  → BackLeg motor
  w14: BackLeg sensor  → FrontLeg motor
  w23: FrontLeg sensor → BackLeg motor
  w24: FrontLeg sensor → FrontLeg motor

Each weight is clamped to [-1, 1]. The neural network is a continuous-time
recurrent neural network (CTRNN) that updates every timestep based on sensor
inputs. Motor neurons output joint position targets.

BEER-FRAMEWORK METRICS
=======================
Each trial returns these scalar metrics (see compute_beer_analytics.py):
  dx, dy:       Net displacement in meters (primary fitness measure)
  speed:        Mean instantaneous speed over the simulation
  efficiency:   Distance traveled per unit of work (distance / work_proxy)
  work_proxy:   Sum of |torque × angular_velocity| over all timesteps
  phase_lock:   Inter-joint phase coherence [0,1] via Hilbert transform
  entropy:      Shannon entropy of 3-bit contact state distribution (bits)
  roll_dom:     Fraction of angular velocity variance in the roll axis
  yaw_net_rad:  Net yaw rotation over the simulation (radians)

Used by: structured_random_verbs.py, structured_random_theorems.py,
         structured_random_bible.py, structured_random_places.py,
         structured_random_compare.py
"""

import json
import subprocess
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from compute_beer_analytics import compute_all, DT, NumpyEncoder

# Neuron indices matching the robot's brain.nndf topology.
# Sensors 0-2 read ground-contact for Torso, BackLeg, FrontLeg respectively.
# Motors 3-4 drive hinge joints Torso_BackLeg and Torso_FrontLeg.
SENSOR_NEURONS = [0, 1, 2]
MOTOR_NEURONS = [3, 4]

# The 6 synapse weight names: "wXY" means source neuron X → target neuron Y.
# This enumerates all sensor→motor connections: w03, w04, w13, w14, w23, w24.
WEIGHT_NAMES = [f"w{s}{m}" for s in SENSOR_NEURONS for m in MOTOR_NEURONS]

# Local LLM configuration. qwen3-coder:30b was chosen as the largest and most
# capable model available on the local Ollama instance. The 30B parameter count
# provides enough capacity to encode meaningful structural representations of
# abstract concepts (theorems, verses, places) while remaining fast (~1s/call).
OLLAMA_MODEL = "qwen3-coder:30b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Each condition runs 100 trials. With ~1s per Ollama call and ~0.1s per
# headless simulation, a full condition takes ~100s. All 5 conditions
# (4 structured + 1 baseline) complete in ~7 minutes.
NUM_TRIALS = 100
PLOT_DIR = PROJECT / "artifacts" / "plots"


# ── Ollama integration ───────────────────────────────────────────────────────
# The LLM serves as a "structured sampler" — it maps a semantic seed (a verb,
# theorem, verse, or place name) into a 6D weight vector. The key question is
# whether this mapping imposes meaningful structure on the weight distribution
# compared to uniform random sampling.

def ask_ollama(prompt, temperature=0.8, max_tokens=200):
    """Send a prompt to the local Ollama LLM and return the response text.

    Uses curl to POST to Ollama's REST API with stream=False (blocking).
    Temperature 0.8 provides moderate variation — high enough that the same
    seed won't always produce identical weights, low enough to stay coherent.

    Args:
        prompt: The prompt string (typically a structured request for 6 weights).
        temperature: Sampling temperature (0.8 gives good seed→weight diversity).
        max_tokens: Maximum tokens to generate (200 is generous for a JSON object).

    Returns:
        Response string from the model (ideally a JSON object with 6 weights).

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    })
    r = subprocess.run(
        ["curl", "-s", OLLAMA_URL, "-d", payload],
        capture_output=True, text=True, timeout=60
    )
    if r.returncode != 0:
        raise RuntimeError(f"Ollama request failed: {r.stderr}")
    data = json.loads(r.stdout)
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")
    return data["response"]


def parse_weights(response):
    """Parse 6 synapse weights from an LLM response string.

    The prompt asks the LLM to return ONLY a JSON object, but LLMs frequently
    wrap their response in markdown code fences (```json ... ```) or add
    explanatory text before/after the JSON. This parser handles all of those
    cases by stripping fences and searching for the outermost { ... } pair.

    Values are clamped to [-1, 1] to stay within the weight space bounds.
    All 6 canonical weight names (w03, w04, w13, w14, w23, w24) must be
    present or the parse is considered a failure.

    Args:
        response: Raw string from the LLM.

    Returns:
        Dict mapping weight names to float values, or None if parsing fails.
    """
    text = response.strip()
    # Strip markdown code fences if present (common LLM behavior)
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Find the JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        raw = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None

    # Validate: must have all 6 weight keys with numeric values in [-1, 1]
    weights = {}
    for wn in WEIGHT_NAMES:
        if wn not in raw:
            return None
        v = float(raw[wn])
        weights[wn] = max(-1.0, min(1.0, v))  # clamp to bounds
    return weights


def generate_weights(prompt, retries=2, temperature=0.8):
    """Generate weights via Ollama with retry logic.

    On parse failure (malformed JSON, missing keys), retries up to `retries`
    times with a 0.5s backoff. In practice, qwen3-coder:30b produces valid
    JSON on the first attempt >99% of the time, so retries are a safety net.

    Args:
        prompt: The structured prompt for weight generation.
        retries: Number of retry attempts on parse failure.
        temperature: Sampling temperature for the LLM (higher = more diverse).

    Returns:
        Tuple of (weights_dict, raw_response) or (None, raw_response) on failure.
    """
    for attempt in range(retries + 1):
        try:
            resp = ask_ollama(prompt, temperature=temperature)
            weights = parse_weights(resp)
            if weights is not None:
                return weights, resp
        except Exception as e:
            resp = str(e)
        if attempt < retries:
            time.sleep(0.5)
    return None, resp


# ── Simulation ───────────────────────────────────────────────────────────────
# Each trial runs a complete headless PyBullet simulation: write brain.nndf,
# load robot, step 4000 times at 240 Hz (~16.7 seconds of simulated time),
# record full telemetry in pre-allocated numpy arrays, compute analytics.
# The entire cycle takes ~0.1s per trial thanks to DIRECT mode and no disk I/O.

def write_brain(weights):
    """Write a brain.nndf file with the standard 6-synapse topology.

    brain.nndf is the neural network definition file that the pyrosim
    library reads to construct the CTRNN. It defines 5 neurons (3 sensor,
    2 motor) and 6 weighted synapses connecting every sensor to every motor.

    This is a shared file that gets overwritten for each trial. The caller
    is responsible for backing up and restoring the original.

    Args:
        weights: Dict mapping synapse names (w03, w04, ...) to float values.
    """
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for s in SENSOR_NEURONS:
            for m in MOTOR_NEURONS:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def run_trial_inmemory(weights):
    """Run a complete headless PyBullet simulation and return Beer-framework analytics.

    This is the core simulation function. It:
    1. Writes brain.nndf with the given weights
    2. Connects PyBullet in DIRECT mode (headless, deterministic)
    3. Loads the ground plane and robot URDF
    4. Runs 4000 timesteps of the sense→think→act loop
    5. Records ALL state variables at every timestep into pre-allocated numpy
       arrays (position, velocity, angular velocity, euler angles, contact
       states for all 3 links, joint positions/velocities/torques)
    6. Passes the arrays to compute_all() for the full Beer-framework analysis

    The in-memory approach avoids writing telemetry JSONL files to disk,
    making each trial ~0.1s instead of ~0.5s. This is critical for running
    500 trials in under 10 minutes.

    Simulations are fully deterministic: identical weights always produce
    identical trajectories (PyBullet DIRECT mode, fixed random seed in physics).

    Args:
        weights: Dict mapping synapse names (w03, w04, ...) to float values.

    Returns:
        Analytics dict with 4 pillars:
          outcome:       dx, dy, yaw, speed, efficiency, work_proxy
          contact:       per-link duty fractions, contact entropy, transitions
          coordination:  FFT dominant freq/amp, phase_lock_score
          rotation_axis: axis dominance, switching rate, periodicity
    """
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
    max_force = float(getattr(c, "MAX_FORCE", 150.0))
    n_steps = c.SIM_STEPS

    # Pre-allocate arrays for all telemetry channels.
    # This avoids list appends and produces contiguous memory for numpy ops.
    # 4000 steps × ~20 channels = ~640KB total — trivially fits in L2 cache.
    t_arr = np.empty(n_steps)
    x = np.empty(n_steps); y = np.empty(n_steps); z = np.empty(n_steps)
    vx = np.empty(n_steps); vy = np.empty(n_steps); vz = np.empty(n_steps)
    wx = np.empty(n_steps); wy = np.empty(n_steps); wz = np.empty(n_steps)
    roll = np.empty(n_steps); pitch = np.empty(n_steps); yaw = np.empty(n_steps)
    contact_torso = np.empty(n_steps, dtype=bool)
    contact_back = np.empty(n_steps, dtype=bool)
    contact_front = np.empty(n_steps, dtype=bool)
    j0_pos = np.empty(n_steps); j0_vel = np.empty(n_steps); j0_tau = np.empty(n_steps)
    j1_pos = np.empty(n_steps); j1_vel = np.empty(n_steps); j1_tau = np.empty(n_steps)

    # Resolve link and joint indices from the URDF. The robot has 3 links
    # (Torso at base index -1, BackLeg and FrontLeg as child links) and
    # 2 joints (Torso_BackLeg, Torso_FrontLeg). We need the indices for
    # contact detection (which link is touching the ground?) and joint state
    # queries (what angle/velocity/torque is each joint at?).
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

    torso_link_idx = link_indices.get("Torso", -1)
    back_link_idx = link_indices.get("BackLeg", -1)
    front_link_idx = link_indices.get("FrontLeg", -1)
    j0_idx = joint_indices.get("Torso_BackLeg", 0)
    j1_idx = joint_indices.get("Torso_FrontLeg", 1)

    # Main simulation loop: 4000 steps of sense→think→act.
    # Act first (motor commands from previous think step), then step physics,
    # then think (update NN from new sensor readings), then record telemetry.
    for i in range(n_steps):
        # ACT: read motor neuron outputs and send as joint position targets.
        for neuronName in nn.neurons:
            n_obj = nn.neurons[neuronName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, n_obj.Get_Value(), max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, p.POSITION_CONTROL,
                                                n_obj.Get_Value(), max_force)
        p.stepSimulation()   # STEP: advance physics by 1/240 second
        nn.Update()          # THINK: update CTRNN from current sensor values

        # RECORD: capture full state at this timestep
        t_arr[i] = i * c.DT
        pos, orn = p.getBasePositionAndOrientation(robotId)
        vel_lin, vel_ang = p.getBaseVelocity(robotId)
        rpy_vals = p.getEulerFromQuaternion(orn)

        x[i] = pos[0]; y[i] = pos[1]; z[i] = pos[2]
        vx[i] = vel_lin[0]; vy[i] = vel_lin[1]; vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]; wy[i] = vel_ang[1]; wz[i] = vel_ang[2]
        roll[i] = rpy_vals[0]; pitch[i] = rpy_vals[1]; yaw[i] = rpy_vals[2]

        # Contact detection: which of the 3 links are touching the ground?
        # This produces a 3-bit contact state (8 possible states) that feeds
        # into the contact entropy metric — a measure of gait complexity.
        contact_pts = p.getContactPoints(robotId)
        torso_contact = False; back_contact = False; front_contact = False
        for cp in contact_pts:
            li = cp[3]
            if li == -1:
                torso_contact = True
            elif li == back_link_idx:
                back_contact = True
            elif li == front_link_idx:
                front_contact = True
        contact_torso[i] = torso_contact
        contact_back[i] = back_contact
        contact_front[i] = front_contact

        js0 = p.getJointState(robotId, j0_idx)
        js1 = p.getJointState(robotId, j1_idx)
        j0_pos[i] = js0[0]; j0_vel[i] = js0[1]; j0_tau[i] = js0[3]
        j1_pos[i] = js1[0]; j1_vel[i] = js1[1]; j1_tau[i] = js1[3]

    p.disconnect()

    # Package all telemetry into the dict format that compute_all() expects.
    # This mirrors the structure produced by the JSONL telemetry logger but
    # with numpy arrays instead of per-row dicts — much faster to analyze.
    data = {
        "t": t_arr, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "contact_torso": contact_torso,
        "contact_back": contact_back,
        "contact_front": contact_front,
        "j0_pos": j0_pos, "j0_vel": j0_vel, "j0_tau": j0_tau,
        "j1_pos": j1_pos, "j1_vel": j1_vel, "j1_tau": j1_tau,
    }
    return compute_all(data, DT)


# ── Runner ───────────────────────────────────────────────────────────────────
# The runner orchestrates the full pipeline for a given experimental condition:
# for each seed, build a prompt, call Ollama, parse weights, run a simulation,
# compute analytics, and collect results. It also handles brain.nndf backup/
# restore, progress reporting, and summary statistics.

def run_structured_search(condition_name, seeds, prompt_fn, out_json, temperature=0.8,
                          weight_transform=None):
    """Run a structured random search for a given experimental condition.

    This is the main entry point that each condition script calls. It iterates
    over the seed list, sends each seed through the LLM→simulation→analytics
    pipeline, and writes the collected results to a JSON file.

    The output JSON is a list of dicts, one per successful trial, each with:
      trial:      trial index (0-based)
      seed:       the original seed string (verb, theorem, verse, or place)
      weights:    dict of {w03, w04, w13, w14, w23, w24} float values
      dx, dy:     net displacement in meters
      speed:      mean instantaneous speed
      efficiency: distance per unit work (distance / work_proxy)
      work_proxy: cumulative |torque × angular_velocity|
      phase_lock: inter-joint phase coherence [0,1]
      entropy:    Shannon entropy of contact state distribution
      roll_dom:   roll-axis dominance of angular velocity
      yaw_net_rad: net yaw rotation in radians

    Args:
        condition_name: Human-readable condition name (e.g. "verbs").
        seeds: List of seed values (verbs, theorems, verses, places).
        prompt_fn: Callable(seed) -> prompt string for Ollama.
        out_json: Path to output JSON file.
        temperature: Sampling temperature for the LLM (default 0.8; use 1.5
            for large-scale experiments that need per-seed uniqueness).
        weight_transform: Optional callable(weights_dict, seed) -> weights_dict.
            Applied after LLM generation, before simulation. Use for per-seed
            perturbation to break weight-vector collapse.

    Returns:
        List of result dicts.
    """
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.sr_backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    results = []
    failures = 0
    n = min(NUM_TRIALS, len(seeds))
    print(f"\n{'='*70}")
    print(f"STRUCTURED RANDOM SEARCH: {condition_name.upper()}")
    print(f"{'='*70}")
    print(f"Running {n} trials via Ollama ({OLLAMA_MODEL}, temp={temperature})...\n")

    t_total = time.perf_counter()

    for trial in range(n):
        seed = seeds[trial]
        prompt = prompt_fn(seed)
        weights, raw = generate_weights(prompt, temperature=temperature)

        if weights is None:
            failures += 1
            print(f"  [{trial+1:3d}/{n}] FAIL: could not parse weights for: {seed}")
            continue

        if weight_transform is not None:
            weights = weight_transform(weights, seed)

        analytics = run_trial_inmemory(weights)
        o = analytics["outcome"]
        coord = analytics["coordination"]
        contact = analytics["contact"]
        ra = analytics["rotation_axis"]

        result = {
            "trial": trial,
            "seed": seed,
            "weights": weights,
            "dx": o["dx"],
            "dy": o.get("dy", 0),
            "speed": o["mean_speed"],
            "efficiency": o["distance_per_work"],
            "work_proxy": o["work_proxy"],
            "phase_lock": coord["phase_lock_score"],
            "entropy": contact["contact_entropy_bits"],
            "roll_dom": ra["axis_dominance"][0],
            "yaw_net_rad": o["yaw_net_rad"],
        }
        results.append(result)

        if (trial + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_total
            print(f"  [{trial+1:3d}/{n}] {elapsed:.1f}s  "
                  f"DX={o['dx']:+7.2f}  seed={str(seed)[:50]}", flush=True)

    total_elapsed = time.perf_counter() - t_total
    print(f"\nDone: {len(results)} successful, {failures} failures, {total_elapsed:.1f}s total")

    # Restore brain.nndf
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    # Save results
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"WROTE {out_json}")

    # Print summary
    if results:
        dxs = [abs(r["dx"]) for r in results]
        dead = sum(1 for d in dxs if d < 1.0)
        print(f"\nSummary ({condition_name}):")
        print(f"  Dead (|DX|<1m): {dead}/{len(results)} ({dead/len(results)*100:.0f}%)")
        print(f"  Median |DX|: {np.median(dxs):.2f}m")
        print(f"  Max |DX|: {max(dxs):.2f}m")
        best = max(results, key=lambda r: abs(r["dx"]))
        print(f"  Best: DX={best['dx']:+.2f}  seed={best['seed']}")

    return results


def run_uniform_baseline(out_json):
    """Run 100 uniform random trials as the baseline (control) condition.

    This is the null hypothesis condition: weights are drawn independently
    from U[-1, 1] with no LLM involvement. It provides the reference
    distribution against which all structured conditions are compared.

    The baseline tests what happens when you sample the 6D weight hypercube
    uniformly — the same approach used by random_search_500.py in the
    broader research campaign. Key baseline statistics from past runs:
    ~8% dead gaits, median |DX| ~6-7m, high behavioral diversity but low
    phase lock (~0.6).

    Args:
        out_json: Path to output JSON file.

    Returns:
        List of result dicts (same schema as structured conditions, but
        seed is always "uniform_random").
    """
    import random

    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.sr_backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    results = []
    n = NUM_TRIALS
    print(f"\n{'='*70}")
    print(f"STRUCTURED RANDOM SEARCH: UNIFORM BASELINE")
    print(f"{'='*70}")
    print(f"Running {n} trials with uniform random weights in [-1,1]^6...\n")

    t_total = time.perf_counter()

    for trial in range(n):
        weights = {wn: random.uniform(-1, 1) for wn in WEIGHT_NAMES}
        analytics = run_trial_inmemory(weights)
        o = analytics["outcome"]
        coord = analytics["coordination"]
        contact = analytics["contact"]
        ra = analytics["rotation_axis"]

        result = {
            "trial": trial,
            "seed": "uniform_random",
            "weights": weights,
            "dx": o["dx"],
            "dy": o.get("dy", 0),
            "speed": o["mean_speed"],
            "efficiency": o["distance_per_work"],
            "work_proxy": o["work_proxy"],
            "phase_lock": coord["phase_lock_score"],
            "entropy": contact["contact_entropy_bits"],
            "roll_dom": ra["axis_dominance"][0],
            "yaw_net_rad": o["yaw_net_rad"],
        }
        results.append(result)

        if (trial + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_total
            print(f"  [{trial+1:3d}/{n}] {elapsed:.1f}s  "
                  f"DX={o['dx']:+7.2f}", flush=True)

    total_elapsed = time.perf_counter() - t_total
    print(f"\nDone: {len(results)} trials, {total_elapsed:.1f}s total")

    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"WROTE {out_json}")

    dxs = [abs(r["dx"]) for r in results]
    dead = sum(1 for d in dxs if d < 1.0)
    print(f"\nSummary (baseline):")
    print(f"  Dead (|DX|<1m): {dead}/{len(results)} ({dead/len(results)*100:.0f}%)")
    print(f"  Median |DX|: {np.median(dxs):.2f}m")
    print(f"  Max |DX|: {max(dxs):.2f}m")

    return results
