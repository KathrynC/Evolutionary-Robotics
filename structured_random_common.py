#!/usr/bin/env python3
"""
structured_random_common.py

Shared infrastructure for the structured random search experiment.
Provides Ollama integration, weight parsing, headless simulation with
in-memory telemetry, and Beer-framework analytics computation.

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

SENSOR_NEURONS = [0, 1, 2]
MOTOR_NEURONS = [3, 4]
WEIGHT_NAMES = [f"w{s}{m}" for s in SENSOR_NEURONS for m in MOTOR_NEURONS]

OLLAMA_MODEL = "qwen3-coder:30b"
OLLAMA_URL = "http://localhost:11434/api/generate"

NUM_TRIALS = 100
PLOT_DIR = PROJECT / "artifacts" / "plots"


# ── Ollama integration ───────────────────────────────────────────────────────

def ask_ollama(prompt, temperature=0.8, max_tokens=200):
    """Send a prompt to Ollama and return the response text.

    Args:
        prompt: The prompt string.
        temperature: Sampling temperature (higher = more varied).
        max_tokens: Maximum tokens to generate.

    Returns:
        Response string from the model.

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

    Attempts to extract a JSON object with keys w03, w04, w13, w14, w23, w24.
    Handles responses that include markdown code fences or extra text around
    the JSON.

    Args:
        response: Raw string from the LLM.

    Returns:
        Dict mapping weight names to float values, or None if parsing fails.
    """
    text = response.strip()
    # Strip markdown code fences if present
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


def generate_weights(prompt, retries=2):
    """Generate weights via Ollama with retry logic.

    Args:
        prompt: The structured prompt for weight generation.
        retries: Number of retry attempts on parse failure.

    Returns:
        Tuple of (weights_dict, raw_response) or (None, raw_response) on failure.
    """
    for attempt in range(retries + 1):
        try:
            resp = ask_ollama(prompt)
            weights = parse_weights(resp)
            if weights is not None:
                return weights, resp
        except Exception as e:
            resp = str(e)
        if attempt < retries:
            time.sleep(0.5)
    return None, resp


# ── Simulation ───────────────────────────────────────────────────────────────

def write_brain(weights):
    """Write a brain.nndf file with the given 6 synapse weights.

    Args:
        weights: Dict mapping synapse names to float weight values.
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
    """Run a headless simulation and return Beer-framework analytics.

    Captures all telemetry in pre-allocated numpy arrays (no disk I/O)
    and passes them to compute_all() for the full 4-pillar analysis.

    Args:
        weights: Dict mapping synapse names to float weight values.

    Returns:
        Analytics dict with keys: outcome, contact, coordination, rotation_axis.
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

    # Pre-allocate arrays
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

    # Link/joint indices
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

    for i in range(n_steps):
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
        p.stepSimulation()
        nn.Update()

        t_arr[i] = i * c.DT
        pos, orn = p.getBasePositionAndOrientation(robotId)
        vel_lin, vel_ang = p.getBaseVelocity(robotId)
        rpy_vals = p.getEulerFromQuaternion(orn)

        x[i] = pos[0]; y[i] = pos[1]; z[i] = pos[2]
        vx[i] = vel_lin[0]; vy[i] = vel_lin[1]; vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]; wy[i] = vel_ang[1]; wz[i] = vel_ang[2]
        roll[i] = rpy_vals[0]; pitch[i] = rpy_vals[1]; yaw[i] = rpy_vals[2]

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

def run_structured_search(condition_name, seeds, prompt_fn, out_json):
    """Run a structured random search for a given condition.

    Args:
        condition_name: Human-readable condition name (e.g. "verbs").
        seeds: List of seed values (verbs, theorems, verses, places).
        prompt_fn: Callable(seed) -> prompt string for Ollama.
        out_json: Path to output JSON file.

    Returns:
        List of result dicts with keys: trial, seed, weights, analytics (scalars).
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
    print(f"Running {n} trials via Ollama ({OLLAMA_MODEL})...\n")

    t_total = time.perf_counter()

    for trial in range(n):
        seed = seeds[trial]
        prompt = prompt_fn(seed)
        weights, raw = generate_weights(prompt)

        if weights is None:
            failures += 1
            print(f"  [{trial+1:3d}/{n}] FAIL: could not parse weights for: {seed}")
            continue

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
    """Run 100 uniform random trials as the baseline condition.

    Args:
        out_json: Path to output JSON file.

    Returns:
        List of result dicts.
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
