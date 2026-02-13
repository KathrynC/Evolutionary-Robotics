#!/usr/bin/env python3
"""
behavioral_embryology.py

Role:
    Research campaign script that tracks the emergence of locomotion behavior
    during the first ~500 simulation steps and beyond. Compares developmental
    trajectories of high-performing evolved gaits vs random-weight controls.

    Answers four key questions:
      1. When does organized locomotion "start"?
      2. Is there a transient (stumbling phase) before stable gait?
      3. When do coordination metrics (phase lock, contact entropy) reach
         steady state?
      4. Does the Novelty Champion develop faster, slower, or differently
         from other gaits?

Approach:
    Run full 4000-step sims with per-step trajectory capture. Compute Beer
    analytics at 9 cumulative time windows. Plot developmental curves.

    Part 1: Simulate named gaits with full trajectory capture.
    Part 2: Load existing telemetry for zoo gaits (no new sims).
    Part 3: Simulate random-weight controls as baseline.
    Part 4: Compute windowed analytics and generate 6 figures.

Simulation budget:
    ~25 sims (NC, Trial 3, + 7 zoo gaits from telemetry, + 10 random controls)
    ~25 sims x ~2s = ~50s total

Notes:
    - This script is self-contained: it defines its own simulation harness
      rather than importing from simulate.py.
    - Simulations run in PyBullet DIRECT mode (headless, deterministic).
    - Zoo gaits are loaded from pre-existing per-step telemetry JSONL files
      rather than re-simulated.
    - All analytics use numpy-only signal processing (no scipy).

Outputs:
    artifacts/behavioral_embryology.json
    artifacts/plots/emb_fig01_displacement_curves.png
    artifacts/plots/emb_fig02_speed_emergence.png
    artifacts/plots/emb_fig03_contact_entropy.png
    artifacts/plots/emb_fig04_phase_lock.png
    artifacts/plots/emb_fig05_developmental_fingerprints.png
    artifacts/plots/emb_fig06_onset_summary.png

Usage:
    python3 behavioral_embryology.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as c
from pyrosim.neuralNetwork import NEURAL_NETWORK
import pyrosim.pyrosim as pyrosim

# ── Config ──────────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent
PLOT_DIR = PROJECT / "artifacts" / "plots"
OUT_JSON = PROJECT / "artifacts" / "behavioral_embryology.json"
TELEMETRY_DIR = PROJECT / "artifacts" / "telemetry"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
DT = c.DT  # 1/240

# Time windows (cumulative, in timesteps)
WINDOWS = [50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000]

WEIGHT_NAMES = ["w03", "w04", "w13", "w14", "w23", "w24"]

# ── Named gaits ─────────────────────────────────────────────────────────────

NOVELTY_CHAMPION = {
    "w03": -1.3083167156740476,
    "w04": -0.34279812804233867,
    "w13": 0.8331363773051514,
    "w14": -0.37582983217830773,
    "w23": -0.0369713954829298,
    "w24": 0.4375020967145814,
}

TRIAL3 = {
    "w03": -0.5971393487736976,
    "w04": -0.4236677331634211,
    "w13": 0.11222931078528431,
    "w14": -0.004679977731207874,
    "w23": 0.2970146930268889,
    "w24": 0.21399448704946855,
}

# Named gaits to simulate (no existing telemetry at per-step resolution)
NAMED_GAITS = {
    "Novelty Champion": NOVELTY_CHAMPION,
    "Trial 3": TRIAL3,
}

# Zoo gaits to load from existing telemetry (already have 4000-step JSONL)
ZOO_GAITS = [
    "43_hidden_cpg_champion",
    "22_curie_amplified",
    "5_pelton",
    "1_original",
    "21_noether_cpg",
    "18_curie",
    "3_mordvintsev",
]

N_RANDOM = 10
RNG_SEED = 42


# ── Simulation ──────────────────────────────────────────────────────────────

def write_brain_standard(weights):
    """Write a standard 3-sensor/2-motor brain.nndf file from a weight dict.

    Creates the standard topology: 3 sensor neurons (Torso, BackLeg, FrontLeg)
    and 2 motor neurons (Torso_BackLeg, Torso_FrontLeg) with 6 synapses
    connecting every sensor to every motor.

    Args:
        weights: Dict mapping synapse names (e.g., "w03") to float weights.
            Must contain keys w03, w04, w13, w14, w23, w24.

    Side effects:
        Overwrites PROJECT / "brain.nndf" on disk.
    """
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for s in [0, 1, 2]:
            for m in [3, 4]:
                w = weights[f"w{s}{m}"]
                f.write(f'    <synapse sourceNeuronName = "{s}" '
                        f'targetNeuronName = "{m}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def simulate_full_trajectory(weights):
    """Run a full simulation with per-step trajectory capture.

    Sets up a headless PyBullet environment, loads the robot with the given
    brain weights, and records position, velocity, orientation, contact,
    and joint state at every timestep.

    Args:
        weights: Dict mapping synapse names (e.g., "w03") to float weights.

    Returns:
        Dict of numpy arrays keyed by channel name, with one entry per
        timestep. Keys include: t, x, y, z, vx, vy, vz, wx, wy, wz,
        roll, pitch, yaw, contact_torso, contact_back, contact_front,
        j0_pos, j0_vel, j0_tau, j1_pos, j1_vel, j1_tau.
        Format matches load_telemetry() output for uniform downstream use.

    Side effects:
        Writes brain.nndf to disk (via write_brain_standard).
        Connects and disconnects a PyBullet physics client.
    """
    write_brain_standard(weights)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(c.DT)

    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    mu = float(getattr(c, "ROBOT_FRICTION", 2.5))
    for link in range(-1, p.getNumJoints(robotId)):
        p.changeDynamics(robotId, link, lateralFriction=mu, restitution=0.0)

    nn = NEURAL_NETWORK("brain.nndf")
    max_force = float(getattr(c, "MAX_FORCE", 150.0))
    n_steps = c.SIM_STEPS

    # Pre-allocate
    t_arr = np.arange(n_steps, dtype=float)
    x = np.empty(n_steps)
    y = np.empty(n_steps)
    z = np.empty(n_steps)
    vx = np.empty(n_steps)
    vy = np.empty(n_steps)
    vz = np.empty(n_steps)
    wx = np.empty(n_steps)
    wy = np.empty(n_steps)
    wz = np.empty(n_steps)
    roll = np.empty(n_steps)
    pitch = np.empty(n_steps)
    yaw = np.empty(n_steps)
    contact_torso = np.empty(n_steps, dtype=bool)
    contact_back = np.empty(n_steps, dtype=bool)
    contact_front = np.empty(n_steps, dtype=bool)
    j0_pos = np.empty(n_steps)
    j0_vel = np.empty(n_steps)
    j0_tau = np.empty(n_steps)
    j1_pos = np.empty(n_steps)
    j1_vel = np.empty(n_steps)
    j1_tau = np.empty(n_steps)

    # Find link/joint indices for contact detection
    n_joints = p.getNumJoints(robotId)
    link_names = {}
    for ji in range(n_joints):
        info = p.getJointInfo(robotId, ji)
        link_names[info[12].decode("utf-8")] = info[0]
    # -1 = base (Torso)
    torso_idx = -1
    back_idx = link_names.get("BackLeg", 0)
    front_idx = link_names.get("FrontLeg", 1)

    for i in range(n_steps):
        # Motor commands
        for nName in nn.neurons:
            n_obj = nn.neurons[nName]
            if n_obj.Is_Motor_Neuron():
                jn = n_obj.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes,
                                                n_obj.Get_Value(), max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes,
                                                p.POSITION_CONTROL,
                                                n_obj.Get_Value(), max_force)
        p.stepSimulation()
        nn.Update()

        # Record state
        pos, orn = p.getBasePositionAndOrientation(robotId)
        vel_lin, vel_ang = p.getBaseVelocity(robotId)
        rpy_val = p.getEulerFromQuaternion(orn)

        x[i] = pos[0]
        y[i] = pos[1]
        z[i] = pos[2]
        vx[i] = vel_lin[0]
        vy[i] = vel_lin[1]
        vz[i] = vel_lin[2]
        wx[i] = vel_ang[0]
        wy[i] = vel_ang[1]
        wz[i] = vel_ang[2]
        roll[i] = rpy_val[0]
        pitch[i] = rpy_val[1]
        yaw[i] = rpy_val[2]

        # Contact detection
        contacts = p.getContactPoints(bodyA=robotId)
        links_touching = set()
        for cp in contacts:
            links_touching.add(cp[3])  # linkIndexA
        contact_torso[i] = torso_idx in links_touching
        contact_back[i] = back_idx in links_touching
        contact_front[i] = front_idx in links_touching

        # Joint states
        js0 = p.getJointState(robotId, 0)
        js1 = p.getJointState(robotId, 1)
        j0_pos[i] = js0[0]
        j0_vel[i] = js0[1]
        j0_tau[i] = js0[3]
        j1_pos[i] = js1[0]
        j1_vel[i] = js1[1]
        j1_tau[i] = js1[3]

    p.disconnect()

    return {
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


# ── Telemetry loader ────────────────────────────────────────────────────────

def load_telemetry(gait_name):
    """Load per-step telemetry from a JSONL file for a zoo gait.

    Reads the JSONL telemetry file and unpacks each record into numpy arrays
    matching the format returned by simulate_full_trajectory().

    Args:
        gait_name: Directory name under TELEMETRY_DIR (e.g., "5_pelton").

    Returns:
        Dict of numpy arrays with the same keys as simulate_full_trajectory().

    Raises:
        FileNotFoundError: If the telemetry JSONL file does not exist.
    """
    path = TELEMETRY_DIR / gait_name / "telemetry.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No telemetry for {gait_name}: {path}")

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n = len(records)
    data = {
        "t": np.empty(n), "x": np.empty(n), "y": np.empty(n), "z": np.empty(n),
        "vx": np.empty(n), "vy": np.empty(n), "vz": np.empty(n),
        "wx": np.empty(n), "wy": np.empty(n), "wz": np.empty(n),
        "roll": np.empty(n), "pitch": np.empty(n), "yaw": np.empty(n),
        "contact_torso": np.empty(n, dtype=bool),
        "contact_back": np.empty(n, dtype=bool),
        "contact_front": np.empty(n, dtype=bool),
        "j0_pos": np.empty(n), "j0_vel": np.empty(n), "j0_tau": np.empty(n),
        "j1_pos": np.empty(n), "j1_vel": np.empty(n), "j1_tau": np.empty(n),
    }

    for i, rec in enumerate(records):
        data["t"][i] = rec["t"]
        data["x"][i] = rec["base"]["x"]
        data["y"][i] = rec["base"]["y"]
        data["z"][i] = rec["base"]["z"]
        data["vx"][i] = rec["vel"]["vx"]
        data["vy"][i] = rec["vel"]["vy"]
        data["vz"][i] = rec["vel"]["vz"]
        data["wx"][i] = rec["ang_vel"]["wx"]
        data["wy"][i] = rec["ang_vel"]["wy"]
        data["wz"][i] = rec["ang_vel"]["wz"]
        data["roll"][i] = rec["rpy"]["r"]
        data["pitch"][i] = rec["rpy"]["p"]
        data["yaw"][i] = rec["rpy"]["y"]
        lc = rec["link_contacts"]
        data["contact_torso"][i] = lc[0]
        data["contact_back"][i] = lc[1]
        data["contact_front"][i] = lc[2]
        joints = rec["joints"]
        data["j0_pos"][i] = joints[0]["pos"]
        data["j0_vel"][i] = joints[0]["vel"]
        data["j0_tau"][i] = joints[0]["tau"]
        data["j1_pos"][i] = joints[1]["pos"]
        data["j1_vel"][i] = joints[1]["vel"]
        data["j1_tau"][i] = joints[1]["tau"]

    return data


# ── Signal processing ───────────────────────────────────────────────────────

def _hilbert_analytic(x):
    """Compute the analytic signal via FFT-based Hilbert transform (no scipy).

    Returns a complex array whose angle gives instantaneous phase.
    """
    n = len(x)
    X = np.fft.fft(x)
    # Build a one-sided frequency mask: keep DC and Nyquist as-is,
    # double positive frequencies, zero out negative frequencies.
    H = np.zeros(n)
    H[0] = 1.0
    if n % 2 == 0:
        H[1:n // 2] = 2.0
        H[n // 2] = 1.0
    else:
        H[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * H)


def _fft_peak(signal, dt):
    """Find the dominant frequency and amplitude in a signal via FFT.

    Returns (peak_frequency_hz, peak_amplitude). Returns (0, 0) for flat signals.
    """
    n = len(signal)
    sig = signal - np.mean(signal)
    if np.max(np.abs(sig)) < EPS:
        return 0.0, 0.0
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.abs(np.fft.rfft(sig)) * (2.0 / n)
    if len(spectrum) < 2:
        return 0.0, 0.0
    spectrum[0] = 0.0  # ignore DC component
    peak_idx = np.argmax(spectrum)
    return float(freqs[peak_idx]), float(spectrum[peak_idx])


# ── Windowed analytics ──────────────────────────────────────────────────────

def window_data(data, end_step):
    """Slice a trajectory data dict to the first end_step timesteps.

    Args:
        data: Dict of numpy arrays (trajectory data from simulation or telemetry).
        end_step: Number of timesteps to keep from the start.

    Returns:
        New dict with each numpy array truncated to [:end_step].
        Non-array values are passed through unchanged.
    """
    sliced = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            sliced[k] = v[:end_step]
        else:
            sliced[k] = v
    return sliced


def compute_windowed_analytics(data, end_step):
    """Compute lightweight Beer-framework analytics for a cumulative time window.

    Computes outcome metrics (displacement, speed, efficiency), contact
    pattern metrics (entropy, duty cycles), and coordination metrics
    (FFT frequencies, Hilbert phase lock) over the first end_step timesteps.

    Args:
        data: Dict of numpy arrays from simulate_full_trajectory() or
            load_telemetry().
        end_step: Number of timesteps to include in the analysis window.

    Returns:
        Dict of rounded metric values, or None if fewer than 10 timesteps.
        Keys include: window_steps, window_seconds, dx, dy, mean_speed,
        work_proxy, efficiency, heading_consistency, contact_entropy_bits,
        dominant_state, duty_torso, duty_back, duty_front, freq0_hz,
        freq1_hz, amp0, amp1, phase_lock_score, delta_phi_rad, z_mean, z_std.
    """
    d = window_data(data, end_step)
    n = len(d["x"])
    if n < 10:
        return None

    dt = DT

    # Outcome
    dx = float(d["x"][-1] - d["x"][0])
    dy = float(d["y"][-1] - d["y"][0])
    speed = np.sqrt(d["vx"]**2 + d["vy"]**2)
    mean_speed = float(np.mean(speed))

    # Work proxy — sum of |torque * angular_velocity| across both joints,
    # integrated over time. Efficiency = net distance traveled / total work.
    power = np.abs(d["j0_tau"] * d["j0_vel"]) + np.abs(d["j1_tau"] * d["j1_vel"])
    work_proxy = float(np.sum(power) * dt)
    net_dist = np.sqrt(dx**2 + dy**2)
    efficiency = float(net_dist / work_proxy) if work_proxy > EPS else 0.0

    # Heading consistency — circular mean of velocity heading angles.
    # Maps each velocity vector to a unit complex number e^(i*theta);
    # the magnitude of their mean measures directional consistency (0=random, 1=straight).
    speed_thresh = 0.1
    moving = speed > speed_thresh
    if np.sum(moving) > 10:
        theta = np.arctan2(d["vy"][moving], d["vx"][moving])
        heading_consistency = float(np.abs(np.mean(np.exp(1j * theta))))
    else:
        heading_consistency = 0.0

    # Contact — encode 3 binary contacts (torso, back, front) as a 3-bit
    # integer (0-7), giving 8 possible contact states per timestep.
    torso = d["contact_torso"].astype(int)
    back = d["contact_back"].astype(int)
    front = d["contact_front"].astype(int)
    state = torso * 4 + back * 2 + front  # 3-bit: torso|back|front
    state_counts = np.bincount(state, minlength=8)
    probs = state_counts / n
    # Shannon entropy over the contact-state distribution (bits).
    # Higher entropy = more varied ground-contact patterns during locomotion.
    nonzero = probs[probs > 0]
    contact_entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    dominant_state = int(np.argmax(state_counts))

    duty_torso = float(np.mean(torso))
    duty_back = float(np.mean(back))
    duty_front = float(np.mean(front))

    # Coordination (needs enough samples for FFT)
    if n >= 50:
        freq0, amp0 = _fft_peak(d["j0_pos"], dt)
        freq1, amp1 = _fft_peak(d["j1_pos"], dt)
        # Extract instantaneous phase of each joint via Hilbert analytic signal,
        # then compute inter-joint phase difference at each timestep.
        a0 = _hilbert_analytic(d["j0_pos"] - np.mean(d["j0_pos"]))
        a1 = _hilbert_analytic(d["j1_pos"] - np.mean(d["j1_pos"]))
        delta_phi = np.angle(a1) - np.angle(a0)
        # Phase-lock score: magnitude of the circular mean of e^(i*delta_phi).
        # 1.0 = perfectly constant phase relationship; 0.0 = no coordination.
        phase_lock = float(np.abs(np.mean(np.exp(1j * delta_phi))))
        mean_delta_phi = float(np.angle(np.mean(np.exp(1j * delta_phi))))
    else:
        freq0 = freq1 = amp0 = amp1 = 0.0
        phase_lock = 0.0
        mean_delta_phi = 0.0

    # Height stability
    z_mean = float(np.mean(d["z"]))
    z_std = float(np.std(d["z"]))

    return {
        "window_steps": end_step,
        "window_seconds": round(end_step * dt, 4),
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "mean_speed": round(mean_speed, 4),
        "work_proxy": round(work_proxy, 4),
        "efficiency": round(efficiency, 6),
        "heading_consistency": round(heading_consistency, 4),
        "contact_entropy_bits": round(contact_entropy, 4),
        "dominant_state": dominant_state,
        "duty_torso": round(duty_torso, 4),
        "duty_back": round(duty_back, 4),
        "duty_front": round(duty_front, 4),
        "freq0_hz": round(freq0, 4),
        "freq1_hz": round(freq1, 4),
        "amp0": round(amp0, 4),
        "amp1": round(amp1, 4),
        "phase_lock_score": round(phase_lock, 4),
        "delta_phi_rad": round(mean_delta_phi, 4),
        "z_mean": round(z_mean, 4),
        "z_std": round(z_std, 4),
    }


def compute_onset_time(analytics_series, metric_key, threshold_frac=0.8):
    """Find the earliest window where a metric reaches a fraction of its final value.

    Walks the analytics windows chronologically and returns the first window
    where the metric crosses threshold_frac of the final-window value.
    Direction-aware: handles both positive and negative metric trajectories.

    Args:
        analytics_series: List of analytics dicts (one per window), as
            returned by compute_windowed_analytics(). May contain None entries.
        metric_key: String key into the analytics dict (e.g., "mean_speed").
        threshold_frac: Fraction of the final value to use as the onset
            threshold. Default 0.8 (80%).

    Returns:
        Tuple (onset_steps, onset_seconds) for the first qualifying window,
        or (None, None) if the metric never reaches the threshold or the
        final value is near zero.
    """
    values = [a[metric_key] for a in analytics_series if a is not None]
    if not values or abs(values[-1]) < EPS:
        return None, None
    final = values[-1]
    target = threshold_frac * final
    # Walk windows chronologically; return the first that crosses the
    # threshold fraction of the final value. Direction-aware: for positive
    # metrics, check >=; for negative metrics (e.g., negative DX), check <=.
    for a in analytics_series:
        if a is None:
            continue
        if final > 0 and a[metric_key] >= target:
            return a["window_steps"], a["window_seconds"]
        if final < 0 and a[metric_key] <= target:
            return a["window_steps"], a["window_seconds"]
    return None, None


# ── Plotting helpers ────────────────────────────────────────────────────────

def clean_ax(ax):
    """Remove top/right spines and shrink tick labels for a cleaner plot."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def save_fig(fig, name):
    """Save a matplotlib figure to PLOT_DIR and close it."""
    path = PLOT_DIR / name
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  WROTE {path}")


# Color scheme
COLORS = {
    "Novelty Champion": "#E24A33",
    "Trial 3": "#55A868",
    "43_hidden_cpg_champion": "#348ABD",
    "22_curie_amplified": "#988ED5",
    "5_pelton": "#FBC15E",
    "1_original": "#8EBA42",
    "21_noether_cpg": "#FFB5B8",
    "18_curie": "#777777",
    "3_mordvintsev": "#E5AE38",
}


def get_color(name):
    """Return the assigned color for a gait, defaulting to gray for randoms."""
    if name in COLORS:
        return COLORS[name]
    # Random controls get gray
    return "#BBBBBB"


def short_name(name):
    """Shorten zoo gait names for labels."""
    mapping = {
        "43_hidden_cpg_champion": "Hidden CPG",
        "22_curie_amplified": "Curie Amp",
        "5_pelton": "Pelton",
        "1_original": "Original",
        "21_noether_cpg": "Noether CPG",
        "18_curie": "Curie",
        "3_mordvintsev": "Mordvintsev",
    }
    return mapping.get(name, name)


# ── Figures ─────────────────────────────────────────────────────────────────

def fig01_displacement_curves(all_results):
    """Plot displacement development over time for all gaits (2-panel figure).

    Left panel: continuous x(t) position trajectories.
    Right panel: cumulative DX at each analysis window.
    Named and zoo gaits are drawn with thick colored lines; random controls
    are drawn as thin gray lines in the background.

    Args:
        all_results: Dict mapping gait name to result dict containing
            "data" (trajectory arrays) and "analytics" (windowed metrics).

    Side effects:
        Saves emb_fig01_displacement_curves.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Displacement Development Over Time", fontsize=13, fontweight="bold")

    # Left: full x(t) trajectory
    ax = axes[0]
    ax.set_title("Position x(t)", fontsize=10)
    for name, result in all_results.items():
        data = result["data"]
        t_sec = data["t"] * DT
        color = get_color(name)
        lw = 2.0 if name in NAMED_GAITS or name in ZOO_GAITS else 0.5
        alpha = 1.0 if lw > 1 else 0.3
        label = short_name(name) if lw > 1 else None
        ax.plot(t_sec, data["x"] - data["x"][0], color=color, lw=lw, alpha=alpha,
                label=label)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("DX (m)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    clean_ax(ax)

    # Right: DX at each window
    ax = axes[1]
    ax.set_title("DX at Cumulative Windows", fontsize=10)
    for name, result in all_results.items():
        windows = [a["window_seconds"] for a in result["analytics"]]
        dxs = [a["dx"] for a in result["analytics"]]
        color = get_color(name)
        lw = 2.0 if name in NAMED_GAITS or name in ZOO_GAITS else 0.5
        alpha = 1.0 if lw > 1 else 0.3
        marker = "o" if lw > 1 else None
        ms = 3 if lw > 1 else 0
        label = short_name(name) if lw > 1 else None
        ax.plot(windows, dxs, color=color, lw=lw, alpha=alpha, marker=marker,
                markersize=ms, label=label)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("DX (m)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "emb_fig01_displacement_curves.png")


def fig02_speed_emergence(all_results):
    """Plot speed, heading consistency, and efficiency emergence (3-panel figure).

    Tracks how locomotion quality metrics develop across cumulative time
    windows, comparing evolved gaits to random controls.

    Args:
        all_results: Dict mapping gait name to result dict.

    Side effects:
        Saves emb_fig02_speed_emergence.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Speed & Efficiency Emergence", fontsize=13, fontweight="bold")

    metrics = [
        ("mean_speed", "Mean Speed (m/s)", axes[0]),
        ("heading_consistency", "Heading Consistency", axes[1]),
        ("efficiency", "Distance / Work", axes[2]),
    ]

    for metric_key, ylabel, ax in metrics:
        ax.set_title(ylabel, fontsize=10)
        for name, result in all_results.items():
            windows = [a["window_seconds"] for a in result["analytics"]]
            vals = [a[metric_key] for a in result["analytics"]]
            color = get_color(name)
            lw = 2.0 if name in NAMED_GAITS or name in ZOO_GAITS else 0.5
            alpha = 1.0 if lw > 1 else 0.3
            label = short_name(name) if (lw > 1 and ax == axes[0]) else None
            ax.plot(windows, vals, color=color, lw=lw, alpha=alpha, label=label)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        clean_ax(ax)
    axes[0].legend(fontsize=6, loc="best", ncol=2)

    fig.tight_layout()
    save_fig(fig, "emb_fig02_speed_emergence.png")


def fig03_contact_entropy(all_results):
    """Plot contact pattern development over time (3-panel figure).

    Shows contact entropy (Shannon entropy over 8 contact states) and
    per-leg duty cycle emergence across cumulative time windows.

    Args:
        all_results: Dict mapping gait name to result dict.

    Side effects:
        Saves emb_fig03_contact_entropy.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Contact Pattern Development", fontsize=13, fontweight="bold")

    metrics = [
        ("contact_entropy_bits", "Contact Entropy (bits)", axes[0]),
        ("duty_back", "BackLeg Duty Cycle", axes[1]),
        ("duty_front", "FrontLeg Duty Cycle", axes[2]),
    ]

    for metric_key, ylabel, ax in metrics:
        ax.set_title(ylabel, fontsize=10)
        for name, result in all_results.items():
            windows = [a["window_seconds"] for a in result["analytics"]]
            vals = [a[metric_key] for a in result["analytics"]]
            color = get_color(name)
            lw = 2.0 if name in NAMED_GAITS or name in ZOO_GAITS else 0.5
            alpha = 1.0 if lw > 1 else 0.3
            label = short_name(name) if (lw > 1 and ax == axes[0]) else None
            ax.plot(windows, vals, color=color, lw=lw, alpha=alpha, label=label)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        clean_ax(ax)
    axes[0].legend(fontsize=6, loc="best", ncol=2)

    fig.tight_layout()
    save_fig(fig, "emb_fig03_contact_entropy.png")


def fig04_phase_lock(all_results):
    """Plot coordination development over time (3-panel figure).

    Shows phase lock score, back-leg frequency, and front-leg frequency
    emergence across cumulative time windows. Phase lock measures how
    consistently the two joints maintain a fixed phase relationship.

    Args:
        all_results: Dict mapping gait name to result dict.

    Side effects:
        Saves emb_fig04_phase_lock.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Coordination Development", fontsize=13, fontweight="bold")

    metrics = [
        ("phase_lock_score", "Phase Lock Score", axes[0]),
        ("freq0_hz", "BackLeg Freq (Hz)", axes[1]),
        ("freq1_hz", "FrontLeg Freq (Hz)", axes[2]),
    ]

    for metric_key, ylabel, ax in metrics:
        ax.set_title(ylabel, fontsize=10)
        for name, result in all_results.items():
            windows = [a["window_seconds"] for a in result["analytics"]]
            vals = [a[metric_key] for a in result["analytics"]]
            color = get_color(name)
            lw = 2.0 if name in NAMED_GAITS or name in ZOO_GAITS else 0.5
            alpha = 1.0 if lw > 1 else 0.3
            label = short_name(name) if (lw > 1 and ax == axes[0]) else None
            ax.plot(windows, vals, color=color, lw=lw, alpha=alpha, label=label)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        clean_ax(ax)
    axes[0].legend(fontsize=6, loc="best", ncol=2)

    fig.tight_layout()
    save_fig(fig, "emb_fig04_phase_lock.png")


def fig05_developmental_fingerprints(all_results):
    """Plot radar/fingerprint charts showing behavioral profiles at 3 time windows.

    For each key gait, draws a polar (radar) chart with 6 normalized metrics.
    Three overlaid polygons show the early, mid, and final developmental
    stages, revealing how the behavioral profile "fills in" over time.

    Metrics shown: speed, heading consistency, contact entropy, phase lock,
    back-leg duty cycle, and front-leg duty cycle.

    Args:
        all_results: Dict mapping gait name to result dict.

    Side effects:
        Saves emb_fig05_developmental_fingerprints.png to PLOT_DIR.
    """
    # Select key gaits only
    key_gaits = ["Novelty Champion", "Trial 3", "43_hidden_cpg_champion",
                 "22_curie_amplified", "5_pelton", "1_original"]
    key_gaits = [g for g in key_gaits if g in all_results]

    # Select 3 time windows: early, mid, final
    window_idxs = [1, 4, -1]  # ~100 steps, ~1000 steps, ~4000 steps
    window_labels = ["Early (~0.4s)", "Mid (~4.2s)", "Final (~16.7s)"]

    # Metrics to show on radar
    radar_metrics = ["mean_speed", "heading_consistency", "contact_entropy_bits",
                     "phase_lock_score", "duty_back", "duty_front"]
    radar_labels = ["Speed", "Heading\nConsist.", "Contact\nEntropy",
                    "Phase\nLock", "Back\nDuty", "Front\nDuty"]

    # Normalize each metric to [0, 1] across all gaits and windows so that
    # radar axes are comparable. Collect global min/max per metric first.
    all_vals = {m: [] for m in radar_metrics}
    for name in key_gaits:
        for wi in window_idxs:
            a = all_results[name]["analytics"][wi]
            for m in radar_metrics:
                all_vals[m].append(a[m])

    ranges = {}
    for m in radar_metrics:
        v = all_vals[m]
        lo, hi = min(v), max(v)
        # Guard against zero-span to avoid division by zero
        ranges[m] = (lo, hi - lo if (hi - lo) > EPS else 1.0)

    n_gaits = len(key_gaits)
    n_cols = min(n_gaits, 3)
    n_rows = (n_gaits + n_cols - 1) // n_cols

    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                                   subplot_kw=dict(projection="polar"))
    fig.suptitle("Developmental Fingerprints", fontsize=13, fontweight="bold", y=1.02)

    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes_grid]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = list(axes_grid.flat) if hasattr(axes_grid, 'flat') else [axes_grid]
    else:
        axes_flat = list(axes_grid.flat)

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    alphas = [0.2, 0.5, 0.9]
    lws = [1, 1.5, 2]

    for gi, gait_name in enumerate(key_gaits):
        ax = axes_flat[gi]
        ax.set_title(short_name(gait_name), fontsize=9, pad=15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticklabels([])

        color = get_color(gait_name)
        for ti, wi in enumerate(window_idxs):
            a = all_results[gait_name]["analytics"][wi]
            values = []
            for m in radar_metrics:
                lo, span = ranges[m]
                values.append((a[m] - lo) / span)
            values += values[:1]
            ax.plot(angles, values, color=color, lw=lws[ti], alpha=alphas[ti],
                    label=window_labels[ti])
            ax.fill(angles, values, color=color, alpha=alphas[ti] * 0.2)

        if gi == 0:
            ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Hide unused axes
    for gi in range(n_gaits, len(axes_flat)):
        axes_flat[gi].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "emb_fig05_developmental_fingerprints.png")


def fig06_onset_summary(all_results):
    """Plot a 4-panel summary of gait onset and maturation.

    Top-left: DX accumulation rate over time (finite-difference speed).
    Top-right: Phase lock onset times as horizontal bar chart.
    Bottom-left: Contact entropy onset times as horizontal bar chart.
    Bottom-right: Box plots comparing evolved vs random |DX| at 3 windows.

    Args:
        all_results: Dict mapping gait name to result dict.

    Side effects:
        Saves emb_fig06_onset_summary.png to PLOT_DIR.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Gait Onset & Maturation Summary", fontsize=13, fontweight="bold")

    # Top-left: DX rate (instantaneous DX/dt approximation per window)
    ax = axes[0, 0]
    ax.set_title("DX Accumulation Rate (m/s per window)", fontsize=10)
    for name, result in all_results.items():
        if name.startswith("Random"):
            continue
        analytics = result["analytics"]
        windows = [a["window_seconds"] for a in analytics]
        dxs = [a["dx"] for a in analytics]
        # Finite-difference DX rate: how much new displacement accrued per
        # second between consecutive windows (approximates instantaneous speed).
        rates = []
        rate_times = []
        for j in range(1, len(dxs)):
            dt_w = windows[j] - windows[j-1]
            if dt_w > 0:
                rates.append((dxs[j] - dxs[j-1]) / dt_w)
                rate_times.append((windows[j] + windows[j-1]) / 2)  # midpoint
        color = get_color(name)
        ax.plot(rate_times, rates, "o-", color=color, lw=1.5, markersize=4,
                label=short_name(name))
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("DX rate (m/s)", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    clean_ax(ax)

    # Top-right: Phase lock onset bar chart
    ax = axes[0, 1]
    ax.set_title("Phase Lock Onset (80% of final)", fontsize=10)
    named_gaits_list = [n for n in all_results if not n.startswith("Random")]
    onset_data = []
    for name in named_gaits_list:
        analytics = all_results[name]["analytics"]
        onset_steps, onset_sec = compute_onset_time(analytics, "phase_lock_score", 0.8)
        onset_data.append((name, onset_sec))
    onset_data.sort(key=lambda x: x[1] if x[1] is not None else 999)
    bar_names = [short_name(n) for n, _ in onset_data]
    bar_vals = [s if s is not None else 0 for _, s in onset_data]
    bar_colors = [get_color(n) for n, _ in onset_data]
    bars = ax.barh(range(len(bar_names)), bar_vals, color=bar_colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(bar_names)))
    ax.set_yticklabels(bar_names, fontsize=7)
    ax.set_xlabel("Onset time (s)", fontsize=9)
    ax.invert_yaxis()
    clean_ax(ax)

    # Bottom-left: Contact entropy onset
    ax = axes[1, 0]
    ax.set_title("Contact Entropy Onset (80% of final)", fontsize=10)
    onset_data2 = []
    for name in named_gaits_list:
        analytics = all_results[name]["analytics"]
        onset_steps, onset_sec = compute_onset_time(analytics, "contact_entropy_bits", 0.8)
        onset_data2.append((name, onset_sec))
    onset_data2.sort(key=lambda x: x[1] if x[1] is not None else 999)
    bar_names = [short_name(n) for n, _ in onset_data2]
    bar_vals = [s if s is not None else 0 for _, s in onset_data2]
    bar_colors = [get_color(n) for n, _ in onset_data2]
    ax.barh(range(len(bar_names)), bar_vals, color=bar_colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(bar_names)))
    ax.set_yticklabels(bar_names, fontsize=7)
    ax.set_xlabel("Onset time (s)", fontsize=9)
    ax.invert_yaxis()
    clean_ax(ax)

    # Bottom-right: Random vs evolved comparison (box plot)
    ax = axes[1, 1]
    ax.set_title("Random vs Evolved: DX at Key Windows", fontsize=10)
    evolved_names = [n for n in all_results
                     if not n.startswith("Random") and n != "Trial 3"]
    random_names = [n for n in all_results if n.startswith("Random")]
    # Pick 3 windows
    w_indices = [2, 4, -1]  # 200, 1000, 4000 steps
    w_labels = ["200 steps\n(0.8s)", "1000 steps\n(4.2s)", "4000 steps\n(16.7s)"]
    positions = [0, 1, 2]

    for wi_offset, (w_idx, w_label) in enumerate(zip(w_indices, w_labels)):
        # Collect |DX| for evolved and random gaits at this time window
        evolved_dxs = []
        for name in evolved_names:
            a = all_results[name]["analytics"][w_idx]
            evolved_dxs.append(abs(a["dx"]))
        random_dxs = []
        for name in random_names:
            a = all_results[name]["analytics"][w_idx]
            random_dxs.append(abs(a["dx"]))

        # Space box pairs with a gap: evolved at x_pos, random at x_pos+1
        x_pos = wi_offset * 3
        bp1 = ax.boxplot([evolved_dxs], positions=[x_pos], widths=0.8,
                          patch_artist=True, showfliers=False)
        bp1["boxes"][0].set_facecolor("#348ABD")
        bp1["boxes"][0].set_alpha(0.6)
        bp2 = ax.boxplot([random_dxs], positions=[x_pos + 1], widths=0.8,
                          patch_artist=True, showfliers=False)
        bp2["boxes"][0].set_facecolor("#BBBBBB")
        bp2["boxes"][0].set_alpha(0.6)

    ax.set_xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels(w_labels, fontsize=8)
    ax.set_ylabel("|DX| (m)", fontsize=9)
    # Manual legend
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="#348ABD", alpha=0.6),
               Patch(facecolor="#BBBBBB", alpha=0.6)],
              ["Evolved", "Random"], fontsize=8, loc="upper left")
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, "emb_fig06_onset_summary.png")


# ── JSON encoder ────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types and ndarrays.

    Converts numpy integers, floats, booleans, and ndarrays to their
    Python equivalents so json.dump() can serialize analytics dicts
    that contain numpy values.
    """

    def default(self, obj):
        """Convert numpy types to JSON-serializable Python types.

        Args:
            obj: Object that the default encoder cannot handle.

        Returns:
            Python-native equivalent of the numpy type.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 6)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    """Run the full behavioral embryology pipeline: simulate, analyze, plot.

    Executes all four parts in sequence:
      Part 1 - Simulate named gaits with per-step trajectory capture.
      Part 2 - Load zoo gaits from existing telemetry JSONL files.
      Part 3 - Simulate random-weight controls as a baseline.
      Analysis - Print onset times, DX comparisons, and evolved vs random stats.
      Figures - Generate 6 publication-quality figures.
      JSON - Save all windowed analytics to artifacts/behavioral_embryology.json.

    Side effects:
        Writes brain.nndf (overwritten per gait), 6 PNG figures, and 1 JSON file.
        Prints a detailed console summary of developmental analysis results.
    """
    print("Behavioral Embryology — Gait Development Analysis")
    print(f"  Time windows: {WINDOWS}")
    print(f"  Named gaits: {list(NAMED_GAITS.keys())}")
    print(f"  Zoo gaits: {ZOO_GAITS}")
    print(f"  Random controls: {N_RANDOM}")
    print()

    all_results = {}

    # ── Part 1: Simulate named gaits ────────────────────────────────────────
    print("=" * 80)
    print("PART 1: Simulating Named Gaits")
    print("=" * 80)
    t0 = time.time()

    for gait_name, weights in NAMED_GAITS.items():
        t_start = time.time()
        data = simulate_full_trajectory(weights)
        elapsed = time.time() - t_start
        dx = data["x"][-1] - data["x"][0]
        print(f"  {gait_name}: DX={dx:+.2f}m  ({elapsed:.1f}s)")

        # Compute windowed analytics
        analytics = []
        for w in WINDOWS:
            a = compute_windowed_analytics(data, w)
            analytics.append(a)

        all_results[gait_name] = {
            "data": data,
            "analytics": analytics,
            "source": "simulated",
        }

    print(f"  Part 1: {len(NAMED_GAITS)} gaits in {time.time()-t0:.1f}s")

    # ── Part 2: Load zoo gaits from telemetry ───────────────────────────────
    print()
    print("=" * 80)
    print("PART 2: Loading Zoo Gaits from Telemetry")
    print("=" * 80)
    t0 = time.time()

    for gait_name in ZOO_GAITS:
        try:
            data = load_telemetry(gait_name)
            dx = data["x"][-1] - data["x"][0]
            print(f"  {gait_name}: DX={dx:+.2f}m  (loaded {len(data['t'])} steps)")

            analytics = []
            for w in WINDOWS:
                w_actual = min(w, len(data["t"]))
                a = compute_windowed_analytics(data, w_actual)
                analytics.append(a)

            all_results[gait_name] = {
                "data": data,
                "analytics": analytics,
                "source": "telemetry",
            }
        except FileNotFoundError as e:
            print(f"  SKIP {gait_name}: {e}")

    print(f"  Part 2: {sum(1 for r in all_results.values() if r['source']=='telemetry')} "
          f"gaits loaded in {time.time()-t0:.1f}s")

    # ── Part 3: Simulate random controls ────────────────────────────────────
    print()
    print("=" * 80)
    print("PART 3: Simulating Random Controls")
    print("=" * 80)
    t0 = time.time()
    rng = np.random.RandomState(RNG_SEED)

    for ri in range(N_RANDOM):
        weights = {wn: rng.uniform(-2, 2) for wn in WEIGHT_NAMES}
        t_start = time.time()
        data = simulate_full_trajectory(weights)
        elapsed = time.time() - t_start
        dx = data["x"][-1] - data["x"][0]

        name = f"Random_{ri:02d}"
        if (ri + 1) % 5 == 0 or ri == 0:
            print(f"  [{ri+1:2d}/{N_RANDOM}] {elapsed:.1f}s  DX={dx:+.2f}m")

        analytics = []
        for w in WINDOWS:
            a = compute_windowed_analytics(data, w)
            analytics.append(a)

        all_results[name] = {
            "data": data,
            "analytics": analytics,
            "source": "random",
            "weights": weights,
        }

    print(f"  Part 3: {N_RANDOM} random gaits in {time.time()-t0:.1f}s")

    # ── Analysis ────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("DEVELOPMENTAL ANALYSIS")
    print("=" * 80)

    # Onset times
    print("\n  ONSET TIMES (80% of final value):")
    print(f"  {'Gait':30s} {'DX':>8s} {'Speed':>8s} {'Phase':>8s} {'Entropy':>8s}")
    print("  " + "-" * 72)
    for name, result in all_results.items():
        if name.startswith("Random"):
            continue
        analytics = result["analytics"]
        dx_onset = compute_onset_time(analytics, "dx", 0.8)
        speed_onset = compute_onset_time(analytics, "mean_speed", 0.8)
        phase_onset = compute_onset_time(analytics, "phase_lock_score", 0.8)
        entropy_onset = compute_onset_time(analytics, "contact_entropy_bits", 0.8)

        dx_str = f"{dx_onset[1]:.1f}s" if dx_onset[1] is not None else "N/A"
        speed_str = f"{speed_onset[1]:.1f}s" if speed_onset[1] is not None else "N/A"
        phase_str = f"{phase_onset[1]:.1f}s" if phase_onset[1] is not None else "N/A"
        entropy_str = f"{entropy_onset[1]:.1f}s" if entropy_onset[1] is not None else "N/A"
        print(f"  {short_name(name):30s} {dx_str:>8s} {speed_str:>8s} "
              f"{phase_str:>8s} {entropy_str:>8s}")

    # Final DX comparison
    print("\n  FINAL DX (4000 steps):")
    final_dxs = []
    for name, result in all_results.items():
        dx = result["analytics"][-1]["dx"]
        final_dxs.append((abs(dx), dx, name))
    final_dxs.sort(reverse=True)
    for abs_dx, dx, name in final_dxs[:10]:
        tag = "[RANDOM]" if name.startswith("Random") else ""
        print(f"    {short_name(name):30s} DX={dx:+8.2f}m  {tag}")

    # Random vs evolved stats
    evolved_final = [abs(all_results[n]["analytics"][-1]["dx"])
                     for n in all_results
                     if not n.startswith("Random") and n != "Trial 3"]
    random_final = [abs(all_results[n]["analytics"][-1]["dx"])
                    for n in all_results if n.startswith("Random")]
    print(f"\n  EVOLVED vs RANDOM:")
    print(f"    Evolved mean |DX|: {np.mean(evolved_final):.2f}m "
          f"(n={len(evolved_final)})")
    print(f"    Random  mean |DX|: {np.mean(random_final):.2f}m "
          f"(n={len(random_final)})")

    # Early development comparison (at 500 steps)
    w_500_idx = WINDOWS.index(500) if 500 in WINDOWS else 3
    evolved_early = [abs(all_results[n]["analytics"][w_500_idx]["dx"])
                     for n in all_results
                     if not n.startswith("Random") and n != "Trial 3"]
    random_early = [abs(all_results[n]["analytics"][w_500_idx]["dx"])
                    for n in all_results if n.startswith("Random")]
    print(f"\n  AT 500 STEPS (~2.1s):")
    print(f"    Evolved mean |DX|: {np.mean(evolved_early):.2f}m")
    print(f"    Random  mean |DX|: {np.mean(random_early):.2f}m")

    # Phase lock comparison
    evolved_phase = [all_results[n]["analytics"][-1]["phase_lock_score"]
                     for n in all_results
                     if not n.startswith("Random") and n != "Trial 3"]
    random_phase = [all_results[n]["analytics"][-1]["phase_lock_score"]
                    for n in all_results if n.startswith("Random")]
    print(f"\n  PHASE LOCK (final):")
    print(f"    Evolved mean: {np.mean(evolved_phase):.3f}")
    print(f"    Random  mean: {np.mean(random_phase):.3f}")

    # ── Figures ─────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    fig01_displacement_curves(all_results)
    fig02_speed_emergence(all_results)
    fig03_contact_entropy(all_results)
    fig04_phase_lock(all_results)
    fig05_developmental_fingerprints(all_results)
    fig06_onset_summary(all_results)

    # ── Save JSON ───────────────────────────────────────────────────────────
    json_out = {
        "windows": WINDOWS,
        "windows_seconds": [round(w * DT, 4) for w in WINDOWS],
        "gaits": {},
    }
    for name, result in all_results.items():
        entry = {
            "source": result["source"],
            "analytics": result["analytics"],
            "final_dx": result["analytics"][-1]["dx"],
        }
        if "weights" in result:
            entry["weights"] = result["weights"]
        json_out["gaits"][name] = entry

    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2, cls=NumpyEncoder)
    print(f"\nWROTE {OUT_JSON}")

    total_sims = len(NAMED_GAITS) + N_RANDOM
    print(f"\nTotal: {total_sims} simulations + {sum(1 for r in all_results.values() if r['source']=='telemetry')} loaded from telemetry")


if __name__ == "__main__":
    main()
