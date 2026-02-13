#!/usr/bin/env python3
"""
compute_beer_analytics.py

Beer-Framework Analytics Pipeline for Synapse Gait Zoo v2.

Reads all 116 telemetry JSONL files, computes 4 pillars of metrics
(outcome, contact, coordination, rotation_axis), and writes
synapse_gait_zoo_v2.json preserving all existing gait fields but
replacing the old `telemetry` object with a comprehensive `analytics` object.

Constraint: numpy-only (no scipy, no sklearn).

Usage:
    python compute_beer_analytics.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

FS = 240            # sampling frequency (Hz)
DT = 1.0 / FS       # timestep (seconds)
EPS = 1e-12          # guard against division by zero

PROJECT = Path(__file__).resolve().parent
ZOO_IN = PROJECT / "synapse_gait_zoo.json"
ZOO_OUT = PROJECT / "synapse_gait_zoo_v2.json"
TELEMETRY_DIR = PROJECT / "artifacts" / "telemetry"

# Keys to preserve from old telemetry → analytics.preserved
PRESERVE_KEYS = ("attractor_type", "attractor_subtype", "pareto_optimal")


# ── JSON encoder for numpy types ─────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalars/arrays; round floats to 6 decimals."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 6)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return self._convert_array(obj)
        return super().default(obj)

    def _convert_array(self, arr):
        """Convert a numpy array to a JSON-safe nested list with rounded floats."""
        flat = arr.tolist()
        return self._round_nested(flat)

    def _round_nested(self, obj):
        """Recursively round all floats in a nested list structure to 6 decimals."""
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, list):
            return [self._round_nested(x) for x in obj]
        return obj


# ── Data loading ─────────────────────────────────────────────────────────────

def load_telemetry(gait_name):
    """Load telemetry JSONL for a gait → dict of numpy arrays."""
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
    # Pre-allocate arrays
    t = np.empty(n)
    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)
    vx = np.empty(n)
    vy = np.empty(n)
    vz = np.empty(n)
    wx = np.empty(n)
    wy = np.empty(n)
    wz = np.empty(n)
    roll = np.empty(n)
    pitch = np.empty(n)
    yaw = np.empty(n)
    contact_torso = np.empty(n, dtype=bool)
    contact_back = np.empty(n, dtype=bool)
    contact_front = np.empty(n, dtype=bool)
    j0_pos = np.empty(n)
    j0_vel = np.empty(n)
    j0_tau = np.empty(n)
    j1_pos = np.empty(n)
    j1_vel = np.empty(n)
    j1_tau = np.empty(n)

    for i, rec in enumerate(records):
        t[i] = rec["t"]
        base = rec["base"]
        x[i] = base["x"]
        y[i] = base["y"]
        z[i] = base["z"]
        vel = rec["vel"]
        vx[i] = vel["vx"]
        vy[i] = vel["vy"]
        vz[i] = vel["vz"]
        av = rec["ang_vel"]
        wx[i] = av["wx"]
        wy[i] = av["wy"]
        wz[i] = av["wz"]
        rpy = rec["rpy"]
        roll[i] = rpy["r"]
        pitch[i] = rpy["p"]
        yaw[i] = rpy["y"]
        lc = rec["link_contacts"]
        contact_torso[i] = lc[0]
        contact_back[i] = lc[1]
        contact_front[i] = lc[2]
        joints = rec["joints"]
        j0_pos[i] = joints[0]["pos"]
        j0_vel[i] = joints[0]["vel"]
        j0_tau[i] = joints[0]["tau"]
        j1_pos[i] = joints[1]["pos"]
        j1_vel[i] = joints[1]["vel"]
        j1_tau[i] = joints[1]["tau"]

    return {
        "t": t, "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy, "vz": vz,
        "wx": wx, "wy": wy, "wz": wz,
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "contact_torso": contact_torso,
        "contact_back": contact_back,
        "contact_front": contact_front,
        "j0_pos": j0_pos, "j0_vel": j0_vel, "j0_tau": j0_tau,
        "j1_pos": j1_pos, "j1_vel": j1_vel, "j1_tau": j1_tau,
    }


# ── Signal processing helpers ───────────────────────────────────────────────

def _hilbert_analytic(x):
    """
    Compute the analytic signal via FFT-based Hilbert transform.
    Returns complex array: x + i*hilbert(x).
    """
    n = len(x)
    X = np.fft.fft(x)
    # Build one-sided spectral mask H: keep DC and Nyquist as-is (×1),
    # double positive frequencies (×2), zero out negative frequencies.
    # This is the standard recipe for the analytic signal via FFT.
    H = np.zeros(n)
    H[0] = 1.0                        # DC component unchanged
    if n % 2 == 0:
        H[1:n // 2] = 2.0             # double positive-frequency bins
        H[n // 2] = 1.0               # Nyquist bin unchanged
    else:
        H[1:(n + 1) // 2] = 2.0       # double positive-frequency bins (odd-length)
    # Multiply in frequency domain and invert: result is x(t) + i·hilbert(x(t))
    return np.fft.ifft(X * H)


def _fft_peak(signal, dt):
    """
    Find dominant frequency and amplitude from FFT of a real signal.
    Returns (freq_hz, amplitude). Ignores DC component.
    """
    n = len(signal)
    # Remove mean so DC doesn't dominate the spectrum
    sig = signal - np.mean(signal)
    if np.max(np.abs(sig)) < EPS:
        return 0.0, 0.0

    freqs = np.fft.rfftfreq(n, d=dt)
    # Real-valued FFT; scale by 2/n to get single-sided amplitude spectrum
    spectrum = np.abs(np.fft.rfft(sig)) * (2.0 / n)

    # Ignore DC (index 0)
    if len(spectrum) < 2:
        return 0.0, 0.0
    spectrum[0] = 0.0
    peak_idx = np.argmax(spectrum)
    return float(freqs[peak_idx]), float(spectrum[peak_idx])


# ── Pillar 1: Outcome ───────────────────────────────────────────────────────

def compute_outcome(data, dt):
    """Compute Pillar 1 (Outcome): displacement, speed, efficiency, and path metrics.

    Returns a dict with dx, dy, yaw_net_rad, speed stats, work proxy, and path quality.
    """
    x, y = data["x"], data["y"]
    vx, vy = data["vx"], data["vy"]
    wz = data["wz"]
    j0_tau, j0_vel = data["j0_tau"], data["j0_vel"]
    j1_tau, j1_vel = data["j1_tau"], data["j1_vel"]

    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])

    # Yaw via integration of angular velocity (handles multi-turn)
    yaw_net_rad = float(np.trapezoid(wz, dx=dt))

    # Speed: 2D ground-plane speed (ignoring vertical component)
    speed = np.sqrt(vx**2 + vy**2)
    mean_speed = float(np.mean(speed))
    speed_std = float(np.std(speed))
    # Coefficient of variation: higher = more irregular speed profile
    speed_cv = float(speed_std / mean_speed) if mean_speed > EPS else 0.0

    # Work proxy: sum of absolute joint power (|torque × angular velocity|) over time
    instantaneous_power = np.abs(j0_tau * j0_vel) + np.abs(j1_tau * j1_vel)
    work_proxy = float(np.sum(instantaneous_power) * dt)

    # Distance per work: locomotion efficiency (meters per unit of mechanical work)
    net_dist = np.sqrt(dx**2 + dy**2)
    distance_per_work = float(net_dist / work_proxy) if work_proxy > EPS else 0.0

    # Path straightness: net_displacement / path_length  (1.0 = perfectly straight)
    # Summing infinitesimal arc lengths from consecutive XY positions
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    path_length = float(np.sum(ds))
    path_straightness = float(net_dist / path_length) if path_length > EPS else 0.0

    # Heading consistency: mean resultant length of unit heading vectors.
    # Maps each velocity to a point on the unit circle e^{iθ}; the magnitude
    # of their average is 1.0 for perfectly consistent heading, 0.0 for random.
    # Only count timesteps where the robot is actually moving (speed > threshold)
    speed_thresh = 0.1  # m/s — ignore near-stationary frames
    moving = speed > speed_thresh
    if np.sum(moving) > 10:
        theta = np.arctan2(vy[moving], vx[moving])
        heading_consistency = float(np.abs(np.mean(np.exp(1j * theta))))
    else:
        heading_consistency = 0.0

    return {
        "dx": dx,
        "dy": dy,
        "yaw_net_rad": yaw_net_rad,
        "mean_speed": mean_speed,
        "speed_cv": speed_cv,
        "work_proxy": work_proxy,
        "distance_per_work": distance_per_work,
        "path_length": path_length,
        "path_straightness": path_straightness,
        "heading_consistency": heading_consistency,
    }


# ── Pillar 2: Contact ───────────────────────────────────────────────────────

def compute_contact(data):
    """Compute Pillar 2 (Contact): duty fractions, state distribution, entropy, transitions.

    Returns a dict with per-link duty cycles, 8-state distribution, Shannon entropy,
    and an 8x8 row-normalized transition matrix.
    """
    torso = data["contact_torso"].astype(int)
    back = data["contact_back"].astype(int)
    front = data["contact_front"].astype(int)
    n = len(torso)

    # Duty fractions: fraction of timesteps each link is in contact with ground
    duty_torso = float(np.mean(torso))
    duty_back = float(np.mean(back))
    duty_front = float(np.mean(front))

    # Support count: how many links touching (0, 1, 2, or 3)
    support = torso + back + front
    support_count_frac = [0.0] * 4
    for k in range(4):
        support_count_frac[k] = float(np.sum(support == k) / n)

    # Encode 3 binary contact signals into a single integer 0..7.
    # Bit layout: torso=bit2 (×4), back=bit1 (×2), front=bit0 (×1).
    # E.g. state 5 = 101 = torso+front touching, back airborne.
    state = torso * 4 + back * 2 + front
    # Count occurrences of each of the 8 possible contact states
    state_counts = np.bincount(state, minlength=8)
    state_distribution = (state_counts / n).tolist()

    # Dominant state: most frequently occupied contact configuration
    dominant_state = int(np.argmax(state_counts))

    # Shannon entropy (bits): measures diversity of contact patterns.
    # Max = log2(8) = 3 bits (all states equally likely); min = 0 (single state).
    probs = state_counts / n
    nonzero = probs[probs > 0]
    contact_entropy_bits = float(-np.sum(nonzero * np.log2(nonzero)))

    # Transition matrix: 8×8 row-normalized (Markov chain of contact state changes).
    # Encodes consecutive state pairs (s_t, s_{t+1}) as a flat index s_t*8 + s_{t+1},
    # then reshapes to a matrix and normalizes rows to get transition probabilities.
    if n > 1:
        trans_idx = state[:-1] * 8 + state[1:]
        trans_flat = np.bincount(trans_idx, minlength=64)
        trans_matrix = trans_flat.reshape(8, 8).astype(float)
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero for unvisited states
        trans_matrix = trans_matrix / row_sums
    else:
        trans_matrix = np.zeros((8, 8))

    return {
        "duty_torso": duty_torso,
        "duty_back": duty_back,
        "duty_front": duty_front,
        "support_count_frac": support_count_frac,
        "state_distribution": state_distribution,
        "dominant_state": dominant_state,
        "contact_entropy_bits": contact_entropy_bits,
        "transition_matrix": trans_matrix.tolist(),
    }


# ── Pillar 3: Coordination ──────────────────────────────────────────────────

def compute_coordination(data, dt):
    """Compute Pillar 3 (Coordination): joint frequency, phase difference, and phase locking.

    Returns per-joint FFT peaks and the inter-joint phase relationship.
    """
    j0_pos = data["j0_pos"]
    j1_pos = data["j1_pos"]

    # FFT peaks for each joint: dominant oscillation frequency and amplitude
    freq0, amp0 = _fft_peak(j0_pos, dt)
    freq1, amp1 = _fft_peak(j1_pos, dt)

    # Compute instantaneous phase of each joint via the Hilbert analytic signal.
    # Mean-subtract first so the Hilbert transform captures oscillation phase,
    # not a DC offset. np.angle extracts the instantaneous phase from the
    # complex analytic signal.
    a0 = _hilbert_analytic(j0_pos - np.mean(j0_pos))
    a1 = _hilbert_analytic(j1_pos - np.mean(j1_pos))
    phi0 = np.angle(a0)
    phi1 = np.angle(a1)
    delta_phi = phi1 - phi0

    # Circular mean of phase difference: maps each delta_phi to a unit-circle
    # point, averages, then extracts the angle. This correctly handles wrapping.
    mean_delta_phi = float(np.angle(np.mean(np.exp(1j * delta_phi))))

    # Phase lock score (mean resultant length): |mean(e^{iΔφ(t)})|
    # 1.0 = joints maintain a constant phase offset (perfect locking)
    # 0.0 = phase difference drifts randomly (no coordination)
    phase_lock_score = float(np.abs(np.mean(np.exp(1j * delta_phi))))

    return {
        "joint_0": {
            "dominant_freq_hz": freq0,
            "dominant_amplitude": amp0,
        },
        "joint_1": {
            "dominant_freq_hz": freq1,
            "dominant_amplitude": amp1,
        },
        "delta_phi_rad": mean_delta_phi,
        "phase_lock_score": phase_lock_score,
    }


# ── Pillar 4: Rotation Axis ─────────────────────────────────────────────────

def compute_rotation_axis(data, dt):
    """Compute Pillar 4 (Rotation Axis): axis dominance, switching rate, and periodicity.

    Returns PCA-derived variance ratios, axis switching frequency, and per-axis FFT peaks.
    """
    wx = data["wx"]
    wy = data["wy"]
    wz = data["wz"]
    n = len(wx)

    # PCA of angular velocity via eigendecomposition of the 3×3 covariance matrix.
    # This reveals which rotation axis carries the most variance (i.e., dominates
    # the robot's rotational motion). No sklearn needed -- the covariance matrix
    # is small enough for direct eigh decomposition.
    omega = np.column_stack([wx, wy, wz])  # (n, 3)
    omega_centered = omega - omega.mean(axis=0)
    # Sample covariance matrix (3×3, symmetric positive semi-definite)
    cov = np.dot(omega_centered.T, omega_centered) / max(n - 1, 1)

    # eigh guarantees real eigenvalues for symmetric matrices
    eig_vals, _ = np.linalg.eigh(cov)
    # eigh returns eigenvalues in ascending order; reverse for descending
    eig_vals = eig_vals[::-1]
    total_var = np.sum(eig_vals)
    # Axis dominance: fraction of total angular variance along each principal axis.
    # [1,0,0] = all rotation on one axis; [0.33,0.33,0.33] = isotropic tumbling.
    if total_var > EPS:
        axis_dominance = (eig_vals / total_var).tolist()
    else:
        axis_dominance = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    # Axis switching rate: how often the instantaneously dominant rotation axis
    # changes, measured in transitions per second (Hz).
    abs_omega = np.abs(omega)  # (n, 3)
    dominant_axis = np.argmax(abs_omega, axis=1)  # per-timestep dominant axis index
    if n > 1:
        switches = np.sum(dominant_axis[1:] != dominant_axis[:-1])
        total_time = (n - 1) * dt
        axis_switching_rate_hz = float(switches / total_time) if total_time > EPS else 0.0
    else:
        axis_switching_rate_hz = 0.0

    # Periodicity: dominant FFT frequency of each angular velocity component
    roll_freq, _ = _fft_peak(wx, dt)
    pitch_freq, _ = _fft_peak(wy, dt)
    yaw_freq, _ = _fft_peak(wz, dt)

    return {
        "axis_dominance": axis_dominance,
        "axis_switching_rate_hz": axis_switching_rate_hz,
        "periodicity": {
            "roll_freq_hz": roll_freq,
            "pitch_freq_hz": pitch_freq,
            "yaw_freq_hz": yaw_freq,
        },
    }


# ── Orchestrator ─────────────────────────────────────────────────────────────

def compute_all(data, dt):
    """Compute all 4 Beer pillars and return them as a single analytics dict."""
    return {
        "outcome": compute_outcome(data, dt),
        "contact": compute_contact(data),
        "coordination": compute_coordination(data, dt),
        "rotation_axis": compute_rotation_axis(data, dt),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Entry point: load zoo JSON, compute analytics for all gaits, write v2 output."""
    t_start = time.time()

    # Load zoo JSON
    print(f"Loading {ZOO_IN} ...")
    with open(ZOO_IN) as f:
        zoo = json.load(f)

    total_gaits = 0
    errors = []
    category_count = 0

    for cat_name, cat_data in zoo["categories"].items():
        gaits = cat_data.get("gaits", {})
        if not gaits:
            continue
        category_count += 1

        for gait_name, gait in gaits.items():
            total_gaits += 1
            try:
                # Load telemetry
                data = load_telemetry(gait_name)

                # Compute analytics
                analytics = compute_all(data, DT)

                # Preserve keys from old telemetry
                old_telemetry = gait.get("telemetry", {})
                preserved = {}
                for key in PRESERVE_KEYS:
                    if key in old_telemetry:
                        preserved[key] = old_telemetry[key]
                if preserved:
                    analytics["preserved"] = preserved

                # Replace telemetry with analytics
                if "telemetry" in gait:
                    del gait["telemetry"]
                gait["analytics"] = analytics

            except Exception as e:
                errors.append((gait_name, str(e)))
                print(f"  ERROR {gait_name}: {e}", file=sys.stderr)

            if total_gaits % 20 == 0:
                print(f"  processed {total_gaits} gaits ...")

    # Update meta
    zoo["meta"]["version"] = "v2"
    zoo["meta"]["analytics_framework"] = "beer"
    zoo["meta"]["analytics_pillars"] = ["outcome", "contact", "coordination", "rotation_axis"]
    zoo["meta"]["analytics_computed"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Write output
    print(f"Writing {ZOO_OUT} ...")
    with open(ZOO_OUT, "w") as f:
        json.dump(zoo, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t_start

    # Summary
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Categories: {category_count}")
    print(f"  Gaits processed: {total_gaits}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for name, err in errors:
            print(f"    {name}: {err}")

    # Spot-check stats
    print("\n── Spot-check ──")
    for cat_data in zoo["categories"].values():
        for gait_name, gait in cat_data.get("gaits", {}).items():
            a = gait.get("analytics", {})
            o = a.get("outcome", {})
            if gait_name in ("18_curie", "44_spinner_champion", "43_hidden_cpg_champion", "19_haraway"):
                print(f"  {gait_name}:")
                print(f"    dx={o.get('dx', '?'):.3f}  dy={o.get('dy', '?'):.3f}")
                print(f"    yaw_net={o.get('yaw_net_rad', '?'):.3f} rad")
                print(f"    mean_speed={o.get('mean_speed', '?'):.3f}")
                print(f"    work_proxy={o.get('work_proxy', '?'):.1f}")
                coord = a.get("coordination", {})
                print(f"    phase_lock={coord.get('phase_lock_score', '?'):.3f}")
                ra = a.get("rotation_axis", {})
                print(f"    axis_dom={ra.get('axis_dominance', '?')}")


if __name__ == "__main__":
    main()
