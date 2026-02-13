#!/usr/bin/env python3
"""
generate_telemetry.py

Role:
    Batch telemetry generator for all 116 zoo gaits. Runs each gait as a
    headless PyBullet simulation at full resolution (every=1, 4000 records
    at 240 Hz) and writes per-step telemetry to disk.

    For each gait, the script writes the appropriate brain.nndf (handling
    standard, crosswired, and hidden architectures), runs the simulation
    loop matching the canonical control path (Act -> Step -> Think), and
    delegates recording to TelemetryLogger.

Output per gait (in artifacts/telemetry/<gait_name>/):
    telemetry.jsonl  -- 4000 JSONL records (one per sim step)
    summary.json     -- displacement, stability metrics

Notes:
    - brain.nndf is a shared file overwritten per gait. The script backs up
      the original before the batch and restores it afterward.
    - Motor commands are sent as raw NN tanh outputs (no scaling/offset) to
      match the control path used in walker_competition.py and record_videos.py.
    - Estimated disk usage: ~1.5 MB per gait (~175 MB total for 116 gaits).

Usage:
    python3 generate_telemetry.py              # all 116 gaits
    python3 generate_telemetry.py --gait 18_curie  # single gait
    python3 generate_telemetry.py --dry-run    # show what would run
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pybullet as p
import pybullet_data
import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import constants as c
from pyrosim.neuralNetwork import NEURAL_NETWORK
import pyrosim.pyrosim as pyrosim
from tools.telemetry.logger import TelemetryLogger


def write_brain_crosswired(w03, w13, w23, w04, w14, w24,
                           w34=0.0, w43=0.0, w33=0.0, w44=0.0):
    """Write brain.nndf for a standard or crosswired topology.

    Creates an NNDF file with 3 sensor neurons (Torso, BackLeg, FrontLeg)
    and 2 motor neurons (Torso_BackLeg, Torso_FrontLeg). The 6 sensor-to-motor
    weights are always included; the 4 motor-to-motor weights (crosswired) are
    only written when non-zero.

    Args:
        w03: Sensor 0 (Torso) -> Motor 3 (BackLeg) weight.
        w13: Sensor 1 (BackLeg) -> Motor 3 (BackLeg) weight.
        w23: Sensor 2 (FrontLeg) -> Motor 3 (BackLeg) weight.
        w04: Sensor 0 (Torso) -> Motor 4 (FrontLeg) weight.
        w14: Sensor 1 (BackLeg) -> Motor 4 (FrontLeg) weight.
        w24: Sensor 2 (FrontLeg) -> Motor 4 (FrontLeg) weight.
        w34: Motor 3 -> Motor 4 cross-connection weight.
        w43: Motor 4 -> Motor 3 cross-connection weight.
        w33: Motor 3 self-feedback weight.
        w44: Motor 4 self-feedback weight.

    Side effects:
        Overwrites PROJECT/brain.nndf.
    """
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        f.write('    <neuron name = "0" type = "sensor" linkName = "Torso" />\n')
        f.write('    <neuron name = "1" type = "sensor" linkName = "BackLeg" />\n')
        f.write('    <neuron name = "2" type = "sensor" linkName = "FrontLeg" />\n')
        f.write('    <neuron name = "3" type = "motor"  jointName = "Torso_BackLeg" />\n')
        f.write('    <neuron name = "4" type = "motor"  jointName = "Torso_FrontLeg" />\n')
        for src, tgt, w in [("0","3",w03), ("1","3",w13), ("2","3",w23),
                             ("0","4",w04), ("1","4",w14), ("2","4",w24),
                             ("3","4",w34), ("4","3",w43), ("3","3",w33), ("4","4",w44)]:
            if w != 0.0:
                f.write(f'    <synapse sourceNeuronName = "{src}" targetNeuronName = "{tgt}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def write_brain_full(neurons, synapses):
    """Write brain.nndf for an arbitrary topology (including hidden neurons).

    Supports any combination of sensor, motor, and hidden neurons with
    arbitrary synapse connectivity, as used by the 'hidden' architecture
    gaits in the zoo.

    Args:
        neurons: List of dicts, each with keys 'id', 'type' ('sensor'/'motor'/'hidden'),
            and 'ref' (linkName for sensors, jointName for motors; absent for hidden).
        synapses: List of dicts, each with keys 'src' (source neuron id),
            'tgt' (target neuron id), and 'w' (weight). Zero-weight synapses
            are skipped.

    Side effects:
        Overwrites PROJECT/brain.nndf.
    """
    path = PROJECT / "brain.nndf"
    with open(path, "w") as f:
        f.write('<neuralNetwork>\n')
        for neuron in neurons:
            nid, ntype, ref = neuron["id"], neuron["type"], neuron.get("ref")
            if ntype == "sensor":
                f.write(f'    <neuron name = "{nid}" type = "sensor" linkName = "{ref}" />\n')
            elif ntype == "motor":
                f.write(f'    <neuron name = "{nid}" type = "motor"  jointName = "{ref}" />\n')
            else:
                f.write(f'    <neuron name = "{nid}" type = "hidden" />\n')
        for syn in synapses:
            w = syn["w"]
            if w != 0.0:
                f.write(f'    <synapse sourceNeuronName = "{syn["src"]}" targetNeuronName = "{syn["tgt"]}" weight = "{w}" />\n')
        f.write('</neuralNetwork>\n')


def safe_get_base_pose(body_id):
    """Retrieve the robot's base position and orientation, with a safe fallback.

    Guards against PyBullet errors (e.g., disconnected session or invalid body id)
    by returning an identity pose at the origin.

    Args:
        body_id: PyBullet body unique id.

    Returns:
        Tuple of (position, orientation) where position is (x, y, z) and
        orientation is a quaternion (x, y, z, w). Falls back to origin with
        identity rotation on any error.
    """
    try:
        return p.getBasePositionAndOrientation(body_id)
    except Exception:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)


def run_gait(gait_name, gait_data, out_dir):
    """Run one gait in headless mode and write full-resolution telemetry.

    Sets up a complete PyBullet simulation from scratch: writes brain.nndf
    for the gait's architecture, loads the robot body, initializes friction
    and the neural network, then runs the canonical Act -> Step -> Think
    loop for SIM_STEPS iterations with telemetry recording at every step.

    Args:
        gait_name: Human-readable gait identifier (e.g., '18_curie').
        gait_data: Gait definition dict from synapse_gait_zoo.json containing
            weights/neurons/synapses and architecture metadata.
        out_dir: Path where telemetry.jsonl and summary.json will be written.

    Returns:
        Tuple of (x, y, z) final base position of the robot.

    Side effects:
        Overwrites brain.nndf.
        Creates out_dir if needed and writes telemetry files.
        Creates and destroys a PyBullet DIRECT-mode physics session.
    """
    # Write brain.nndf
    arch = gait_data.get("architecture", "standard_6")
    if arch == "hidden":
        write_brain_full(gait_data["neurons"], gait_data["synapses"])
    else:
        w = gait_data["weights"]
        write_brain_crosswired(
            w.get("w03", 0.0), w.get("w13", 0.0), w.get("w23", 0.0),
            w.get("w04", 0.0), w.get("w14", 0.0), w.get("w24", 0.0),
            w.get("w34", 0.0), w.get("w43", 0.0), w.get("w33", 0.0), w.get("w44", 0.0),
        )

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

    # Telemetry
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetryLogger(
        robotId, out_dir, every=1,
        variant_id=gait_name, run_id="definitive",
        enabled=True,
    )

    max_force = float(getattr(c, "MAX_FORCE", 150.0))

    for i in range(c.SIM_STEPS):
        # Act: send raw NN motor outputs to joints (no scaling, no offsets —
        # matches record_videos.py and the control path that produced zoo measurements)
        for neuronName in nn.neurons:
            n = nn.neurons[neuronName]
            if n.Is_Motor_Neuron():
                jn = n.Get_Joint_Name()
                jn_bytes = jn.encode("ASCII") if isinstance(jn, str) else jn
                try:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, n.Get_Value(), max_force)
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robotId, jn_bytes, p.POSITION_CONTROL, n.Get_Value(), max_force)

        p.stepSimulation()

        # Think: update NN from sensors (after stepping, so sensors reflect current state)
        nn.Update()

        telemetry.log_step(i)

    telemetry.finalize()
    pos = safe_get_base_pose(robotId)[0]
    p.disconnect()
    return pos


def main():
    """Entry point: parse arguments and run telemetry generation for selected gaits.

    Supports three modes:
        Default: Generate telemetry for all 116 gaits in the zoo.
        --gait NAME: Generate telemetry for a single named gait.
        --dry-run: List gaits that would be processed without running simulations.

    Side effects:
        Backs up and restores brain.nndf around the batch run.
        Creates telemetry output directories and files under --out path.
        Prints per-gait progress and displacement to stdout.
    """
    ap = argparse.ArgumentParser(description="Generate full-resolution telemetry for zoo gaits.")
    ap.add_argument("--gait", type=str, default=None, help="Run a single gait by name")
    ap.add_argument("--dry-run", action="store_true", help="List gaits without running")
    ap.add_argument("--out", type=str, default="artifacts/telemetry",
                    help="Output root directory (default: artifacts/telemetry)")
    args = ap.parse_args()

    zoo_path = PROJECT / "synapse_gait_zoo.json"
    zoo = json.loads(zoo_path.read_text())

    # Collect all gaits in zoo order
    gaits = []
    for cat_name, cat in zoo["categories"].items():
        for gait_name, gait_data in cat["gaits"].items():
            gaits.append((gait_name, gait_data, cat_name))

    if args.gait:
        gaits = [(n, d, c) for n, d, c in gaits if n == args.gait]
        if not gaits:
            print(f"Gait '{args.gait}' not found in zoo.", file=sys.stderr)
            sys.exit(1)

    out_root = Path(args.out)

    if args.dry_run:
        print(f"Would generate full-resolution telemetry for {len(gaits)} gaits:")
        for name, data, cat in gaits:
            arch = data.get("architecture", "standard_6")
            print(f"  {name} ({arch}, {cat})")
        print(f"\nOutput: {out_root}/<gait_name>/telemetry.jsonl (4000 records each)")
        print(f"Estimated disk: ~{len(gaits) * 1.5:.0f} MB")
        return

    # Back up brain.nndf
    brain_path = PROJECT / "brain.nndf"
    backup_path = PROJECT / "brain.nndf.backup"
    if brain_path.exists():
        shutil.copy2(brain_path, backup_path)

    print(f"Generating full-resolution telemetry for {len(gaits)} gaits...")
    t_start = time.perf_counter()

    for idx, (gait_name, gait_data, cat_name) in enumerate(gaits):
        gait_out = out_root / gait_name
        t0 = time.perf_counter()
        pos = run_gait(gait_name, gait_data, gait_out)
        elapsed = time.perf_counter() - t0
        dx = pos[0]
        print(f"  [{idx+1:3d}/{len(gaits)}] {gait_name:40s} DX={dx:+7.2f}  ({elapsed:.2f}s)")

    total = time.perf_counter() - t_start

    # Restore brain.nndf
    if backup_path.exists():
        shutil.copy2(backup_path, brain_path)

    print(f"\nDone. {len(gaits)} gaits in {total:.1f}s")
    print(f"Output: {out_root}/")


if __name__ == "__main__":
    main()
