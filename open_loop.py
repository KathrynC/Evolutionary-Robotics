"""
open_loop.py

Open-loop gait playback + angle-vector export.

This script generates simple sinusoidal joint angle trajectories for the back and front
leg joints, saves them to disk, and (optionally) plays them back in a headless PyBullet run.

Why "open loop"?
    There is no sensing/feedback control here: joint targets are precomputed functions
    of time. This is useful for:
        - debugging motor plumbing
        - producing deterministic motion for video rendering
        - exporting angle vectors for other tools (plots, frame rendering, etc.)

Outputs:
    - data/backAngles.npy
    - data/frontAngles.npy

Env vars:
    MAX_FORCE         : motor force limit (Newtons), default 200
    SAVE_VECTORS_ONLY : if 1, only save .npy vectors and exit without simulating

Usage:
    python3 open_loop.py
    SAVE_VECTORS_ONLY=1 python3 open_loop.py
    MAX_FORCE=400 python3 open_loop.py

Notes:
    - This script uses PyBullet DIRECT (no GUI).
    - It drives joints via pyrosim.Set_Motor_For_Joint().
    - Joint keys may be bytes or str depending on pyrosim; this file handles both.

Ludobots role:
  - Open-loop motor control utilities (e.g., sine-wave gaits).

Beyond Ludobots (this repo):
  - Gait libraries / 'zoo' variants and parameter export (verify).
"""

import os

import numpy as np
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim


# --- Simulation defaults (local to this script) ---
SIM_STEPS = 1000
DT = 1 / 240
MAX_FORCE = float(os.getenv("MAX_FORCE", "200"))

# --- Start values (tune these) ---
# Units: amplitude=radians, frequency=cycles over theta in [0, 2π], phase=radians
amplitudeBack = np.pi / 4
frequencyBack = 1
phaseOffsetBack = 0.0

amplitudeFront = np.pi / 4
frequencyFront = 1
phaseOffsetFront = 0.0

SAVE_VECTORS_ONLY = os.getenv("SAVE_VECTORS_ONLY", "0") == "1"


def main():
    """Generate open-loop motor targets, save them, and optionally run a headless playback.

    Steps:
        1) Build backAngles and frontAngles arrays of length SIM_STEPS.
        2) Save them into data/*.npy.
        3) If SAVE_VECTORS_ONLY=1: exit.
        4) Otherwise, load the URDF and apply angle targets for SIM_STEPS steps.
        5) Print the XY displacement as a simple locomotion metric.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("data", exist_ok=True)

    i = np.arange(SIM_STEPS)
    theta = 2.0 * np.pi * i / (SIM_STEPS - 1)

    backAngles = amplitudeBack * np.sin(frequencyBack * theta + phaseOffsetBack)
    frontAngles = amplitudeFront * np.sin(frequencyFront * theta + phaseOffsetFront)

    np.save("data/backAngles.npy", backAngles)
    np.save("data/frontAngles.npy", frontAngles)

    if SAVE_VECTORS_ONLY:
        print("Saved motor vectors to data/backAngles.npy and data/frontAngles.npy; exiting.")
        return

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(DT)

    p.loadURDF("plane.urdf")
    if os.path.exists("world.sdf"):
        try:
            p.loadSDF("world.sdf")
        except Exception as e:
            if os.getenv('SIM_DEBUG','0') == '1':
                print('[WARN]', __name__, 'suppressed exception:', repr(e), flush=True)

    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    # pyrosim joint name keys may be bytes or str depending on version/setup.
    back_key = b"Torso_BackLeg" if b"Torso_BackLeg" in pyrosim.jointNamesToIndices else "Torso_BackLeg"
    front_key = b"Torso_FrontLeg" if b"Torso_FrontLeg" in pyrosim.jointNamesToIndices else "Torso_FrontLeg"

    start = p.getBasePositionAndOrientation(robotId)[0]

    for t in range(SIM_STEPS):
        pyrosim.Set_Motor_For_Joint(robotId, back_key, p.POSITION_CONTROL, float(backAngles[t]), MAX_FORCE)
        pyrosim.Set_Motor_For_Joint(robotId, front_key, p.POSITION_CONTROL, float(frontAngles[t]), MAX_FORCE)
        p.stepSimulation()

    end = p.getBasePositionAndOrientation(robotId)[0]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = (dx * dx + dy * dy) ** 0.5
    print("DIST", dist, "DX_DY", dx, dy, "MAX_FORCE", MAX_FORCE)

    p.disconnect()


if __name__ == "__main__":
    main()
