"""
simulate_video.py

GUI-only helper for recording a PyBullet run while printing touch sensor values.

This script is meant for *screen recording* (OBS / QuickTime screen capture, etc.):
    - Opens the PyBullet GUI and frames the camera.
    - Pauses for you to arrange windows.
    - Runs for SIM_STEPS steps.
    - Applies an external "poke" early in the run so contacts occur without mouse input.
    - Prints touch sensor values for BackLeg / FrontLeg periodically.

Requirements:
    - body.urdf (robot)
    - world.sdf (optional scenery; REQUIRED by this script as written)
    - plane.urdf (from pybullet_data)
    - pyrosim must have been prepared to simulate the loaded robot

Notes:
    - This file executes immediately when run (no main() wrapper).
    - `python3 -m py_compile simulate_video.py` only checks syntax; it does not open GUI.

Typical usage:
    python3 simulate_video.py

Ludobots role:
  - Utility for producing submission/demo videos of the robot in simulation.

Beyond Ludobots (this repo):
  - (Document camera presets, frame capture method, headless options.)
"""

import time
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim

SIM_STEPS = 6000
DT = 1 / 240


def safe_touch(link_name: str) -> float:
    """Read a pyrosim touch sensor value safely.

    On some macOS/PyBullet/pyrosim combinations, the underlying contact query can error
    (e.g., if getContactPoints() returns None). In that case we return -1.0 as a sentinel.

    Args:
        link_name: Link name as used in the URDF / pyrosim link map (e.g., "BackLeg").

    Returns:
        Touch sensor value (float). Returns -1.0 on known TypeError edge cases.
    """
    try:
        return pyrosim.Get_Touch_Sensor_Value_For_Link(link_name)
    except TypeError:
        # Sometimes getContactPoints() returns None on macOS; treat as "no contact"
        return -1.0


# --- GUI setup (intentionally interactive) ---
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)        # hide side panels
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)    # optional

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(DT)

# World + robot assets (world.sdf is required here; other scripts may load it optionally)
p.loadURDF("plane.urdf")
p.loadSDF("world.sdf")
robotId = p.loadURDF("body.urdf")

pyrosim.Prepare_To_Simulate(robotId)

# Camera framing
p.resetDebugVisualizerCamera(
    cameraDistance=3.0,
    cameraYaw=45,
    cameraPitch=-25,
    cameraTargetPosition=[0, 0, 0.7],
)

# Pause so you can arrange windows before recording
input("\nArrange/resize the PyBullet window + Terminal now. Press Enter to start the simulation... ")

# --- Simulation loop ---
for i in range(SIM_STEPS):
    # Gentle poke so contacts happen without mouse interaction
    if 240 <= i <= 1200:
        p.applyExternalForce(
            objectUniqueId=robotId,
            linkIndex=-1,          # torso/root link
            forceObj=[2000, 0, 0], # stronger poke for visible switching
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
        )

    p.stepSimulation()

    back = safe_touch("BackLeg")
    front = safe_touch("FrontLeg")

    # Print at a lower rate to keep Terminal readable while filming
    if i % 10 == 0:
        print(i, back, front, flush=True)

    # Slow to realtime-ish for filming (DT = 1/240)
    time.sleep(DT)

p.disconnect()
