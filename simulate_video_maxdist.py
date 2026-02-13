"""
simulate_video_maxdist.py

Role:
    Run a GUI simulation using the best parameters found by optimize_gait.py,
    record an MP4 video via PyBullet state logging, and track distance traveled.
    Designed for visual evaluation and demo recording of the optimized sine gait.

Pipeline:
    1. Set up physics with the optimized friction split (slippery torso, grippy legs).
    2. Settle the robot for 1 second to let it rest on the ground plane.
    3. Record 10 seconds of locomotion, following the robot with a chase camera.
    4. Print periodic position updates and the final XY distance.

Hardcoded parameters:
    - Best gait found by optimize_gait.py (seed=2, 800 trials, 10s each).
    - Friction split: torso_friction=0.22, foot_friction=9.68.
    - Solver: 350 iterations for high-fidelity contact resolution.

Outputs:
    - motors_run.mp4: recorded video of the GUI simulation.
    - Console: periodic position/distance updates and final distance.

Requirements:
    - body.urdf must exist (run `python3 generate.py` first).
    - Requires a display for PyBullet GUI mode.

Ludobots role:
    - Video capture plus distance-tracking to evaluate/compare gaits.

Beyond Ludobots (this repo):
    - Companion to optimize_gait.py; uses its best-found parameters for demo recording.
"""

import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data


import os
def dist_xy(pos) -> float:
    """Return the XY-plane distance of a 3D position from the origin."""
    return math.hypot(pos[0], pos[1])


def main():
    """Run a GUI simulation with optimized gait parameters, recording video and tracking distance.

    The gait parameters are hardcoded from the best result of optimize_gait.py
    (seed=2, 800 trials). Uses a friction split strategy (slippery torso, grippy
    feet) and dual-joint sine control with amplitude, frequency, phase offset,
    and per-joint biases.

    Side effects:
        - Opens a PyBullet GUI window with chase camera.
        - Records motors_run.mp4 via PyBullet state logging.
        - Prints position and distance to stdout every 120 steps.
    """
    # Best parameters found by optimize_gait.py (seed=2, trials=800, 10 seconds)
    amp = 1.349812021807046
    freq = 3.7746450934703417
    phase_diff = 3.0955305577005983
    bias0 = -0.5329560058358066
    bias1 = 0.6552171426923192
    force = 455.6268696042062
    foot_friction = 9.67705422977286
    torso_friction = 0.221567754013287
    position_gain = 0.7854402883531477
    velocity_gain = 0.25207047180610126

    dt = 1.0 / 240.0
    record_seconds = 10.0
    settle_seconds = 1.0

    cid = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(dt)
    p.setPhysicsEngineParameter(numSolverIterations=350)

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=1.6, spinningFriction=0.05, rollingFriction=0.05)

    if not Path("body.urdf").exists():
        raise FileNotFoundError("body.urdf not found. Run: python3 generate.py")

    robot_id = p.loadURDF("body.urdf", basePosition=[0, 0, 0.6])

    # Find revolute joints
    revolute = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] == p.JOINT_REVOLUTE:
            revolute.append(j)

    print(f"Revolute joints: {revolute}")
    if len(revolute) < 2:
        print("Need at least 2 revolute joints to use this gait.")
        p.disconnect(cid)
        return

    j0, j1 = revolute[0], revolute[1]

    # Friction split: slippery torso/base, grippy legs
    p.changeDynamics(robot_id, -1, lateralFriction=torso_friction, spinningFriction=0.0, rollingFriction=0.0)
    p.changeDynamics(robot_id, j0, lateralFriction=foot_friction, spinningFriction=0.05, rollingFriction=0.05)
    p.changeDynamics(robot_id, j1, lateralFriction=foot_friction, spinningFriction=0.05, rollingFriction=0.05)

    # Settle before recording: let the robot come to rest on the ground plane,
    # then zero out any residual velocity so the recording starts cleanly.
    for _ in range(int(settle_seconds / dt)):
        p.stepSimulation()
        time.sleep(dt)
    p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    out_name = "motors_run.mp4"
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, out_name)
    print(f"Recording -> {out_name}")

    steps = int(record_seconds / dt)

    for t in range(steps):
        phase = 2.0 * math.pi * freq * (t * dt)

        target0 = amp * math.sin(phase) + bias0
        target1 = amp * math.sin(phase + phase_diff) + bias1

        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=j0,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target0,
            force=force,
            positionGain=position_gain,
            velocityGain=velocity_gain,
        )
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=j1,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target1,
            force=force,
            positionGain=position_gain,
            velocityGain=velocity_gain,
        )

        p.stepSimulation()

        pos, _ = p.getBasePositionAndOrientation(robot_id)

        # Follow camera
        p.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=pos,
        )

        if t % 120 == 0:
            print(f"t={t:4d} pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) dist={dist_xy(pos):.2f}m")

        time.sleep(dt)

    end_pos, _ = p.getBasePositionAndOrientation(robot_id)
    print(f"Final dist from origin: {dist_xy(end_pos):.2f} m")

    p.stopStateLogging(log_id)
    p.disconnect(cid)
    print("Done.")


if __name__ == "__main__":
    main()
