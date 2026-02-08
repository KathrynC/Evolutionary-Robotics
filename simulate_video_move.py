"""simulate_video_move.py

GUI video recorder that drives all revolute joints with a simple sinusoidal pattern.

Purpose:
    - Produce a quick MP4 recording ("motors_run.mp4") of a robot attempting locomotion.
    - Bypass pyrosim/neural-network plumbing and control joints directly via PyBullet.
    - Provide a visible baseline motion pattern for debugging URDF, friction, and motor limits.

What it does:
    - Loads plane.urdf and body.urdf.
    - Finds all revolute joints and applies a traveling-wave sine target across them.
    - Uses startStateLogging(...) to record an MP4.
    - Uses a follow camera so the robot stays centered during recording.

Requirements:
    - body.urdf in the current directory (run `python3 generate.py` if missing).

Usage:
    python3 simulate_video_move.py

Outputs:
    motors_run.mp4 (in the current directory)

Notes:
    - This script intentionally uses PyBullet GUI mode and sleeps at dt for stable video.
    - If the robot does not move, the usual culprits are friction, motor force, or joint
      axis/limits in the URDF.
"""

import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data


def horiz_dist_xy(pos):
    """Return horizontal distance from origin given a base position (x, y, z)."""
    return math.hypot(pos[0], pos[1])


def main():
    """Run a ~10s GUI simulation while recording an MP4.

    Flow:
        1) Setup physics + plane friction
        2) Load body.urdf and apply baseline friction to all links
        3) Discover revolute joints (motors)
        4) Settle for 1s (no motors), then start MP4 recording
        5) Drive all revolute joints with a phase-offset sine pattern
        6) Follow camera + print periodic telemetry
    """
    # GUI is simplest for MP4 recording and for seeing what's happening.
    cid = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0 / 240.0)
    p.setPhysicsEngineParameter(numSolverIterations=200)

    # Always load a plain plane to avoid mystery extra objects from world.sdf.
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(
        plane_id,
        -1,
        lateralFriction=1.6,
        spinningFriction=0.05,
        rollingFriction=0.05,
    )

    # Load robot URDF
    if not Path("body.urdf").exists():
        raise FileNotFoundError("body.urdf not found. Run: python3 generate.py")

    # Start slightly above ground for stable settling.
    # (URDF base frame can be high; keep this modest.)
    robot_id = p.loadURDF("body.urdf", basePosition=[0, 0, 0.6])

    # Give every link traction (helps a lot).
    for link in range(-1, p.getNumJoints(robot_id)):
        p.changeDynamics(
            robot_id,
            link,
            lateralFriction=1.2,
            spinningFriction=0.05,
            rollingFriction=0.05,
        )

    # Discover revolute joints (motors)
    revolute = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] == p.JOINT_REVOLUTE:
            revolute.append(j)

    print(f"Revolute joints: {revolute}")
    for j in revolute:
        info = p.getJointInfo(robot_id, j)
        name = info[1].decode("utf-8", errors="replace")
        axis = info[13]
        print(f"  joint {j:2d}: {name:>20s} axis={axis}")

    if not revolute:
        print("No revolute joints found. If you expected motors, check your URDF joints.")
        p.disconnect(cid)
        return

    dt = 1.0 / 240.0
    record_seconds = 10.0
    steps = int(record_seconds / dt)  # record ~10 seconds
    target_dist = 1e9  # disable early stop on distance

    # Let the robot settle before recording/motors (avoids "it moved because it fell").
    settle_seconds = 1.0
    for _ in range(int(settle_seconds / dt)):
        p.stepSimulation()
        time.sleep(dt)

    # Kill any residual drift from settling.
    p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    # Record MP4 after settling
    out_name = "motors_run.mp4"
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, out_name)
    print(f"Recording -> {out_name}")

    # --- Tune knobs (main behavior is here) ---
    # Locomotion attempt: traveling wave + small bias breaks symmetry
    amp = 1.1   # radians
    freq = 2.0  # Hz
    force = 200
    n = max(1, len(revolute))

    for t in range(steps):
        phase = 2.0 * math.pi * freq * (t * dt)

        for k, j in enumerate(revolute):
            offset = 2.0 * math.pi * k / n
            bias = 0.20 if (k % 2) else -0.05
            target = amp * math.sin(phase + offset) + bias

            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=force,
                positionGain=0.6,
                velocityGain=0.3,
            )

        p.stepSimulation()

        pos, _ = p.getBasePositionAndOrientation(robot_id)
        d = horiz_dist_xy(pos)

        # Follow-cam
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=pos,
        )

        if t % 120 == 0:
            print(f"t={t:4d} pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) dist={d:.2f}m")

        # Keep recording for full duration; do not stop early on distance.
        if False and d >= target_dist:
            print(f"Reached {d:.2f} m from origin. Continuing to record.")
            # break  # disabled; keep recording full duration

        time.sleep(dt)

    p.stopStateLogging(log_id)
    p.disconnect(cid)
    print("Done.")


if __name__ == "__main__":
    main()
