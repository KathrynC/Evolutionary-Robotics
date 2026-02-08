"""
simulate_openloop_legal.py

Ludobots role:
  - Simulation runner specialized for open-loop 'legal' gait expressions.

Beyond Ludobots (this repo):
  - (Document telemetry, logging, or video capture hooks if present.)
"""

import argparse
import math
import time
from pathlib import Path

import numpy
import pybullet as p
import pybullet_data

# Fixed motor strength. Do NOT tune this (assignment rule).
MOTOR_MAX_FORCE = 50

def find_joint(robot_id: int, needle: str):
    for j in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, j)[1].decode("utf-8", errors="replace")
        if needle in name:
            return j
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=1.0 / 240.0)
    ap.add_argument("--dump-only", action="store_true")
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    steps = args.steps
    dt = args.dt

    # Six legal open-loop variables (two motors)
    amplitudeBack = numpy.pi / 4
    frequencyBack = 4.0
    phaseOffsetBack = 0.0

    amplitudeFront = numpy.pi / 4
    frequencyFront = 4.4
    phaseOffsetFront = numpy.pi/2

    x = numpy.linspace(0.0, 2.0 * numpy.pi, steps, endpoint=False)
    targetAnglesBack  = amplitudeBack  * numpy.sin(frequencyBack  * x + phaseOffsetBack)
    targetAnglesFront = amplitudeFront * numpy.sin(frequencyFront * x + phaseOffsetFront)

    if args.dump_only:
        numpy.savetxt("motor_back.txt", targetAnglesBack)
        numpy.savetxt("motor_front.txt", targetAnglesFront)
        print("Wrote motor_back.txt and motor_front.txt")
        return

    cid = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(dt)

    p.loadURDF("plane.urdf")

    if not Path("body.urdf").exists():
        raise FileNotFoundError("body.urdf not found. Run: python3 generate.py")
    robot_id = p.loadURDF("body.urdf", basePosition=[0, 0, 0.6])

    back_joint = find_joint(robot_id, "BackLeg")
    front_joint = find_joint(robot_id, "FrontLeg")

    if back_joint is None or front_joint is None:
        rev = [j for j in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, j)[2] == p.JOINT_REVOLUTE]
        if len(rev) < 2:
            raise RuntimeError("Could not find two revolute joints to control.")
        back_joint, front_joint = rev[0], rev[1]

    log_id = None
    if args.video:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "motors_run.mp4")
        print("Recording -> motors_run.mp4")

    for i in range(steps):
        p.setJointMotorControl2(robot_id, back_joint, p.POSITION_CONTROL,
                                targetPosition=float(targetAnglesBack[i]), force=MOTOR_MAX_FORCE)
        p.setJointMotorControl2(robot_id, front_joint, p.POSITION_CONTROL,
                                targetPosition=float(targetAnglesFront[i]), force=MOTOR_MAX_FORCE)
        p.stepSimulation()
        time.sleep(dt)

    if log_id is not None:
        p.stopStateLogging(log_id)

    pos, _ = p.getBasePositionAndOrientation(robot_id)
    print(f"Final distance from origin (XY): {math.hypot(pos[0], pos[1]):.3f} m")

    p.disconnect(cid)

if __name__ == "__main__":
    main()
