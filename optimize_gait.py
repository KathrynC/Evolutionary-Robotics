import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pybullet as p
import pybullet_data


@dataclass
class Params:
    amp: float
    freq: float
    phase_diff: float
    bias0: float
    bias1: float
    force: float
    foot_friction: float
    torso_friction: float
    position_gain: float
    velocity_gain: float


def dist_xy(pos) -> float:
    return math.hypot(pos[0], pos[1])


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def get_limits(robot_id, joint_index):
    info = p.getJointInfo(robot_id, joint_index)
    lo, hi = info[8], info[9]  # lower/upper
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return None
    # Sometimes limits are absurdly wide; clamp to something sane
    if abs(lo) > 50 or abs(hi) > 50:
        return None
    return lo, hi


def run_trial(params: Params, seconds: float, dt: float) -> float:
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(dt)
    p.setPhysicsEngineParameter(numSolverIterations=350)

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=1.6, spinningFriction=0.05, rollingFriction=0.05)

    if not Path("body.urdf").exists():
        p.disconnect(cid)
        raise FileNotFoundError("body.urdf not found. Run: python3 generate.py")

    robot_id = p.loadURDF("body.urdf", basePosition=[0, 0, 0.6])

    # Find revolute joints
    motors = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] == p.JOINT_REVOLUTE:
            motors.append(j)

    if len(motors) < 2:
        p.disconnect(cid)
        return 0.0

    j0, j1 = motors[0], motors[1]
    lim0 = get_limits(robot_id, j0)
    lim1 = get_limits(robot_id, j1)

    # Friction split: slippery torso, grippy legs (big distance multiplier)
    p.changeDynamics(robot_id, -1, lateralFriction=params.torso_friction, spinningFriction=0.0, rollingFriction=0.0)
    p.changeDynamics(robot_id, j0, lateralFriction=params.foot_friction, spinningFriction=0.05, rollingFriction=0.05)
    p.changeDynamics(robot_id, j1, lateralFriction=params.foot_friction, spinningFriction=0.05, rollingFriction=0.05)

    # Settle
    for _ in range(int(1.0 / dt)):
        p.stepSimulation()
    p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    steps = int(seconds / dt)
    for t in range(steps):
        phase = 2.0 * math.pi * params.freq * (t * dt)
        target0 = params.amp * math.sin(phase) + params.bias0
        target1 = params.amp * math.sin(phase + params.phase_diff) + params.bias1

        if lim0 is not None:
            target0 = clamp(target0, lim0[0], lim0[1])
        if lim1 is not None:
            target1 = clamp(target1, lim1[0], lim1[1])

        p.setJointMotorControl2(
            robot_id, j0, p.POSITION_CONTROL,
            targetPosition=target0,
            force=params.force,
            positionGain=params.position_gain,
            velocityGain=params.velocity_gain,
        )
        p.setJointMotorControl2(
            robot_id, j1, p.POSITION_CONTROL,
            targetPosition=target1,
            force=params.force,
            positionGain=params.position_gain,
            velocityGain=params.velocity_gain,
        )

        p.stepSimulation()

    pos, orn = p.getBasePositionAndOrientation(robot_id)
    roll, pitch, _ = p.getEulerFromQuaternion(orn)

    score = dist_xy(pos)

    # Penalize “catapult flips” so the winner is actual locomotion
    if abs(roll) > 1.2 or abs(pitch) > 1.2 or pos[2] > 3.0:
        score *= 0.6

    p.disconnect(cid)
    return score


def sample(rng: random.Random) -> Params:
    return Params(
        amp=rng.uniform(0.3, 1.8),
        freq=rng.uniform(0.6, 5.0),
        phase_diff=rng.uniform(0.0, 2.0 * math.pi),
        bias0=rng.uniform(-0.7, 0.7),
        bias1=rng.uniform(-0.7, 0.7),
        force=rng.uniform(80, 650),
        foot_friction=rng.uniform(1.5, 10.0),
        torso_friction=rng.uniform(0.02, 0.6),
        position_gain=rng.uniform(0.3, 1.0),
        velocity_gain=rng.uniform(0.05, 0.7),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=800)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=1.0 / 240.0)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    best_score = -1.0
    best = None

    for i in range(1, args.trials + 1):
        params = sample(rng)
        score = run_trial(params, seconds=args.seconds, dt=args.dt)
        if score > best_score:
            best_score = score
            best = params
            print(f"[best @ trial {i}] dist={best_score:.3f}  {best}")

    print("\nBest parameters:")
    print(best)
    print(f"Best distance in {args.seconds:.1f}s: {best_score:.3f} m")
    print("\nCopy into simulate_video_move.py:")
    print(f"amp={best.amp:.3f}, freq={best.freq:.3f}, force={best.force:.1f}")
    print(f"phase_diff={best.phase_diff:.3f}, bias0={best.bias0:.3f}, bias1={best.bias1:.3f}")
    print(f"foot_friction={best.foot_friction:.2f}, torso_friction={best.torso_friction:.2f}")
    print(f"positionGain={best.position_gain:.2f}, velocityGain={best.velocity_gain:.2f}")


if __name__ == "__main__":
    main()
