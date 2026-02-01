import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy
import pybullet as p
import pybullet_data

# Must match your "legal" script: fixed motor strength (do not tune)
MOTOR_MAX_FORCE = 50


@dataclass
class Result:
    dist_xy: float
    freq_front: float
    phase_front: float
    freq_back: float
    phase_back: float
    amp_back: float
    amp_front: float
    z: float
    roll: float
    pitch: float


def parse_expr(s: str) -> float:
    """
    Accept simple expressions like:
      pi/3, 0.45*pi, 1.2, 2*pi/3
    """
    allowed = {"pi": math.pi}
    return float(eval(s, {"__builtins__": {}}, allowed))


def find_joint(robot_id: int, needle: str):
    for j in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, j)[1].decode("utf-8", errors="replace")
        if needle in name:
            return j
    return None


def run_once(
    steps: int,
    dt: float,
    amp_back: float,
    freq_back: float,
    phase_back: float,
    amp_front: float,
    freq_front: float,
    phase_front: float,
) -> Result:
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(dt)
    p.setPhysicsEngineParameter(numSolverIterations=300)

    p.loadURDF("plane.urdf")

    if not Path("body.urdf").exists():
        p.disconnect(cid)
        raise FileNotFoundError("body.urdf not found. Run: python3 generate.py")

    robot_id = p.loadURDF("body.urdf", basePosition=[0, 0, 0.6])

    back_joint = find_joint(robot_id, "BackLeg")
    front_joint = find_joint(robot_id, "FrontLeg")
    if back_joint is None or front_joint is None:
        rev = [j for j in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, j)[2] == p.JOINT_REVOLUTE]
        if len(rev) < 2:
            p.disconnect(cid)
            raise RuntimeError("Need at least 2 revolute joints.")
        back_joint, front_joint = rev[0], rev[1]

    # Open-loop motor targets, legal range enforced by caller (amp <= pi/4).
    x = numpy.linspace(0.0, 2.0 * numpy.pi, steps, endpoint=False)
    target_back = amp_back * numpy.sin(freq_back * x + phase_back)
    target_front = amp_front * numpy.sin(freq_front * x + phase_front)

    for i in range(steps):
        p.setJointMotorControl2(
            robot_id, back_joint, p.POSITION_CONTROL,
            targetPosition=float(target_back[i]),
            force=MOTOR_MAX_FORCE,
        )
        p.setJointMotorControl2(
            robot_id, front_joint, p.POSITION_CONTROL,
            targetPosition=float(target_front[i]),
            force=MOTOR_MAX_FORCE,
        )
        p.stepSimulation()

    pos, orn = p.getBasePositionAndOrientation(robot_id)
    roll, pitch, _ = p.getEulerFromQuaternion(orn)

    dxy = math.hypot(pos[0], pos[1])
    out = Result(
        dist_xy=dxy,
        freq_front=freq_front,
        phase_front=phase_front,
        freq_back=freq_back,
        phase_back=phase_back,
        amp_back=amp_back,
        amp_front=amp_front,
        z=pos[2],
        roll=roll,
        pitch=pitch,
    )

    p.disconnect(cid)
    return out


def main():
    ap = argparse.ArgumentParser(description="Bulk test legal open-loop gaits (DIRECT mode).")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=1.0 / 240.0)

    # Defaults: keep back leg fixed and sweep front leg (most useful first sweep)
    ap.add_argument("--amp-back", type=float, default=math.pi / 4)
    ap.add_argument("--freq-back", type=float, default=4.0)
    ap.add_argument("--phase-back", type=str, default="0")

    ap.add_argument("--amp-front", type=float, default=math.pi / 4)
    ap.add_argument("--freq-front", type=float, nargs="*", default=[4.0])
    ap.add_argument("--phase-front", type=str, nargs="*", default=[
        "pi/3", "0.45*pi", "pi/2", "0.55*pi", "2*pi/3"
    ])

    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    # Legal amplitude clamp
    amax = math.pi / 4
    if not (0.0 <= args.amp_back <= amax and 0.0 <= args.amp_front <= amax):
        raise SystemExit("Amplitude must be within [0, pi/4] to stay in the legal zone.")

    phase_back = parse_expr(args.phase_back)
    phase_fronts = [parse_expr(s) for s in args.phase_front]

    results = []
    total = 0

    for ff in args.freq_front:
        for pf in phase_fronts:
            total += 1
            r = run_once(
                steps=args.steps,
                dt=args.dt,
                amp_back=args.amp_back,
                freq_back=args.freq_back,
                phase_back=phase_back,
                amp_front=args.amp_front,
                freq_front=ff,
                phase_front=pf,
            )
            results.append(r)

    results.sort(key=lambda r: r.dist_xy, reverse=True)

    print(f"Ran {total} combinations (DIRECT). MOTOR_MAX_FORCE fixed at {MOTOR_MAX_FORCE}.")
    print(f"Back: amp={args.amp_back:.3f}, freq={args.freq_back:.3f}, phase={phase_back:.3f}")
    print(f"Front amp={args.amp_front:.3f}; swept freq_front={args.freq_front} and {len(phase_fronts)} phases.\n")

    print(f"Top {min(args.top, len(results))}:")
    for i, r in enumerate(results[: args.top], 1):
        print(
            f"{i:2d}) dist={r.dist_xy:7.3f} m | "
            f"front(freq={r.freq_front:.3f}, phase={r.phase_front:.3f}) | "
            f"z={r.z:.2f} roll={r.roll:.2f} pitch={r.pitch:.2f}"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(Result.__annotations__.keys())
            for r in results:
                w.writerow([getattr(r, k) for k in Result.__annotations__.keys()])
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
