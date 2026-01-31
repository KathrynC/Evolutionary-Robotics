import os
import math
import numpy as np
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim

import matplotlib.image as mpimg

DT = 1/240
STEPS_SETTLE = 1200
STEPS_EVAL_TAIL = 200  # measure z over last N steps
FORCE = float(os.getenv("MAX_FORCE", "200"))

# Candidate angles to try for "tip toes" pose
CAND = [-math.pi/2, -math.pi/4, 0.0, math.pi/4, math.pi/2]

def run_pose(back_target: float, front_target: float) -> tuple[float, float]:
    """Return (mean_z_tail, max_z) for this pose."""
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(DT)

    p.loadURDF("plane.urdf")
    # If you have a local world.sdf, load it; otherwise skip.
    if os.path.exists("world.sdf"):
        try:
            p.loadSDF("world.sdf")
        except Exception:
            pass

    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    back_key = b"Torso_BackLeg" if b"Torso_BackLeg" in pyrosim.jointNamesToIndices else "Torso_BackLeg"
    front_key = b"Torso_FrontLeg" if b"Torso_FrontLeg" in pyrosim.jointNamesToIndices else "Torso_FrontLeg"

    zs = []
    max_z = -1e9

    for i in range(STEPS_SETTLE):
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=robotId,
            jointName=back_key,
            controlMode=p.POSITION_CONTROL,
            targetPosition=back_target,
            maxForce=FORCE,
        )
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=robotId,
            jointName=front_key,
            controlMode=p.POSITION_CONTROL,
            targetPosition=front_target,
            maxForce=FORCE,
        )

        p.stepSimulation()

        z = p.getBasePositionAndOrientation(robotId)[0][2]
        max_z = max(max_z, z)
        zs.append(z)

    tail = zs[-STEPS_EVAL_TAIL:] if len(zs) >= STEPS_EVAL_TAIL else zs
    mean_tail = float(np.mean(tail)) if tail else float("nan")
    return mean_tail, float(max_z)

def render_png(robotId: int, out_path: str):
    base_pos = p.getBasePositionAndOrientation(robotId)[0]
    target = [base_pos[0], base_pos[1], base_pos[2]]

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=2.5,
        yaw=50,
        pitch=-25,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=16/9,
        nearVal=0.1,
        farVal=50,
    )

    w, h = 1280, 720
    img = p.getCameraImage(width=w, height=h, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER)
    rgba = np.array(img[2], dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[:, :, :3]
    mpimg.imsave(out_path, rgb)

def main():
    p.connect(p.DIRECT)
    # Ensure we’re in repo root (so body.urdf is found)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Search for a good "tip toes" pose by maximizing mean z near the end,
    # but avoid insane launches by preferring lower max_z if mean_z ties.
    best = None
    for b in CAND:
        for f in CAND:
            mean_z, max_z = run_pose(b, f)
            score = (mean_z, -max_z)
            if best is None or score > best[0]:
                best = (score, b, f, mean_z, max_z)

    assert best is not None
    _, b, f, mean_z, max_z = best
    print("BEST_BACK_TARGET", b)
    print("BEST_FRONT_TARGET", f)
    print("MEAN_Z_TAIL", mean_z)
    print("MAX_Z", max_z)
    print("MAX_FORCE", FORCE)

    # Rerun best pose and render image
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(DT)
    p.loadURDF("plane.urdf")
    if os.path.exists("world.sdf"):
        try:
            p.loadSDF("world.sdf")
        except Exception:
            pass
    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

    back_key = b"Torso_BackLeg" if b"Torso_BackLeg" in pyrosim.jointNamesToIndices else "Torso_BackLeg"
    front_key = b"Torso_FrontLeg" if b"Torso_FrontLeg" in pyrosim.jointNamesToIndices else "Torso_FrontLeg"

    for _ in range(800):
        pyrosim.Set_Motor_For_Joint(robotId, back_key, p.POSITION_CONTROL, b, FORCE)
        pyrosim.Set_Motor_For_Joint(robotId, front_key, p.POSITION_CONTROL, f, FORCE)
        p.stepSimulation()

    out_path = os.path.join("data", "tiptoes.png")
    render_png(robotId, out_path)
    print("WROTE", out_path)

    p.disconnect()

if __name__ == "__main__":
    main()
