import os
import numpy as np
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim
import matplotlib.image as mpimg

SIM_STEPS = 1000
DT = 1/240
MAX_FORCE = float(os.getenv("MAX_FORCE", "200"))
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "1"))

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join("data", "openloop_frames")
    os.makedirs(out_dir, exist_ok=True)

    back = np.load("data/backAngles.npy")
    front = np.load("data/frontAngles.npy")

    p.connect(p.DIRECT)
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

    w, h = 1280, 720

    for i in range(SIM_STEPS):
        pyrosim.Set_Motor_For_Joint(robotId, back_key, p.POSITION_CONTROL, float(back[i]), MAX_FORCE)
        pyrosim.Set_Motor_For_Joint(robotId, front_key, p.POSITION_CONTROL, float(front[i]), MAX_FORCE)
        p.stepSimulation()

        if i % FRAME_STRIDE == 0:
            base = p.getBasePositionAndOrientation(robotId)[0]
            target = [base[0], base[1], base[2]]
            view = p.computeViewMatrixFromYawPitchRoll(target, distance=2.5, yaw=50, pitch=-25, roll=0, upAxisIndex=2)
            proj = p.computeProjectionMatrixFOV(fov=60, aspect=16/9, nearVal=0.1, farVal=50)
            img = p.getCameraImage(w, h, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER)
            rgba = np.array(img[2], dtype=np.uint8).reshape(h, w, 4)
            rgb = rgba[:, :, :3]
            mpimg.imsave(os.path.join(out_dir, f"frame_{i:04d}.png"), rgb)

    p.disconnect()
    print("WROTE frames to", out_dir)

if __name__ == "__main__":
    main()
