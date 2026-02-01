import os
import numpy as np
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim

SIM_STEPS = 1000
DT = 1/240
MAX_FORCE = float(os.getenv("MAX_FORCE", "200"))

# Start values (you'll tune these)
amplitudeBack = np.pi/4
frequencyBack = 1
phaseOffsetBack = 0.0

amplitudeFront = np.pi/4
frequencyFront = 1
phaseOffsetFront = 0.0

SAVE_VECTORS_ONLY = os.getenv("SAVE_VECTORS_ONLY", "0") == "1"

def main():
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
        except Exception:
            pass

    robotId = p.loadURDF("body.urdf")
    pyrosim.Prepare_To_Simulate(robotId)

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
    dist = (dx*dx + dy*dy) ** 0.5
    print("DIST", dist, "DX_DY", dx, dy, "MAX_FORCE", MAX_FORCE)

    p.disconnect()

if __name__ == "__main__":
    main()
