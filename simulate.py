import os
import numpy
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim

SIM_STEPS = 4000
DT = 1 / 240  # physics timestep


def main():
    # Headless physics (no GUI). This avoids the macOS GUI segfaults.
    p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(DT)

    p.loadURDF("plane.urdf")
    p.loadSDF("world.sdf")
    robotId = p.loadURDF("body.urdf")

    # Prepare pyrosim's link/joint dictionaries and remember robotId
    pyrosim.Prepare_To_Simulate(robotId)

    backLegSensorValues = numpy.zeros(SIM_STEPS)
    frontLegSensorValues = numpy.zeros(SIM_STEPS)

    # Optional: give the robot a little "kick" so contacts change even without a GUI.
    # We apply a short sideways force to the torso (link index -1) early in the run.
    KICK_START = 200
    KICK_END = 350
    KICK_FORCE = [250, 0, 0]  # adjust magnitude if needed

    for i in range(SIM_STEPS):
        if KICK_START <= i <= KICK_END:
            p.applyExternalForce(
                objectUniqueId=robotId,
                linkIndex=-1,              # torso/root link
                forceObj=KICK_FORCE,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
            )

        p.stepSimulation()

        backLegSensorValues[i] = pyrosim.Get_Touch_Sensor_Value_For_Link("BackLeg")
        frontLegSensorValues[i] = pyrosim.Get_Touch_Sensor_Value_For_Link("FrontLeg")
 
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=robotId,
            jointName=b"Torso_BackLeg",
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,
            maxForce=500
        )
        # Print occasionally (keeps output readable and avoids slowing the sim)
        if i % 10 == 0:
            print(i, backLegSensorValues[i], frontLegSensorValues[i], flush=True)

    os.makedirs("data", exist_ok=True)
    numpy.save("data/backLegSensorValues.npy", backLegSensorValues)
    numpy.save("data/frontLegSensorValues.npy", frontLegSensorValues)

    p.disconnect()


if __name__ == "__main__":
    main()
