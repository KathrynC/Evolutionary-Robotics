import numpy as np
import pybullet as p
import pyrosim.pyrosim as pyrosim
import constants as c

class MOTOR:
    def __init__(self, jointName):
        # Keep the original dict key exactly (bytes or str)
        self.jointName = jointName
        # Convenience string form for pattern matching (BackLeg/FrontLeg)
        self.jointNameStr = jointName.decode() if isinstance(jointName, (bytes, bytearray)) else str(jointName)
        self.motorValues = np.zeros(c.SIM_STEPS)

    def Set_Value(self, robot, t: int, targetPosition: float, max_force: float):
        self.motorValues[t] = float(targetPosition)

        # IMPORTANT: pass jointName using the original key type
        try:
            pyrosim.Set_Motor_For_Joint(
                bodyIndex=robot.robotId,
                jointName=self.jointName,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(targetPosition),
                maxForce=float(max_force),
            )
        except TypeError:
            # fallback: some versions only accept positional args
            pyrosim.Set_Motor_For_Joint(robot.robotId, self.jointName, p.POSITION_CONTROL, float(targetPosition), float(max_force))
