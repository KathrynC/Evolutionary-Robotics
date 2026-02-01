import numpy as np
import pybullet as p
import pyrosim.pyrosim as pyrosim
import constants as c

class MOTOR:
    def __init__(self, jointName):
        # jointName may be bytes or str; keep it exactly as in pyrosim.jointNamesToIndices
        self.jointName = jointName
        self.jointNameStr = jointName.decode() if isinstance(jointName, (bytes, bytearray)) else str(jointName)
        self.motorValues = np.zeros(c.SIM_STEPS)

    def Set_Value(self, robot, t: int, targetPosition: float, max_force: float):
        self.motorValues[t] = float(targetPosition)

        # Call pyrosim using positional args to avoid keyword mismatches
        # Expected signature: (bodyIndex, jointName, controlMode, targetPosition, maxForce)
        pyrosim.Set_Motor_For_Joint(robot.robotId, self.jointName, p.POSITION_CONTROL, float(targetPosition), float(max_force))
