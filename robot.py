import pybullet as p
import pyrosim.pyrosim as pyrosim

from sensor import SENSOR
from motor import MOTOR

class ROBOT:
    def __init__(self, robotId=None, already_prepared=False):
        # If robotId not provided, load + prepare.
        if robotId is None:
            self.robotId = p.loadURDF("body.urdf")
            pyrosim.Prepare_To_Simulate(self.robotId)
        else:
            self.robotId = robotId
            if not already_prepared:
                pyrosim.Prepare_To_Simulate(self.robotId)

        self.Prepare_To_Sense()
        self.Prepare_To_Act()

    def Prepare_To_Sense(self):
        self.sensors = {}
        # pyrosim.linkNamesToIndices keys may be bytes or str
        for k in pyrosim.linkNamesToIndices:
            name = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            self.sensors[name] = SENSOR(name)

    def Sense(self, t: int):
        for s in self.sensors.values():
            s.Get_Value(t)

    def Prepare_To_Act(self):
        self.motors = {}
        for jointName in pyrosim.jointNamesToIndices:
            self.motors[jointName] = MOTOR(jointName)

    def Act(self, t: int, back_angle: float, front_angle: float, max_force: float):
        # Generic loop: choose target based on joint name text, not hardcoded joint IDs.
        for m in self.motors.values():
            j = m.jointNameStr
            if "BackLeg" in j:
                target = back_angle
            elif "FrontLeg" in j:
                target = front_angle
            else:
                continue
            m.Set_Value(self, t, target, max_force)
