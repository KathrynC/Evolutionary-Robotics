import pybullet as p
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
import constants as c

from sensor import SENSOR
from motor import MOTOR

class ROBOT:
    def __init__(self, robotId=None, already_prepared=False):
        if robotId is None:
            self.robotId = p.loadURDF("body.urdf")
            self.nn = NEURAL_NETWORK("brain.nndf")
            pyrosim.Prepare_To_Simulate(self.robotId)
        else:
            self.robotId = robotId
            if not already_prepared:
                pyrosim.Prepare_To_Simulate(self.robotId)

        # Friction helps locomotion; kill bounce
        try:
            mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
            for link in range(-1, p.getNumJoints(self.robotId)):
                p.changeDynamics(self.robotId, link, lateralFriction=mu, restitution=0.0)
        except Exception:
            pass

        self.Prepare_To_Sense()
        self.Prepare_To_Act()

    def Prepare_To_Sense(self):
        self.sensors = {}
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

    # Accept both old-style and new-style calls; angles are ignored when motors have trajectories.
    def Think(self):
        self.nn.Update()
        self.nn.Print()
    def Act(self, t: int, max_force: float=None, back_angle=None, front_angle=None, **kwargs):
        if max_force is None:
            max_force = float(kwargs.get("MAX_FORCE", 500.0))
        if hasattr(self, "nn"):
            used_nn = False
            for neuronName in self.nn.neurons:
                n = self.nn.neurons[neuronName]
                if n.Is_Motor_Neuron():
                    jointName = n.Get_Joint_Name()
                    desiredAngle = n.Get_Value()
                    pyrosim.Set_Motor_For_Joint(bodyIndex=self.robotId, jointName=jointName, desiredAngle=desiredAngle, maxForce=max_force)
                    used_nn = True
            if used_nn:
                return
        for m in self.motors.values():
            m.Set_Value(self, t, max_force)
