import pybullet as p
import pybullet_data
import time
from world import WORLD
from robot import ROBOT
import constants as c

class SIMULATION:
    def __init__(self, directOrGUI, solutionID):
        self.directOrGUI = directOrGUI
        self.solutionID = solutionID
        if self.directOrGUI == "DIRECT":
            self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.world = WORLD()
        self.robot = ROBOT(solutionID)

    def Run(self):
        for t in range(c.numTimeSteps):
            p.stepSimulation()
            self.robot.Sense(t)
            self.robot.Think()
            self.robot.Act(t)
            if self.directOrGUI == "GUI":
                time.sleep(1/240)

    def Get_Fitness(self):
        fitness = self.robot.Get_Fitness()
        return fitness

    def __del__(self):
        p.disconnect()
