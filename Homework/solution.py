import numpy
import os
import random
import time
import pyrosim.pyrosim as pyrosim
import constants as c

class SOLUTION:
    def __init__(self, nextAvailableID):
        self.myID = nextAvailableID
        self.weights = numpy.random.rand(c.numSensorNeurons, c.numMotorNeurons) * 2 - 1

    def Set_ID(self, nextAvailableID):
        self.myID = nextAvailableID

    def Evaluate(self, directOrGUI):
        self.Create_World()
        self.Create_Body()
        self.Create_Brain()
        os.system("python3 simulate.py " + directOrGUI + " " + str(self.myID) + " 2&>1")
        fitnessFileName = "fitness" + str(self.myID) + ".txt"
        f = open(fitnessFileName, "r")
        self.fitness = float(f.read())
        f.close()

    def Start_Simulation(self, directOrGUI):
        self.Create_World()
        self.Create_Body()
        self.Create_Brain()
        os.system("python3 simulate.py " + directOrGUI + " " + str(self.myID) + " 2&>1 &")

    def Wait_For_Simulation_To_End(self):
        fitnessFileName = "fitness" + str(self.myID) + ".txt"
        while not os.path.exists(fitnessFileName):
            time.sleep(0.01)
        f = open(fitnessFileName, "r")
        self.fitness = float(f.read())
        f.close()
        os.system("rm " + fitnessFileName)

    def Mutate(self):
        randomRow = random.randint(0, c.numSensorNeurons - 1)
        randomColumn = random.randint(0, c.numMotorNeurons - 1)
        self.weights[randomRow, randomColumn] = random.random() * 2 - 1

    def Create_World(self):
        pyrosim.Start_SDF("world.sdf")
        pyrosim.Send_Cube(name="WorldBlock", pos=[3, 0, 0.5], size=[1, 1, 1])
        pyrosim.End()

    def Create_Body(self):
        pyrosim.Start_URDF("body.urdf")
        # Torso raised so legs don't penetrate floor
        pyrosim.Send_Cube(name="Torso", pos=[0, 0, 1.5], size=[1, 1, 0.5])

        # Front leg — hip at front edge, bottom of torso
        # Joint axis "0 1 0" so legs swing forward/backward in x-z plane
        pyrosim.Send_Joint(name="Torso_FrontLeg", parent="Torso", child="FrontLeg",
                           type="revolute", position=[0.5, 0, 1.25], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="FrontLeg", pos=[0, 0, -0.25], size=[0.2, 0.2, 0.5])
        pyrosim.Send_Joint(name="FrontLeg_FrontLowerLeg", parent="FrontLeg", child="FrontLowerLeg",
                           type="revolute", position=[0, 0, -0.5], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="FrontLowerLeg", pos=[0, 0, -0.25], size=[0.2, 0.2, 0.5])

        # Back leg — hip at back edge, bottom of torso
        pyrosim.Send_Joint(name="Torso_BackLeg", parent="Torso", child="BackLeg",
                           type="revolute", position=[-0.5, 0, 1.25], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="BackLeg", pos=[0, 0, -0.25], size=[0.2, 0.2, 0.5])
        pyrosim.Send_Joint(name="BackLeg_BackLowerLeg", parent="BackLeg", child="BackLowerLeg",
                           type="revolute", position=[0, 0, -0.5], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="BackLowerLeg", pos=[0, 0, -0.25], size=[0.2, 0.2, 0.5])

        # Left leg — hip at left edge, bottom of torso
        pyrosim.Send_Joint(name="Torso_LeftLeg", parent="Torso", child="LeftLeg",
                           type="revolute", position=[0, -0.5, 1.25], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="LeftLeg", pos=[0, 0, -0.25], size=[0.2, 0.2, 0.5])
        pyrosim.Send_Joint(name="LeftLeg_LeftLowerLeg", parent="LeftLeg", child="LeftLowerLeg",
                           type="revolute", position=[0, 0, -0.5], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="LeftLowerLeg", pos=[0, 0, -0.25], size=[0.2, 0.2, 0.5])

        # Right leg — hip at right edge, bottom of torso
        pyrosim.Send_Joint(name="Torso_RightLeg", parent="Torso", child="RightLeg",
                           type="revolute", position=[0, 0.5, 1.25], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="RightLeg", pos=[0, 0, -0.25], size=[0.2, 0.2, 0.5])
        pyrosim.Send_Joint(name="RightLeg_RightLowerLeg", parent="RightLeg", child="RightLowerLeg",
                           type="revolute", position=[0, 0, -0.5], jointAxis="0 1 0")
        pyrosim.Send_Cube(name="RightLowerLeg", pos=[0, 0, -0.25], size=[0.2, 0.2, 0.5])

        pyrosim.End()

    def Create_Brain(self):
        pyrosim.Start_NeuralNetwork("brain" + str(self.myID) + ".nndf")

        # Sensor neurons on the 4 lower legs (feet)
        pyrosim.Send_Sensor_Neuron(name=0, linkName="FrontLowerLeg")
        pyrosim.Send_Sensor_Neuron(name=1, linkName="BackLowerLeg")
        pyrosim.Send_Sensor_Neuron(name=2, linkName="LeftLowerLeg")
        pyrosim.Send_Sensor_Neuron(name=3, linkName="RightLowerLeg")

        # Motor neurons for all 8 joints
        pyrosim.Send_Motor_Neuron(name=4, jointName="Torso_FrontLeg")
        pyrosim.Send_Motor_Neuron(name=5, jointName="FrontLeg_FrontLowerLeg")
        pyrosim.Send_Motor_Neuron(name=6, jointName="Torso_BackLeg")
        pyrosim.Send_Motor_Neuron(name=7, jointName="BackLeg_BackLowerLeg")
        pyrosim.Send_Motor_Neuron(name=8, jointName="Torso_LeftLeg")
        pyrosim.Send_Motor_Neuron(name=9, jointName="LeftLeg_LeftLowerLeg")
        pyrosim.Send_Motor_Neuron(name=10, jointName="Torso_RightLeg")
        pyrosim.Send_Motor_Neuron(name=11, jointName="RightLeg_RightLowerLeg")

        for currentRow in range(c.numSensorNeurons):
            for currentColumn in range(c.numMotorNeurons):
                pyrosim.Send_Synapse(
                    sourceNeuronName=currentRow,
                    targetNeuronName=currentColumn + c.numSensorNeurons,
                    weight=self.weights[currentRow][currentColumn])
        pyrosim.End()
