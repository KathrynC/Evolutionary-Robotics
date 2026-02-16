import pybullet as p
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
from sensor import SENSOR
from motor import MOTOR
import os
import constants as c

class ROBOT:
    def __init__(self, solutionID):
        self.solutionID = solutionID
        self.robotId = p.loadURDF("body.urdf")
        pyrosim.Prepare_To_Simulate(self.robotId)
        self.nn = NEURAL_NETWORK("brain" + str(solutionID) + ".nndf")
        self.Create_Sensors()
        self.Create_Motors()

    def Create_Sensors(self):
        self.sensors = {}
        for neuronName in self.nn.neurons:
            if self.nn.neurons[neuronName].Is_Sensor_Neuron():
                linkName = self.nn.neurons[neuronName].Get_Link_Name()
                self.sensors[neuronName] = SENSOR(linkName)

    def Create_Motors(self):
        self.motors = {}
        for neuronName in self.nn.neurons:
            if self.nn.neurons[neuronName].Is_Motor_Neuron():
                jointName = self.nn.neurons[neuronName].Get_Joint_Name()
                self.motors[neuronName] = MOTOR(jointName)

    def Sense(self, t):
        for neuronName in self.sensors:
            self.nn.neurons[neuronName].Set_Value(self.sensors[neuronName].Get_Value())

    def Think(self):
        for neuronName in self.nn.neurons:
            if self.nn.neurons[neuronName].Is_Motor_Neuron():
                self.nn.neurons[neuronName].Set_Value(0.0)
                for synapseName in self.nn.synapses:
                    if self.nn.synapses[synapseName].Get_Target_Neuron_Name() == neuronName:
                        sourceNeuronName = self.nn.synapses[synapseName].Get_Source_Neuron_Name()
                        sourceValue = self.nn.neurons[sourceNeuronName].Get_Value()
                        weight = self.nn.synapses[synapseName].Get_Weight()
                        self.nn.neurons[neuronName].Add_To_Value(sourceValue * weight)
                self.nn.neurons[neuronName].Threshold()

    def Act(self, t):
        for neuronName in self.motors:
            desiredAngle = self.nn.neurons[neuronName].Get_Value() * c.motorJointRange
            self.motors[neuronName].Set_Value(self.robotId, desiredAngle)

    def Get_Fitness(self):
        basePositionAndOrientation = p.getBasePositionAndOrientation(self.robotId)
        basePosition = basePositionAndOrientation[0]
        xPosition = basePosition[0]
        return xPosition
