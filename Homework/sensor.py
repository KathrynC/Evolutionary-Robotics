import pyrosim.pyrosim as pyrosim

class SENSOR:
    def __init__(self, linkName):
        self.linkName = linkName

    def Get_Value(self):
        return pyrosim.Get_Touch_Sensor_Value_For_Link(self.linkName)
