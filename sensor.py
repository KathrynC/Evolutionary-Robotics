import numpy as np
import pyrosim.pyrosim as pyrosim
import constants as c

class SENSOR:
    def __init__(self, linkName: str):
        self.linkName = linkName
        self.values = np.zeros(c.SIM_STEPS)

    def Get_Value(self, t: int):
        try:
            v = pyrosim.Get_Touch_Sensor_Value_For_Link(self.linkName)
        except Exception:
            v = 0.0
        self.values[t] = float(v)
