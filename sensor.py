"""
sensor.py

Role:
    Records touch sensor readings for a single link over the course of a simulation run.

Data model:
    - Each SENSOR owns a fixed-length numpy array `values` with one entry per timestep.
    - At each timestep t, Get_Value(t) queries pyrosim for the link's touch value and stores it.

Notes:
    - The array length is `constants.SIM_STEPS`. If SIM_STEPS changes, sensors should be rebuilt.
    - Touch reads may fail (e.g., link name mismatch or pyrosim state not prepared); failures record 0.0.

Ludobots role:
  - Defines sensor objects that read from PyBullet/pyrosim.
  - In F. Sensors, sensor values are recorded; in I. Neurons, they become neural inputs.

Beyond Ludobots (this repo):
  - (Document any added sensors, logging formats, or telemetry integration.)
"""

import numpy as np
import pyrosim.pyrosim as pyrosim
import constants as c


class SENSOR:
    """Touch sensor logger for a single robot link."""

    def __init__(self, linkName: str):
        """Create a sensor bound to one link.

        Args:
            linkName: Link name as used by pyrosim / the URDF.
        """
        self.linkName = linkName
        self.values = np.zeros(c.SIM_STEPS)

    def Get_Value(self, t: int):
        """Read and store the touch sensor value at timestep t.

        Args:
            t: Integer timestep index.

        Side effects:
            Updates self.values[t] with the latest touch value (float).
        """
        try:
            v = pyrosim.Get_Touch_Sensor_Value_For_Link(self.linkName)
        except Exception:
            v = 0.0
        self.values[t] = float(v)
