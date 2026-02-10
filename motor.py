import os
import numpy as np
import pybullet as p
import pyrosim.pyrosim as pyrosim
import constants as c

class MOTOR:
    def __init__(self, jointName):
        # Keep original key type for pyrosim dict lookups (bytes or str)
        self.jointName = jointName
        self.jointNameStr = jointName.decode() if isinstance(jointName, (bytes, bytearray)) else str(jointName)

        # Allow quick overrides from env so demos are easy to tune
        base_f = float(os.getenv("GAIT_FREQ_HZ", str(getattr(c, "GAIT_FREQ_HZ", 1.0))))
        amp    = float(os.getenv("GAIT_AMPLITUDE", str(getattr(c, "GAIT_AMPLITUDE", 0.7))))

        demo_pure = os.getenv("DEMO_PURE_SINE", "0") == "1"
        half_demo = os.getenv("HALF_FREQ_DEMO", "0") == "1"

        # Defaults (locomotion-oriented)
        if "BackLeg" in self.jointNameStr:
            offset = float(getattr(c, "BACK_OFFSET", -0.25))
            phase  = float(getattr(c, "BACK_PHASE", 0.0))
            freq   = base_f
            sign   = +1.0
        elif "FrontLeg" in self.jointNameStr:
            offset = float(getattr(c, "FRONT_OFFSET", 0.20))
            phase  = float(getattr(c, "FRONT_PHASE", 3.1415926535))
            freq   = base_f
            sign   = -1.0
        else:
            offset, phase, freq, sign = 0.0, 0.0, base_f, 1.0

        # DEMO: make it visually countable (pure sine, no offsets/phases)
        if demo_pure:
            offset = 0.0
            phase  = 0.0
            sign   = +1.0

        # Assignment demo: one leg half frequency of the other
        # (FrontLeg slower by 2×)
        if half_demo and "FrontLeg" in self.jointNameStr:
            freq *= 0.5

        self.freq_hz = freq * (0.5 if "back" in str(self.jointName).lower() else 1.0)  # for debugging


        dt = float(getattr(c, "DT", 1/240))
        t = np.arange(c.SIM_STEPS) * dt
        self.motorValues = offset + sign * amp * np.sin(2.0 * np.pi * freq * t + phase)

        # Keep angles reasonable
        self.motorValues = np.clip(self.motorValues, -1.3, 1.3)

    def Set_Value(self, robot, t: int, max_force: float):
        target = float(self.motorValues[t])
        try:
            pyrosim.Set_Motor_For_Joint(
                bodyIndex=robot.robotId,
                jointName=self.jointName,          # IMPORTANT: original key type
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                maxForce=float(max_force),
            )
        except TypeError:
            pyrosim.Set_Motor_For_Joint(robot.robotId, self.jointName, p.POSITION_CONTROL, target, float(max_force))
