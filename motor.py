import os
import numpy as np
import pybullet as p
import pyrosim.pyrosim as pyrosim
import constants as c


# GAIT_VARIANT sine controller (loaded from GAIT_VARIANT_PATH)
import os, json, math
from pathlib import Path
_GAIT = None
_GAIT_PATH = os.getenv('GAIT_VARIANT_PATH','')
if _GAIT_PATH:
    try:
        _GAIT = json.loads(Path(_GAIT_PATH).read_text(encoding='utf-8'))
    except Exception:
        _GAIT = None

def _gget(k, default=None):
    return default if _GAIT is None else _GAIT.get(k, default)

class MOTOR:
    def __init__(self, jointName):
        # Variant-provided frequency for debugging/telemetry
        if _GAIT is not None:
            try:
                self.freq_hz = float(_gget('f', _gget('Frequency', getattr(self,'freq_hz', None) or 0.0)))
            except Exception:
                pass
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
        # If a gait variant is provided, drive this joint directly from sine params
        if _GAIT is not None:
            # timestep (fallbacks)
            try:
                import constants as _c
                dt = float(os.getenv('PHYSICS_DT', str(getattr(_c, 'TIME_STEP', 1/240.0))))
            except Exception:
                dt = float(os.getenv('PHYSICS_DT', '0.0041666667'))
            A = float(_gget('A', _gget('Amplitude', 0.5)))
            f_hz = float(_gget('f', _gget('Frequency', 1.0)))
            jname = getattr(self, 'jointNameStr', getattr(self, 'jointName', ''))
            if 'BackLeg' in jname:
                O = float(_gget('O_back', _gget('O_BACK', 0.0)))
                phi = float(_gget('phi_back', _gget('PHI_BACK', 0.0)))
            else:
                O = float(_gget('O_front', _gget('O_FRONT', 0.0)))
                phi = float(_gget('phi_front', _gget('PHI_FRONT', 0.0)))
            angle = O + A * math.sin(2.0 * math.pi * f_hz * (t * dt) + phi)
            try:
                self.freq_hz = f_hz
            except Exception:
                pass
            # max force: prefer passed in, else variant, else constants default
            mf = max_force if 'max_force' in locals() else None
            if mf is None:
                mf = float(_gget('MAX_FORCE', 500.0))
            try:
                import pybullet as p
                import pyrosim.pyrosim as pyrosim
                try:
                    pyrosim.Set_Motor_For_Joint(bodyIndex=robot.robotId, jointName=jname, controlMode=p.POSITION_CONTROL, targetPosition=angle, maxForce=float(mf))
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robot.robotId, jname, p.POSITION_CONTROL, angle, float(mf))
                return
            except Exception:
                # fall back to the original method body if anything goes sideways
                pass
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
