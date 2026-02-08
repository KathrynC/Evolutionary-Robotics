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
                self.freq_hz = float(_gget('GAIT_FREQ_HZ', _gget('GAIT_FREQ', _gget('f', _gget('Frequency', getattr(self,'freq_hz', None) or 0.0)))))
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

        # Demo: optional half-frequency back leg (only when HALF_FREQ_DEMO=1)
        if os.getenv('HALF_FREQ_DEMO','0') == '1':
            self.freq_hz = freq * (0.5 if 'back' in str(self.jointName).lower() else 1.0)
        else:
            self.freq_hz = freq


        dt = float(getattr(c, "DT", 1/240))
        t = np.arange(c.SIM_STEPS) * dt
        self.motorValues = offset + sign * amp * np.sin(2.0 * np.pi * freq * t + phase)

        # Keep angles reasonable
        self.motorValues = np.clip(self.motorValues, -1.3, 1.3)

    def Set_Value(self, robot, t: int, max_force: float):

        if os.getenv('SIM_DEBUG','0') == '1' and t == 0:

            jn = getattr(self,'jointNameStr', getattr(self,'jointName',''))

            print('[SETVAL]', jn, 'GAIT?', (_GAIT is not None), flush=True)

        # If a gait variant is provided, drive this joint directly from sine params
        if _GAIT is not None:
            # timestep (fallbacks)
            try:
                import constants as _c
                dt = float(os.getenv('PHYSICS_DT', str(getattr(_c, 'TIME_STEP', 1/240.0))))
            except Exception:
                dt = float(os.getenv('PHYSICS_DT', '0.0041666667'))
            A = float(_gget('GAIT_AMPLITUDE', _gget('GAIT_AMP', _gget('A', _gget('Amplitude', 0.5)))))
            base_f = float(_gget('GAIT_FREQ_HZ', _gget('GAIT_FREQ', _gget('f', _gget('Frequency', 1.0)))))
            jname = getattr(self, 'jointNameStr', getattr(self, 'jointName', ''))
            is_back = ('BackLeg' in str(jname)) or ('back' in str(jname).lower())
            # preserve refactoring demo semantics: back leg half frequency, front full
            f_hz = base_f * (0.5 if is_back else 1.0)
            if is_back:
                O = float(_gget('BACK_OFFSET', _gget('O_back', _gget('O_BACK', 0.0))))
                phi = float(_gget('BACK_PHASE', _gget('phi_back', _gget('PHI_BACK', 0.0))))
            else:
                O = float(_gget('FRONT_OFFSET', _gget('O_front', _gget('O_FRONT', 0.0))))
                phi = float(_gget('FRONT_PHASE', _gget('phi_front', _gget('PHI_FRONT', 0.0))))
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


# === GAIT_VARIANT runtime override (auto-wrap) ===
import os as _os, json as _json, math as _math
from pathlib import Path as _Path

_GAIT = None
_GAIT_PATH = _os.getenv("GAIT_VARIANT_PATH","")
if _GAIT_PATH:
    try:
        _GAIT = _json.loads(_Path(_GAIT_PATH).read_text(encoding="utf-8"))
    except Exception:
        _GAIT = None

def _gget(k, default=None):
    return default if _GAIT is None else _GAIT.get(k, default)

def _dt_fallback():
    try:
        import constants as _c
        return float(_os.getenv("PHYSICS_DT", str(getattr(_c, "TIME_STEP", 1/240.0))))
    except Exception:
        return float(_os.getenv("PHYSICS_DT", "0.0041666667"))

def _angle_for_joint(jname, t):
    A = float(_gget("A", 0.5))
    f_hz = float(_gget("f", 1.0))
    if "Back" in jname:
        O = float(_gget("O_back", 0.0))
        phi = float(_gget("phi_back", 0.0))
    else:
        O = float(_gget("O_front", 0.0))
        phi = float(_gget("phi_front", 0.0))
    dt = _dt_fallback()
    return O + A * _math.sin(2.0 * _math.pi * f_hz * (float(t) * dt) + phi), f_hz

def _find_robot(args):
    for a in args:
        if hasattr(a, "robotId"):
            return a
        if hasattr(a, "robot") and hasattr(a.robot, "robotId"):
            return a.robot
    return None

def _find_t(args, kwargs):
    if "t" in kwargs:
        return kwargs["t"]
    # common call: (robot, t, max_force) after self
    for a in reversed(args):
        if isinstance(a, (int, float)):
            return int(a)
    return 0

def _find_max_force(args, kwargs):
    if "max_force" in kwargs:
        return kwargs["max_force"]
    if "MAX_FORCE" in kwargs:
        return kwargs["MAX_FORCE"]
    return float(_gget("MAX_FORCE", _os.getenv("MAX_FORCE", "500.0")))

def _wrap_method(orig):
    if getattr(orig, "_gait_wrapped", False):
        return orig

    def wrapped(self, *args, **kwargs):
        if _GAIT is None:
            return orig(self, *args, **kwargs)

        robot = _find_robot(args)
        if robot is None:
            return orig(self, *args, **kwargs)

        jname = getattr(self, "jointNameStr", getattr(self, "jointName", ""))
        if not jname:
            return orig(self, *args, **kwargs)

        t = _find_t(args, kwargs)
        angle, f_hz = _angle_for_joint(jname, t)
        try:
            self.freq_hz = f_hz
        except Exception:
            pass

        mf = _find_max_force(args, kwargs)

        try:
            import pybullet as p
            import pyrosim.pyrosim as pyrosim
            try:
                pyrosim.Set_Motor_For_Joint(bodyIndex=robot.robotId,
                                            jointName=jname,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=angle,
                                            maxForce=float(mf))
            except TypeError:
                # older signature
                pyrosim.Set_Motor_For_Joint(robot.robotId, jname, p.POSITION_CONTROL, angle, float(mf))
            return
        except Exception:
            return orig(self, *args, **kwargs)

    wrapped._gait_wrapped = True
    return wrapped

# Patch any class in this module that has a plausible motor-setting method
for _name, _obj in list(globals().items()):
    if isinstance(_obj, type):
        for _meth in ("Set_Value","SetValue","Set_Motor","SetMotor","Set_Values"):
            if hasattr(_obj, _meth):
                try:
                    setattr(_obj, _meth, _wrap_method(getattr(_obj, _meth)))
                except Exception:
                    pass
