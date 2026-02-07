import os
import json
import math
from pathlib import Path

SIM_STEPS = 4000
PRINT_EVERY = 10
MAX_FORCE = 500.0  # debug default: lower flip risk; increase after tuning
RANDOM_TARGETS = False

SLEEP_TIME = 1/60  # tweak later for filming

# Added from simulate.py during refactor
DT = 1/240  # physics timestep
RNG_SEED = 0
TARGET_RANGE = 1.5707963267948966  # pi/2
SINE_CYCLES = 1.0
KICK_START = 200
KICK_END = 350

# Added during refactor
GRAVITY_Z = -9.8
MOTOR_FREQ_HZ = 1.0     # base frequency; other motor will be half
GAIT_AMPLITUDE = 0.55
GAIT_FREQ_HZ = 1.3
BACK_OFFSET = -0.20
FRONT_OFFSET = 0.15
ROBOT_FRICTION = 1.0  # debug default
PLANE_FRICTION = 1.0  # debug default
DEMO_SLEEP_TIME = 1/15  # slower for human eyes
DEMO_STRETCH = 6       # sample motor targets slower (bigger=slower)
DEMO_AMP_MULT = 1.5  # debug default: visible but less violent



def _coerce(name, old, s):
    t = type(old)
    if t is bool:
        return s.strip().lower() in ("1","true","yes","on")
    if t is int:
        try:
            return int(float(s))
        except Exception:
            return old
    if t is float:
        try:
            return float(s)
        except Exception:
            return old
    try:
        if s.strip().lower() in ("none","null"):
            return None
    except Exception:
        pass
    return s

for _k,_v in list(globals().items()):
    if isinstance(_k, str) and _k.isupper():
        _ev = os.getenv(_k)
        if _ev is not None:
            globals()[_k] = _coerce(_k, _v, _ev)

_vp = os.getenv("GAIT_VARIANT_PATH")
if _vp:
    try:
        _d = json.loads(Path(_vp).read_text(encoding="utf-8"))
        if isinstance(_d, dict):
            for _k,_v in _d.items():
                if isinstance(_k, str) and _k.isupper():
                    globals()[_k] = _v
    except Exception:
        pass
