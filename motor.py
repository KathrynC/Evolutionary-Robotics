"""motor.py

Role:
    Motor controller for a single joint. Generates target joint angles over time and sends
    POSITION_CONTROL commands to PyBullet via pyrosim.Set_Motor_For_Joint.

Control modes (highest priority first):
    1) GAIT_VARIANT_PATH provided:
         - Load a JSON "gait variant" dict and compute a sine target each timestep.
         - Supports per-leg offsets/phases and optional half-frequency semantics.
    2) No gait variant:
         - Precompute a sine trajectory array (motorValues) using constants/env overrides.

Important details:
    - jointName may be bytes or str; keep the original type for pyrosim lookups.
    - freq_hz is recorded on each MOTOR instance for debugging/telemetry.

Key env vars:
    GAIT_VARIANT_PATH: path to JSON gait variant file
    GAIT_FREQ_HZ / GAIT_AMPLITUDE: quick overrides for demos
    DEMO_PURE_SINE: force offsets/phases to 0 for visually countable motion
    HALF_FREQ_DEMO: half frequency for one leg (demo/assignment)
    PHYSICS_DT: timestep used when computing gait variant targets (fallbacks exist)
    SIM_DEBUG: extra prints at t==0

Notes for maintainers:
    - This file previously contained an "auto-wrap" gait override block.
      It was removed to keep control flow explicit and easier to reason about.
"""

import os
import numpy as np
import pybullet as p
import pyrosim.pyrosim as pyrosim
import constants as c


# GAIT_VARIANT sine controller (loaded from GAIT_VARIANT_PATH)
import json, math
from pathlib import Path
_GAIT = None
_GAIT_PATH = os.getenv('GAIT_VARIANT_PATH','')
if _GAIT_PATH:
    try:
        _GAIT = json.loads(Path(_GAIT_PATH).read_text(encoding='utf-8'))
    except Exception:
        _GAIT = None

def _gget(k, default=None):
    """Get a key from the loaded gait variant dict, or return default when no variant is loaded."""
    return default if _GAIT is None else _GAIT.get(k, default)

class MOTOR:
    """Per-joint motor controller.

Precomputes a trajectory (motorValues) unless a gait variant is active,
then sends target angles to the joint each timestep.
"""
    def __init__(self, jointName):
        """Create a motor controller for a joint.

Args:
    jointName: Joint name key (bytes or str). Preserve the original type for pyrosim.

Side effects:
    - Reads env/constant overrides for frequency/amplitude and demo modes.
    - Precomputes motorValues for the whole run when no GAIT_VARIANT is active.
"""
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
        """Send a target position command for this joint at timestep t.

Args:
    robot: ROBOT instance (must expose robotId).
    t: integer timestep index.
    max_force: max motor force in Newtons.

Behavior:
    - If a gait variant is loaded, compute a sine target angle on the fly.
    - Otherwise, use the precomputed motorValues[t].
"""

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
                try:
                    pyrosim.Set_Motor_For_Joint(bodyIndex=robot.robotId, jointName=self.jointName, controlMode=p.POSITION_CONTROL, targetPosition=angle, maxForce=float(mf))
                except TypeError:
                    pyrosim.Set_Motor_For_Joint(robot.robotId, self.jointName, p.POSITION_CONTROL, angle, float(mf))
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

# NOTE: Removed the auto-wrap GAIT_VARIANT monkeypatch block.
# Use the explicit GAIT_VARIANT logic inside MOTOR.Set_Value instead.
