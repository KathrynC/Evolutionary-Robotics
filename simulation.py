"""
simulation.py

Role:
    Runs a single PyBullet simulation episode.

What happens:
    - Connects to PyBullet (GUI or DIRECT) based on HEADLESS.
    - Configures physics (gravity + timestep).
    - Builds WORLD and ROBOT.
    - Steps the simulation for SIM_STEPS and applies motor commands each step.
    - Optionally logs telemetry.

Entrypoint:
    Usually launched via `python3 simulate.py` (this module may not call Run() on import).

Config precedence:
    Environment variables > constants.py > defaults in code.

Common env vars:
    HEADLESS            : 1/true => run in DIRECT; otherwise GUI
    SIM_STEPS           : number of steps in this episode
    MAX_FORCE           : motor force limit
    SLEEP_TIME          : sleep between steps (seconds) when DEMO_SLEEP_TIME is not set
    DEMO_SLEEP_TIME     : preferred sleep between steps for recording/visibility
    SIM_DEBUG           : print debug banners at startup

Gait/debug env vars:
    GAIT_MODE           : 1 => direct-drive joints from GAIT_VARIANT_PATH (bypass Robot.Act)
    GAIT_VARIANT_PATH   : path to gait JSON used by GAIT_MODE (and/or motor.py variant logic)
    GAIT_HALF_BACK      : when GAIT_MODE=1, optionally halve back-leg frequency

Telemetry env vars:
    TELEMETRY, TELEMETRY_EVERY, TELEMETRY_OUT, TELEMETRY_VARIANT_ID, TELEMETRY_RUN_ID

Notes:
    - Units: angles=radians, time=seconds, frequency=Hz, force=Newtons.
    - If you want the neural network to influence motion, the typical architecture is
      Sense(t) -> Think() -> Act(t). (Some code paths may call Act directly.)

Ludobots role:
  - Simulation runner: sets up PyBullet, loads world/robot, steps SIM_STEPS, and calls Robot.Act(t).

This repo’s extensions:
  - HEADLESS / GUI toggle via env var
  - SLEEP_TIME override for demo pacing
  - GAIT_VARIANT_PATH support for experiment variants
  - Telemetry output (jsonl + summaries) when enabled

Run:
  python3 simulation.py
"""

import os
import pybullet as p

# [GAIT_MODE] direct drive override
import json, math
from pathlib import Path as _Path

_GAIT_PATH = os.getenv("GAIT_VARIANT_PATH", "")
_GAIT = None
if _GAIT_PATH:
    try:
        _GAIT = json.loads(_Path(_GAIT_PATH).read_text(encoding="utf-8"))
    except Exception:
        _GAIT = None

def _gget(k, default=None):
    return default if _GAIT is None else _GAIT.get(k, default)

_GAIT_MODE = os.getenv("GAIT_MODE", "0") == "1"
_GAIT_HALF_BACK = os.getenv("GAIT_HALF_BACK", "1") == "1"
_GAIT_BACK_JOINT = os.getenv("GAIT_BACK_JOINT", "Torso_BackLeg")
_GAIT_FRONT_JOINT = os.getenv("GAIT_FRONT_JOINT", "Torso_FrontLeg")

if os.getenv("SIM_DEBUG","0") == "1":
    print("[SIMFILE]", __file__, flush=True)
    print("[ENV] GAIT_VARIANT_PATH", os.getenv("GAIT_VARIANT_PATH","<none>"), flush=True)

# Silence pyrosim neuralNetwork debug prints unless PYROSIM_NN_VERBOSE=1
if os.getenv("PYROSIM_NN_VERBOSE","0") != "1":
    try:
        import pyrosim.neuralNetwork as _nn
        _nn.print = (lambda *a, **k: None)
    except Exception as e:
        if os.getenv('SIM_DEBUG','0') == '1':
            print('[WARN]', __name__, 'suppressed exception:', repr(e), flush=True)

import time
import numpy as np
import pybullet_data
import pyrosim.pyrosim as pyrosim

import constants as c
from world import WORLD
from robot import ROBOT

def safe_get_base_pose(body_id):
    """Return (pos, orn) for a body; fall back to safe defaults on PyBullet errors."""
    try:
        return p.getBasePositionAndOrientation(body_id)
    except Exception:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)



from tools.telemetry.logger import TelemetryLogger

import signal

signal.signal(signal.SIGINT, signal.default_int_handler)

class SIMULATION:
    """Owns one PyBullet connection, one WORLD, and one ROBOT; runs a single episode."""
    def __init__(self):
        """Connect to PyBullet (GUI or DIRECT), configure physics, then build WORLD and ROBOT."""
        use_gui = os.getenv('HEADLESS','').lower() not in ('1','true','yes','on')

        mode = p.GUI if use_gui else p.DIRECT
        p.connect(mode)


        self.use_gui = use_gui
        # Optional: cleaner GUI
        if use_gui:
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            except Exception as e:
                if os.getenv('SIM_DEBUG','0') == '1':
                    print('[WARN]', __name__, 'suppressed exception:', repr(e), flush=True)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, getattr(c, "GRAVITY_Z", -9.8))
        p.setTimeStep(getattr(c, "DT", 1/240))

        self.world = WORLD()
        self.robot = ROBOT()

    def Run(self):


        """Run a fixed-length simulation episode.


        Each timestep:

            - (optional) apply a "kick" external force for debugging

            - drive motion either via:

                * GAIT_MODE direct joint drive (bypasses Robot.Act), or

                * normal control path (Robot.Act), optionally preceded by Sense/Think

            - step PyBullet physics

            - log telemetry (optional)

            - sleep for visibility (DEMO_SLEEP_TIME override preferred)


        Side effects:

            - Opens a GUI window when HEADLESS is falsey

            - Writes telemetry artifacts when enabled

        """
        SIM_STEPS = int(os.getenv("SIM_STEPS", str(getattr(c, "SIM_STEPS", 2000))))
        robotId = self.robot.robotId
        robot = self.robot
        # telemetry (optional)
        _telemetry_on = os.getenv('TELEMETRY','').lower() in ('1','true','yes','on')
        _telemetry_every = int(os.getenv('TELEMETRY_EVERY','10'))
        _variant_id = os.getenv('TELEMETRY_VARIANT_ID','manual')
        _run_id = os.getenv('TELEMETRY_RUN_ID','run0')
        _out_dir = __import__('pathlib').Path(os.getenv('TELEMETRY_OUT','artifacts/telemetry')) / _variant_id / _run_id
        if _telemetry_on:
            _out_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry = TelemetryLogger(robot.robotId, _out_dir, every=_telemetry_every, variant_id=_variant_id, run_id=_run_id, enabled=_telemetry_on)
        telemetry = self.telemetry  # Keep local variable for compatibility



        if os.getenv("PRINT_VARIANT","0") == "1":
            print("[VARIANT] GAIT_VARIANT_PATH", os.getenv("GAIT_VARIANT_PATH",""), flush=True)

        # ensure summary.json even if GUI closes / Ctrl-C
        __import__('atexit').register(telemetry.finalize)

        _telemetry_step = 0
        if os.getenv("PRINT_MOTOR_FREQS","1") == "1":
            for m in getattr(robot, "motors", {}).values():
                if ("BackLeg" in m.jointNameStr) or ("FrontLeg" in m.jointNameStr):
                    print("[MOTOR]", m.jointNameStr, "freq_hz", getattr(m, "freq_hz", None))
        # Locomotion metrics
        _start_pos = safe_get_base_pose(robotId)[0]
        start_x = float(_start_pos[0])
        start_y = float(_start_pos[1])
        max_x = start_x
        max_z = float("-inf")

        # Targets
        RANDOM_TARGETS = getattr(c, "RANDOM_TARGETS", False)
        RNG_SEED = getattr(c, "RNG_SEED", 0)
        TARGET_RANGE = getattr(c, "TARGET_RANGE", np.pi/2)
        SINE_CYCLES = getattr(c, "SINE_CYCLES", 3)
        SINE_SCALE = np.pi/4

        if RANDOM_TARGETS:
            rng = np.random.default_rng(RNG_SEED)
            targetAngles = rng.uniform(-TARGET_RANGE, TARGET_RANGE, size=SIM_STEPS)
        else:
            t = np.linspace(0.0, 2.0 * np.pi * SINE_CYCLES, SIM_STEPS)
            targetAngles = np.sin(t) * SINE_SCALE

        # Optional kick (off by default)
        enable_kick = os.getenv("ENABLE_KICK", "0") == "1"
        KICK_START = getattr(c, "KICK_START", 200)
        KICK_END = getattr(c, "KICK_END", 350)
        KICK_FORCE = [250, 0, 0]

        MAX_FORCE = float(os.getenv("MAX_FORCE", str(getattr(c, "MAX_FORCE", 500.0))))

        follow_cam = getattr(self, 'use_gui', False) and (os.getenv('FOLLOW_CAMERA','1') == '1')
        sleep_time = float(os.getenv('SLEEP_TIME', str(getattr(c, 'SLEEP_TIME', 0.0))))

        # Run loop
        try:
            for i in range(SIM_STEPS):
                current_target = targetAngles[i]

                if enable_kick and KICK_START <= i <= KICK_END:
                    p.applyExternalForce(
                        objectUniqueId=robotId,
                        linkIndex=-1,
                        forceObj=KICK_FORCE,
                        posObj=[0, 0, 0],
                        flags=p.LINK_FRAME,
                    )

                # Motors drive themselves (motor.py builds trajectories; HALF_FREQ_DEMO env var can be used there)
                # [GAIT_MODE] if enabled, bypass motor/neural plumbing and drive joints directly
                if _GAIT_MODE and _GAIT is not None:
                    # timestep (seconds)
                    try:
                        dt = float(os.getenv('PHYSICS_DT', os.getenv('BULLET_DT', str(getattr(c, 'DT', getattr(c, 'TIME_STEP', 1/240.0))))))
                    except Exception:
                        dt = float(os.getenv('PHYSICS_DT', '0.0041666667'))
                    tsec = float(i) * dt
                    A = float(_gget('GAIT_AMPLITUDE', _gget('GAIT_AMP', 0.5)))
                    base_f = float(_gget('GAIT_FREQ_HZ', _gget('GAIT_FREQ', 1.0)))
                    back_f = base_f * (0.5 if _GAIT_HALF_BACK else 1.0)
                    front_f = base_f
                    back_O = float(_gget('BACK_OFFSET', 0.0))
                    back_phi = float(_gget('BACK_PHASE', 0.0))
                    front_O = float(_gget('FRONT_OFFSET', 0.0))
                    front_phi = float(_gget('FRONT_PHASE', 0.0))
                    back_angle = back_O + A * math.sin(2.0 * math.pi * back_f * tsec + back_phi)
                    front_angle = front_O + A * math.sin(2.0 * math.pi * front_f * tsec + front_phi)
                    mf = float(os.getenv('MAX_FORCE', str(_gget('MAX_FORCE', getattr(c, 'MAX_FORCE', 500.0)))))
                    try:
                        import pyrosim.pyrosim as pyrosim
                        try:
                            pyrosim.Set_Motor_For_Joint(bodyIndex=robot.robotId, jointName=_GAIT_BACK_JOINT, controlMode=p.POSITION_CONTROL, targetPosition=float(back_angle), maxForce=float(mf))
                            pyrosim.Set_Motor_For_Joint(bodyIndex=robot.robotId, jointName=_GAIT_FRONT_JOINT, controlMode=p.POSITION_CONTROL, targetPosition=float(front_angle), maxForce=float(mf))
                        except TypeError:
                            pyrosim.Set_Motor_For_Joint(robot.robotId, _GAIT_BACK_JOINT, p.POSITION_CONTROL, float(back_angle), float(mf))
                            pyrosim.Set_Motor_For_Joint(robot.robotId, _GAIT_FRONT_JOINT, p.POSITION_CONTROL, float(front_angle), float(mf))
                    except Exception as e:
                        if os.getenv('SIM_DEBUG','0') == '1':
                            print('[WARN]', __name__, 'suppressed exception:', repr(e), flush=True)
                    if os.getenv('SIM_DEBUG','0') == '1' and i == 0:
                        print('[GAITMODE]', 'A', A, 'base_f', base_f, 'back_f', back_f, 'front_f', front_f, 'mf', mf, flush=True)
                        print('[GAITMODE]', 'back', _GAIT_BACK_JOINT, 'O', back_O, 'phi', back_phi, flush=True)
                        print('[GAITMODE]', 'front', _GAIT_FRONT_JOINT, 'O', front_O, 'phi', front_phi, flush=True)
                    # Intentionally do NOT call robot.Act() in this mode (it may overwrite our targets).
                else:
                    # Sense -> Think -> Act (course architecture)
                    try:
                        robot.Sense(i)
                    except Exception as e:
                        if os.getenv('SIM_DEBUG','0') == '1':
                            print('[WARN]', __name__, 'suppressed exception:', repr(e), flush=True)
                    try:
                        if hasattr(robot, 'nn'):
                            robot.Think()
                    except Exception as e:
                        if os.getenv('SIM_DEBUG','0') == '1':
                            print('[WARN]', __name__, 'suppressed exception:', repr(e), flush=True)
                    robot.Act(i, max_force=MAX_FORCE)

                p.stepSimulation()
                if _telemetry_on:
                    if _telemetry_on:
                        if _telemetry_on:
                            telemetry.log_step(_telemetry_step)
                _telemetry_step += 1
                if sleep_time:
                    time.sleep(float(os.getenv("DEMO_SLEEP_TIME", str(getattr(c, "DEMO_SLEEP_TIME", sleep_time)))))

                pos, _ = safe_get_base_pose(robotId)
                x, z = pos[0], pos[2]
                max_x = max(max_x, x)
                max_z = max(max_z, z)

                if bool(getattr(c, 'CAMERA_FOLLOW', False)) and (i % int(getattr(c, 'CAMERA_EVERY_N', 1)) == 0):
                    try:
                        rid = getattr(robot, 'robotId', None)
                        if rid is not None:
                            pos, _ = p.getBasePositionAndOrientation(rid)
                            target = [pos[0], pos[1], pos[2] + float(getattr(c, 'CAMERA_TARGET_Z', 0.5))]
                            p.resetDebugVisualizerCamera(
                                cameraDistance=float(getattr(c, 'CAMERA_DISTANCE', 3.0)),
                                cameraYaw=float(getattr(c, 'CAMERA_YAW', 60.0)),
                                cameraPitch=float(getattr(c, 'CAMERA_PITCH', -25.0)),
                                cameraTargetPosition=target,
                            )
                    except Exception as e:
                        if os.getenv('SIM_DEBUG','0') == '1':
                            print('[WARN]', __name__, 'suppressed exception:', repr(e), flush=True)
                if os.getenv("DEBUG_PRINT","0")=="1" and i % 10 == 0:
                    # keep your familiar debug print, but avoid joint-index assumptions
                    bl = robot.sensors.get("BackLeg")
                    fl = robot.sensors.get("FrontLeg")
                    blv = bl.values[i] if bl else None
                    flv = fl.values[i] if fl else None
                    print(i, "back", blv, "front", flv, flush=True)

            end_x = safe_get_base_pose(robotId)[0][0]
            os.makedirs("data", exist_ok=True)
            np.save("data/targetAngles.npy", targetAngles)

            print("DX", end_x - start_x, "MAX_DX", max_x - start_x, flush=True)
            print("MAX_Z", max_z, "MAX_FORCE", MAX_FORCE, "RANDOM_TARGETS", RANDOM_TARGETS, flush=True)
            # Write per-run summary.json for zoo scoring
            try:
                import json
                _end_pos = safe_get_base_pose(robotId)[0]
                end_x = float(_end_pos[0])
                end_y = float(_end_pos[1])

                dx = float(end_x - start_x)
                dy = float(end_y - start_y)
                dxy = float((dx*dx + dy*dy) ** 0.5)

                _out_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "variant_id": str(_variant_id),
                    "run_id": str(_run_id),
                    "dx": dx,
                    "dy": dy,
                    "dxy_net": dxy,
                    "max_dx": float(max_x - start_x),
                    "max_z": float(max_z),
                    "max_force": float(MAX_FORCE),
                    "random_targets": bool(RANDOM_TARGETS),
                    "sim_steps": int(SIM_STEPS),
                    "telemetry_enabled": bool(_telemetry_on),
                    "telemetry_every": int(_telemetry_every),
                }
                (_out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            except Exception as e:
                print("WARN: summary.json write failed:", e, flush=True)


            if getattr(self, 'use_gui', False):
                input('Done. Press Enter to close the GUI...')
        finally:
            try:
                if getattr(self, 'telemetry', None) is not None:
                    self.telemetry.finalize()
            except Exception as e:
                if os.getenv('SIM_DEBUG','0') == '1':
                    print('[WARN]', __name__, 'suppressed exception:', repr(e), flush=True)
            try:
                p.disconnect()
            except Exception as e:
                if os.getenv('SIM_DEBUG','0') == '1':
                    print('[WARN]', __name__, 'suppressed exception:', repr(e), flush=True)
