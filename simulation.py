import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim

import constants as c
from world import WORLD
from robot import ROBOT

def safe_get_base_pose(body_id):
    try:
        return safe_get_base_pose(body_id)
    except Exception:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)



from tools.telemetry.logger import TelemetryLogger

import signal

signal.signal(signal.SIGINT, signal.default_int_handler)

class SIMULATION:
    def __init__(self):
        use_gui = os.getenv('HEADLESS','').lower() not in ('1','true','yes','on')

        mode = p.GUI if use_gui else p.DIRECT
        p.connect(mode)


        self.use_gui = use_gui
        # Optional: cleaner GUI
        if use_gui:
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            except Exception:
                pass

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, getattr(c, "GRAVITY_Z", -9.8))
        p.setTimeStep(getattr(c, "DT", 1/240))

        self.world = WORLD()
        self.robot = ROBOT()

    def Run(self):
        SIM_STEPS = int(os.getenv('SIM_STEPS', str(getattr(c, 'SIM_STEPS', 2000))))
        robotId = self.robot.robotId
        robot = self.robot
        # telemetry (optional)
        _telemetry_on = os.getenv('TELEMETRY','').lower() in ('1','true','yes','on')
        _telemetry_every = int(os.getenv('TELEMETRY_EVERY','10'))
        _variant_id = os.getenv('TELEMETRY_VARIANT_ID','manual')
        _run_id = os.getenv('TELEMETRY_RUN_ID','run0')
        _out_dir = __import__('pathlib').Path(os.getenv('TELEMETRY_OUT','artifacts/telemetry')) / _variant_id / _run_id
        telemetry = TelemetryLogger(robot.robotId, _out_dir, every=_telemetry_every, variant_id=_variant_id, run_id=_run_id, enabled=_telemetry_on)



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
            robot.Act(i, max_force=MAX_FORCE)

            p.stepSimulation()
            telemetry.log_step(_telemetry_step)
            _telemetry_step += 1
            if sleep_time:
                time.sleep(getattr(c, "DEMO_SLEEP_TIME", sleep_time))

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
                except Exception:
                    pass
            robot.Sense(i)


            robot.Think()
            if i % 10 == 0:
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
def __del__(self):
        try:
            telemetry.finalize()
            p.disconnect()
        except Exception:
            pass
