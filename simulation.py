import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim

import constants as c
from world import WORLD
from robot import ROBOT
from pathlib import Path
from datetime import datetime
from telemetry.trace import TraceWriter



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
        SIM_STEPS = c.SIM_STEPS
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
        start_x = p.getBasePositionAndOrientation(robotId)[0][0]
        max_x = start_x
        max_z = float("-inf")

        # Targets
        RANDOM_TARGETS = getattr(c, "RANDOM_TARGETS", False)
        RNG_SEED = getattr(c, "RNG_SEED", 0)
        TARGET_RANGE = getattr(c, "TARGET_RANGE", np.pi/2)
        SINE_CYCLES = getattr(c, "SINE_CYCLES", 3)
        RULEBOOK_PATH = os.getenv("RULEBOOK", "").strip()
        rulebook = None
        RB_BINS = None
        RB_RULES = None
        if RULEBOOK_PATH:
            import json
            with open(RULEBOOK_PATH, "r", encoding="utf-8") as f:
                rulebook = json.load(f)
            RB_BINS = int(rulebook.get("bins", 64))
            RB_RULES = rulebook.get("rules", {})
            print("RULEBOOK", RULEBOOK_PATH, "bins", RB_BINS)
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


        # Telemetry: JSONL trace for neuron dynamics (neurons-reimagined)
        trace = None
        safe_snapshot = None
        try:
            from pathlib import Path
            from datetime import datetime
            import atexit
            from telemetry.trace import TraceWriter
            from telemetry.nn_snapshot import safe_snapshot_nn as _safe_snapshot_nn
            safe_snapshot = _safe_snapshot_nn
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_path = Path("artifacts/traces") / f"run_{stamp}.jsonl"
            trace = TraceWriter(trace_path, run_meta={'type':'meta','script':'simulation.py'})
            atexit.register(trace.close)
        except Exception as e:
            trace = None
            safe_snapshot = None
            print(f"[telemetry] disabled: {e}")
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

            if trace is not None and safe_snapshot is not None:
                # B-mode (RULEBOOK): overwrite motor commands based on phase bin + contact bits
                if rulebook is not None:
                    phase = (2.0*3.141592653589793*float(SINE_CYCLES))*float(i)/max(1.0,float(SIM_STEPS-1))
                    frac = (phase % (2.0*3.141592653589793)) / (2.0*3.141592653589793)
                    b = int(frac * float(RB_BINS)) % int(RB_BINS)
                    bits = ""
                    for sid in (0,1,2):
                        try:
                            v = float(robot.nn.neurons[sid].value)
                        except Exception:
                            v = 0.0
                        bits += "1" if v > 0.0 else "0"
                    mp = RB_RULES.get(str(b), {}) or {}
                    r = mp.get(bits) or mp.get("000") or (next(iter(mp.values()), None) if mp else None)
                    if r is not None:
                        back = float(r.get("Torso_BackLeg", 0.0))
                        front = float(r.get("Torso_FrontLeg", 0.0))
                        pyrosim.Set_Motor_For_Joint(robot.robotId, b"Torso_BackLeg", p.POSITION_CONTROL, back, float(MAX_FORCE))
                        pyrosim.Set_Motor_For_Joint(robot.robotId, b"Torso_FrontLeg", p.POSITION_CONTROL, front, float(MAX_FORCE))
                        d = getattr(robot, "_last_motor_targets", None) or {}
                        d["Torso_BackLeg"] = back
                        d["Torso_FrontLeg"] = front
                        setattr(robot, "_last_motor_targets", d)
                
                # Open-loop A-mode: mirror actuator commands into motor neurons (IDs 3,4)
                try:
                    mt = getattr(robot, '_last_motor_targets', None) or {}
                    back = mt.get('Torso_BackLeg')
                    front = mt.get('Torso_FrontLeg')
                    nn = getattr(robot, 'nn', None)
                    neurons = getattr(nn, 'neurons', None) if nn is not None else None
                    if isinstance(neurons, dict):
                        for nid, val in ((3, back), (4, front)):
                            if val is None:
                                continue
                            n = neurons.get(nid) or neurons.get(str(nid))
                            if n is not None and hasattr(n, 'value'):
                                n.value = float(val)
                except Exception:
                    pass
                trace.write({'type':'step','i':i,'t_norm': float(i)/max(1.0,float(SIM_STEPS-1)),'phase': (2.0*np.pi*float(SINE_CYCLES))*float(i)/max(1.0,float(SIM_STEPS-1)),'target':float(current_target),'motor_targets': getattr(robot, '_last_motor_targets', None),'base_pos': list(p.getBasePositionAndOrientation(robot.robotId)[0]), 'base_orn': list(p.getBasePositionAndOrientation(robot.robotId)[1]), 'nn': safe_snapshot(robot)})
            p.stepSimulation()
            telemetry.log_step(_telemetry_step)
            _telemetry_step += 1
            if sleep_time:
                time.sleep(getattr(c, "DEMO_SLEEP_TIME", sleep_time))

            pos, _ = p.getBasePositionAndOrientation(robotId)
            x, z = pos[0], pos[2]
            max_x = max(max_x, x)
            max_z = max(max_z, z)

            if follow_cam:
                try:
                    p.resetDebugVisualizerCamera(
                        cameraDistance=2.0,
                        cameraYaw=45,
                        cameraPitch=-30,
                        cameraTargetPosition=pos,
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

        end_x = p.getBasePositionAndOrientation(robotId)[0][0]
        os.makedirs("data", exist_ok=True)
        np.save("data/targetAngles.npy", targetAngles)

        print("DX", end_x - start_x, "MAX_DX", max_x - start_x, flush=True)
        print("MAX_Z", max_z, "MAX_FORCE", MAX_FORCE, "RANDOM_TARGETS", RANDOM_TARGETS, flush=True)

        if getattr(self, 'use_gui', False):
            input('Done. Press Enter to close the GUI...')
def __del__(self):
        try:
            telemetry.finalize()
            p.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    # Run the simulation when executing: python3 simulation.py
    SIMULATION().Run()
