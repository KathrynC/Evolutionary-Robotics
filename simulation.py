import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim

import constants as c
from world import WORLD
from robot import ROBOT


class SIMULATION:
    def __init__(self):
        use_gui = os.getenv("PYBULLET_GUI", "1") == "1"
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


        if os.getenv("PRINT_MOTOR_FREQS","1") == "1":
            for m in getattr(robot, "motors", {}).values():
                if ("BackLeg" in m.jointNameStr) or ("FrontLeg" in m.jointNameStr):
                    print("[MOTOR]", m.jointNameStr, "freq_hz", getattr(m, "freq_hz", None))
        # Locomotion metrics
        try:
        try:
        start_x = p.getBasePositionAndOrientation(robot.robotId)
        except Exception:
            pos = (0.0, 0.0, 0.0)
        except Exception:
            pos = (0.0, 0.0, 0.0)
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
            if sleep_time:
                time.sleep(getattr(c, "DEMO_SLEEP_TIME", sleep_time))

            try:
            try:
            pos, _ = p.getBasePositionAndOrientation(robot.robotId)
            except Exception:
                pos = (0.0, 0.0, 0.0)
            except Exception:
                pos = (0.0, 0.0, 0.0)
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

        try:
        try:
        end_x = p.getBasePositionAndOrientation(robot.robotId)
        except Exception:
            pos = (0.0, 0.0, 0.0)
        except Exception:
            pos = (0.0, 0.0, 0.0)
        os.makedirs("data", exist_ok=True)
        np.save("data/targetAngles.npy", targetAngles)

        print("DX", end_x - start_x, "MAX_DX", max_x - start_x, flush=True)
        print("MAX_Z", max_z, "MAX_FORCE", MAX_FORCE, "RANDOM_TARGETS", RANDOM_TARGETS, flush=True)

        if getattr(self, 'use_gui', False):
            input('Done. Press Enter to close the GUI...')
def __del__(self):
        try:
            p.disconnect()
        except Exception:
            pass
