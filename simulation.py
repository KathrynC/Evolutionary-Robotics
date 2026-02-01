import pybullet as p
import pybullet_data
import constants as c

from world import WORLD
from robot import ROBOT

class SIMULATION:
    def __init__(self):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, c.GRAVITY_Z)
        p.setTimeStep(c.DT)

        self.world = WORLD()
        self.robot = ROBOT()
    def Run(self):
        import os
        import numpy
        import pybullet as p
        import pyrosim.pyrosim as pyrosim
        import constants as c

        robotId = self.robot.robotId
        robot = self.robot
        SIM_STEPS = c.SIM_STEPS

        back_key = b"Torso_BackLeg" if b"Torso_BackLeg" in pyrosim.jointNamesToIndices else "Torso_BackLeg"
        back_j = pyrosim.jointNamesToIndices[back_key]
        
        
        front_key = b"Torso_FrontLeg" if b"Torso_FrontLeg" in pyrosim.jointNamesToIndices else "Torso_FrontLeg"
        front_j = pyrosim.jointNamesToIndices[front_key]
        # Sensor vectors live in robot.sensors[...].values
        RANDOM_TARGETS = c.RANDOM_TARGETS
        RNG_SEED = c.RNG_SEED
        TARGET_RANGE = c.TARGET_RANGE
        SINE_CYCLES = c.SINE_CYCLES
        SINE_SCALE = numpy.pi/4  # 1.0 gives range [-1,+1]; later set to numpy.pi/4
        
        if RANDOM_TARGETS:
        
            rng = numpy.random.default_rng(RNG_SEED)
        
            targetAngles = rng.uniform(-TARGET_RANGE, TARGET_RANGE, size=SIM_STEPS)
        
        else:
        
            t = numpy.linspace(0.0, 2.0 * numpy.pi * SINE_CYCLES, SIM_STEPS)
        
            targetAngles = (numpy.sin(t) * SINE_SCALE)
        
        # Optional: give the robot a little "kick" so contacts change even without a GUI.
        # We apply a short sideways force to the torso (link index -1) early in the run.
        KICK_START = c.KICK_START
        KICK_END = c.KICK_END
        KICK_FORCE = [250, 0, 0]  # adjust magnitude if needed
        
        # Motor experiment parameters (edit these)
        TARGET = -numpy.pi/4
        MAX_FORCE = float(os.getenv("MAX_FORCE", str(c.MAX_FORCE)))
        max_z = float('-inf')
        for i in range(SIM_STEPS):
            current_target = targetAngles[i]
            current_target = targetAngles[i]
            current_target = targetAngles[i]
            current_target = targetAngles[i]
            current_target = targetAngles[i]
            back_angle = float(current_target)
            front_angle = float(-current_target)
            if KICK_START <= i <= KICK_END:
                p.applyExternalForce(
                    objectUniqueId=robotId,
                    linkIndex=-1,              # torso/root link
                    forceObj=KICK_FORCE,
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                )
        
            robot.Act(i, back_angle=float(current_target), front_angle=float(-current_target), max_force=MAX_FORCE)
            p.stepSimulation()
        
        
            z = p.getBasePositionAndOrientation(robotId)[0][2]
            max_z = max(max_z, z)
            robot.Sense(i)
        # Print occasionally (keeps output readable and avoids slowing the sim)
            if i % 10 == 0:
                back_angle = p.getJointState(robotId, back_j)[0]
                front_angle = p.getJointState(robotId, front_j)[0]
                print(i, robot.sensors["BackLeg"].values[i], robot.sensors["FrontLeg"].values[i], "back", back_angle, "front", front_angle, flush=True)
        os.makedirs("data", exist_ok=True)
        numpy.save("data/targetAngles.npy", targetAngles)
        print("MAX_Z", max_z, "MAX_FORCE", MAX_FORCE, "RANDOM_TARGETS", RANDOM_TARGETS, flush=True)
        p.disconnect()
        
        

    def __del__(self):
        try:
            p.disconnect()
        except Exception:
            pass
