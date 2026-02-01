import os
import numpy
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim
import constants as c
from robot import ROBOT

SIM_STEPS = c.SIM_STEPS
DT = c.DT# physics timestep


def main(do_setup=True, existing_robotId=None):
    # Headless physics (no GUI). This avoids the macOS GUI segfaults.
    p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, c.GRAVITY_Z)
    p.setTimeStep(DT)

    p.loadURDF("plane.urdf")
    p.loadSDF("world.sdf")
    robotId = p.loadURDF("body.urdf")


    max_z = -1e9
    # Prepare pyrosim's link/joint dictionaries and remember robotId
    pyrosim.Prepare_To_Simulate(robotId)
    robot = ROBOT(robotId=robotId, already_prepared=True)
    # Motor joint index (pyrosim may key joint names as bytes or str)
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
    for i in range(SIM_STEPS):
        back_angle = 0.0
        front_angle = 0.0
        back_angle = 0.0
        front_angle = 0.0
        current_target = targetAngles[i]
        if KICK_START <= i <= KICK_END:
            p.applyExternalForce(
                objectUniqueId=robotId,
                linkIndex=-1,              # torso/root link
                forceObj=KICK_FORCE,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
            )

        robot.Act(i, back_angle=back_angle, front_angle=front_angle, max_force=MAX_FORCE)
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
    numpy.save("data/backLegSensorValues.npy", robot.sensors["BackLeg"].values)
    numpy.save("data/frontLegSensorValues.npy", robot.sensors["FrontLeg"].values)
    numpy.save("data/targetAngles.npy", targetAngles)
    print("MAX_Z", max_z, "MAX_FORCE", MAX_FORCE, "RANDOM_TARGETS", RANDOM_TARGETS, flush=True)
    p.disconnect()


if __name__ == "__main__":
    main()

