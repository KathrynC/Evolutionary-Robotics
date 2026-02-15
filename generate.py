"""
generate.py

Artifact generator for the Evolutionary Robotics toy robot.

This script produces the core files that most other scripts depend on:
    - world.sdf   : simple SDF scene (optional in many runners)
    - body.urdf   : the robot's physical structure (links + joints)
    - brain.nndf  : a minimal neural network (sensor + motor neurons + synapses)

Why this matters:
    - If body.urdf is missing, many scripts will fail immediately.
    - If brain.nndf is missing, NN-driven control paths may fail.

Usage:
    python3 generate.py

Outputs (in the current directory):
    world.sdf, body.urdf, brain.nndf

Notes:
    - The robot is intentionally minimal: a torso and two legs connected by revolute joints.
    - The NN here is a starter scaffold; later assignments typically evolve this structure.

Ludobots role:
  - Module: E. Joints (world + robot description files)
  - Produces: world.sdf (environment), body.urdf (robot body)
  - Consumed by: simulation.py / simulate.py via PyBullet

Run:
  python3 generate.py

Beyond Ludobots (this repo):
  - (Fill in any variant-parameterized morphology/world generation you added.)
"""

import random
import pyrosim.pyrosim as pyrosim


def Create_World():
    """Create a minimal SDF scene in world.sdf.

    Current content:
        - One cube named "WorldBlock" placed at x=3 so it's out of the way of the robot.

    Many runners load only plane.urdf and ignore world.sdf; this file exists mainly as
    a convenient place to add obstacles later.
    """
    pyrosim.Start_SDF("world.sdf")
    pyrosim.Send_Cube(name="WorldBlock", pos=[3, 0, 0.5], size=[1, 1, 1])
    pyrosim.End()


def Create_Robot():
    """Create the robot URDF (body.urdf): torso + two legs with revolute joints.

    Link layout (roughly):
        - Torso: centered at z=1.5 (a 1x1x1 cube)
        - BackLeg: attached at the rear of the torso
        - FrontLeg: attached at the front of the torso

    Joint naming is important because other scripts refer to these exact names:
        - "Torso_BackLeg"
        - "Torso_FrontLeg"
    """
    pyrosim.Start_URDF("body.urdf")

    # Base link
    pyrosim.Send_Cube(name="Torso", pos=[0, 0, 1.5], size=[1, 1, 1])

    # Rear leg
    pyrosim.Send_Joint(
        name="Torso_BackLeg",
        parent="Torso",
        child="BackLeg",
        type="revolute",
        position=[-0.5, 0, 1.0],
    )
    pyrosim.Send_Cube(name="BackLeg", pos=[-0.5, 0, -0.5], size=[1, 1, 1])

    # Front leg
    pyrosim.Send_Joint(
        name="Torso_FrontLeg",
        parent="Torso",
        child="FrontLeg",
        type="revolute",
        position=[0.5, 0, 1.0],
    )
    pyrosim.Send_Cube(name="FrontLeg", pos=[0.5, 0, -0.5], size=[1, 1, 1])

    pyrosim.End()


def Generate_Body():
    """Generate the physical artifacts: world.sdf and body.urdf."""
    Create_World()
    Create_Robot()


def Generate_Brain():
    """Generate a minimal neural controller in brain.nndf.

    Neurons:
        - Sensor neurons:
            0: Torso
            1: BackLeg
            2: FrontLeg
        - Motor neurons:
            3: Torso_BackLeg joint
            4: Torso_FrontLeg joint

    Synapses:
        Fully connected: each of the 3 sensor neurons connects to each of the 2 motor
        neurons (6 synapses total), with random weights in [-1, 1].

    Each call produces a different random brain.
    """
    pyrosim.Start_NeuralNetwork("brain.nndf")

    sensorNeurons = [0, 1, 2]
    sensorLinks   = ["Torso", "BackLeg", "FrontLeg"]
    motorNeurons  = [3, 4]
    motorJoints   = ["Torso_BackLeg", "Torso_FrontLeg"]

    for name, link in zip(sensorNeurons, sensorLinks):
        pyrosim.Send_Sensor_Neuron(name=name, linkName=link)

    for name, joint in zip(motorNeurons, motorJoints):
        pyrosim.Send_Motor_Neuron(name=name, jointName=joint)

    for sensorName in sensorNeurons:
        for motorName in motorNeurons:
            weight = random.uniform(-1, 1)
            pyrosim.Send_Synapse(sourceNeuronName=sensorName,
                                 targetNeuronName=motorName,
                                 weight=weight)

    pyrosim.End()


def Generate_Brain_Extended():
    """Generate a 22-synapse neural controller with touch AND proximity sensors.

    Neurons:
        - Touch sensor neurons (0-2): Torso, BackLeg, FrontLeg
        - Motor neurons (3-4): Torso_BackLeg, Torso_FrontLeg
        - Proximity neurons (5-12): 6 torso faces + 2 leg downward

    Synapses:
        - 6 touch→motor (unchanged from classic)
        - 16 proximity→motor (8 proximity sensors × 2 motors)
        Total: 22 synapses
    """
    pyrosim.Start_NeuralNetwork("brain.nndf")

    # Touch sensor neurons (unchanged)
    sensorNeurons = [0, 1, 2]
    sensorLinks   = ["Torso", "BackLeg", "FrontLeg"]
    motorNeurons  = [3, 4]
    motorJoints   = ["Torso_BackLeg", "Torso_FrontLeg"]

    for name, link in zip(sensorNeurons, sensorLinks):
        pyrosim.Send_Sensor_Neuron(name=name, linkName=link)

    for name, joint in zip(motorNeurons, motorJoints):
        pyrosim.Send_Motor_Neuron(name=name, jointName=joint)

    # Proximity neurons
    proximityNeurons = [
        (5,  "Torso",    "front"),
        (6,  "Torso",    "back"),
        (7,  "Torso",    "left"),
        (8,  "Torso",    "right"),
        (9,  "Torso",    "up"),
        (10, "Torso",    "down"),
        (11, "BackLeg",  "down"),
        (12, "FrontLeg", "down"),
    ]

    for nid, linkName, rayDir in proximityNeurons:
        pyrosim.Send_Proximity_Neuron(name=nid, linkName=linkName, rayDir=rayDir)

    # Touch→motor synapses (6)
    for sensorName in sensorNeurons:
        for motorName in motorNeurons:
            weight = random.uniform(-1, 1)
            pyrosim.Send_Synapse(sourceNeuronName=sensorName,
                                 targetNeuronName=motorName,
                                 weight=weight)

    # Proximity→motor synapses (16)
    for nid, _, _ in proximityNeurons:
        for motorName in motorNeurons:
            weight = random.uniform(-1, 1)
            pyrosim.Send_Synapse(sourceNeuronName=nid,
                                 targetNeuronName=motorName,
                                 weight=weight)

    pyrosim.End()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate robot body and brain files.")
    parser.add_argument("--extended", action="store_true",
                        help="Generate 22-synapse brain with proximity sensors (default: classic 6-synapse)")
    args = parser.parse_args()

    Generate_Body()
    if args.extended:
        Generate_Brain_Extended()
    else:
        Generate_Brain()
