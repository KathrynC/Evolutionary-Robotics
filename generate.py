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
        - Sensors excite/drive the back leg motor neuron (positive weights).
        - Sensors inhibit/drive the front leg motor neuron (negative weights).

    This produces a simple, asymmetric baseline controller that can be replaced later.
    """
    pyrosim.Start_NeuralNetwork("brain.nndf")

    # Sensors (touch/contact-style sensors in pyrosim)
    pyrosim.Send_Sensor_Neuron(name=0, linkName="Torso")
    pyrosim.Send_Sensor_Neuron(name=1, linkName="BackLeg")
    pyrosim.Send_Sensor_Neuron(name=2, linkName="FrontLeg")

    # Motors
    pyrosim.Send_Motor_Neuron(name=3, jointName="Torso_BackLeg")
    pyrosim.Send_Motor_Neuron(name=4, jointName="Torso_FrontLeg")

    # Simple fixed synapses (starter scaffold)
    pyrosim.Send_Synapse(sourceNeuronName=0, targetNeuronName=3, weight=1.0)
    pyrosim.Send_Synapse(sourceNeuronName=1, targetNeuronName=3, weight=1.0)
    pyrosim.Send_Synapse(sourceNeuronName=2, targetNeuronName=3, weight=1.0)

    pyrosim.Send_Synapse(sourceNeuronName=0, targetNeuronName=4, weight=-1.0)
    pyrosim.Send_Synapse(sourceNeuronName=1, targetNeuronName=4, weight=-1.0)
    pyrosim.Send_Synapse(sourceNeuronName=2, targetNeuronName=4, weight=-1.0)

    pyrosim.End()


if __name__ == "__main__":
    Generate_Body()
    Generate_Brain()
