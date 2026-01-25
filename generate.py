import pyrosim.pyrosim as pyrosim

def Create_World():
    pyrosim.Start_SDF("world.sdf")
    pyrosim.Send_Cube(name="WorldBlock", pos=[3, 0, 0.5], size=[1, 1, 1])
    pyrosim.End()

def Create_Robot():
    pyrosim.Start_URDF("body.urdf")

    # Root link: absolute coordinates
    pyrosim.Send_Cube(name="Link0", pos=[0, 0, 0.5], size=[1, 1, 1])

    # First joint: absolute coordinates
    pyrosim.Send_Joint(
        name="Link0_Link1",
        parent="Link0",
        child="Link1",
        type="revolute",
        position=[0, 0, 1.0]
    )

    # Child link: relative to its joint
    pyrosim.Send_Cube(name="Link1", pos=[0, 0, 0.5], size=[1, 1, 1])

    pyrosim.End()

if __name__ == "__main__":
    Create_World()
    Create_Robot()
