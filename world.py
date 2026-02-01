import pybullet as p

class WORLD:
    def __init__(self):
        # Loads belong here (Gate D)
        self.planeId = p.loadURDF("plane.urdf")
        self.worldIds = p.loadSDF("world.sdf")
