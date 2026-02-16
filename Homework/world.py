import pybullet as p
import pybullet_data

class WORLD:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.loadSDF("world.sdf")
