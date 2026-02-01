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
        # Transitional bridge: reuse existing run logic but skip setup.
        import simulate_legacy as legacy
        legacy.main(do_setup=False, existing_robotId=self.robot.robotId)

    def __del__(self):
        try:
            p.disconnect()
        except Exception:
            pass
