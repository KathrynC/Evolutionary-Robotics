"""robot.py

Role:
    Robot wrapper around a PyBullet-loaded URDF plus optional Pyrosim neural controller.

Responsibilities:
    - Load the robot body URDF and (optionally) a neural network brain (brain.nndf)
    - Call pyrosim.Prepare_To_Simulate() to populate name->index maps
    - Construct SENSOR and MOTOR objects for links/joints discovered by pyrosim
    - Apply motor commands each timestep via either:
        (a) neural-network motor neurons (if present), or
        (b) per-joint MOTOR trajectories (fallback)

Notes / gotchas:
    - Typical loop is Sense(t) -> Think() -> Act(t). If Think() isn't called, NN outputs may be stale.
    - pyrosim joint/link names may be bytes; some APIs expect bytes rather than str.
    - This module uses a friction tweak to reduce bounce and improve locomotion stability.
"""

import pybullet as p
import pyrosim.pyrosim as pyrosim
from pyrosim.neuralNetwork import NEURAL_NETWORK
import constants as c

from sensor import SENSOR
from motor import MOTOR

class ROBOT:
    """Robot instance: loads URDF/brain, prepares pyrosim maps, and owns sensors/motors."""
    def __init__(self, robotId=None, already_prepared=False):
        """Initialize the robot.

Args:
    robotId: If None, load body.urdf and brain.nndf; otherwise wrap an existing PyBullet body id.
    already_prepared: If True, skip pyrosim.Prepare_To_Simulate() for an externally-prepared body.

Side effects:
    - Loads URDF (if robotId is None)
    - Creates/loads neural network brain (if robotId is None)
    - Calls pyrosim.Prepare_To_Simulate()
    - Tweaks dynamics (friction/restitution) for stability
    - Populates self.sensors and self.motors
"""
        if robotId is None:
            self.robotId = p.loadURDF("body.urdf", flags=(getattr(p,"URDF_USE_SELF_COLLISION",0) | getattr(p,"URDF_USE_SELF_COLLISION_EXCLUDE_PARENT",0)))
            self.nn = NEURAL_NETWORK("brain.nndf")
            pyrosim.Prepare_To_Simulate(self.robotId)
        else:
            self.robotId = robotId
            if not already_prepared:
                pyrosim.Prepare_To_Simulate(self.robotId)

        # Friction helps locomotion; kill bounce
        try:
            mu = float(getattr(c, "ROBOT_FRICTION", 2.0))
            for link in range(-1, p.getNumJoints(self.robotId)):
                p.changeDynamics(self.robotId, link, lateralFriction=mu, restitution=0.0)
        except Exception:
            pass

        self.Prepare_To_Sense()
        self.Prepare_To_Act()

    def Prepare_To_Sense(self):
        """Create SENSOR objects for each link discovered by pyrosim.

Uses:
    pyrosim.linkNamesToIndices (populated by pyrosim.Prepare_To_Simulate).
"""
        self.sensors = {}
        for k in pyrosim.linkNamesToIndices:
            name = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            self.sensors[name] = SENSOR(name)

    def Sense(self, t: int):
        """Update all sensors at timestep t.

Args:
    t: integer timestep index.
"""
        for s in self.sensors.values():
            s.Get_Value(t)

    def Prepare_To_Act(self):
        """Create MOTOR objects for each joint discovered by pyrosim.

Uses:
    pyrosim.jointNamesToIndices (populated by pyrosim.Prepare_To_Simulate).
"""
        self.motors = {}
        for jointName in pyrosim.jointNamesToIndices:
            self.motors[jointName] = MOTOR(jointName)

    # Accept both old-style and new-style calls; angles are ignored when motors have trajectories.
    def Think(self):
        """Advance the neural network one step (if present).

Note:
    Printing every step is expensive; consider gating Print() behind an env var for long runs.
"""
        self.nn.Update()
        self.nn.Print()
    def Act(self, t: int, max_force: float=None, back_angle=None, front_angle=None, **kwargs):
        """Apply motor commands for timestep t.

Strategy:
    1) If a neural network exists and contains motor neurons, use its outputs to drive joints.
       (Caller should ensure the NN was updated via Think().)
    2) Otherwise, fall back to per-joint MOTOR trajectories (self.motors).

Args:
    t: integer timestep index.
    max_force: motor force limit (Newtons). If None, taken from kwargs/MAX_FORCE.
    back_angle/front_angle: legacy args (ignored when motors have trajectories).
    **kwargs: compatibility and optional overrides.
"""
        if max_force is None:
            max_force = float(kwargs.get("MAX_FORCE", 500.0))
        if hasattr(self, "nn"):
            used_nn = False
            for neuronName in self.nn.neurons:
                n = self.nn.neurons[neuronName]
                if n.Is_Motor_Neuron():
                    jointName = n.Get_Joint_Name()

                    jointName = jointName.encode("ASCII") if isinstance(jointName, str) else jointName
                    desiredAngle = n.Get_Value()
                    try:
                        pyrosim.Set_Motor_For_Joint(self.robotId, jointName, desiredAngle, max_force)
                    except TypeError:
                        pyrosim.Set_Motor_For_Joint(self.robotId, jointName, p.POSITION_CONTROL, desiredAngle, max_force)
                    used_nn = True
            if used_nn:
                return
        for m in self.motors.values():
            m.Set_Value(self, t, max_force)
