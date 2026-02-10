import time
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as pyrosim

SIM_STEPS = 6000
DT = 1 / 240

def safe_touch(link_name: str) -> float:
    try:
        return pyrosim.Get_Touch_Sensor_Value_For_Link(link_name)
    except TypeError:
        # Sometimes getContactPoints() returns None on macOS; treat as "no contact"
        return -1.0

# --- GUI setup ---
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)        # hide side panels
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)    # optional

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(DT)

p.loadURDF("plane.urdf")
p.loadSDF("world.sdf")
robotId = p.loadURDF("body.urdf")

pyrosim.Prepare_To_Simulate(robotId)

# Camera framing
p.resetDebugVisualizerCamera(
    cameraDistance=3.0,
    cameraYaw=45,
    cameraPitch=-25,
    cameraTargetPosition=[0, 0, 0.7],
)

# Pause so you can arrange windows before recording
input("\nArrange/resize the PyBullet window + Terminal now. Press Enter to start the simulation... ")

# --- Simulation loop ---
for i in range(SIM_STEPS):
    # Gentle poke so contacts happen without mouse interaction
    if 240 <= i <= 1200:
        p.applyExternalForce(
            objectUniqueId=robotId,
            linkIndex=-1,          # torso/root link
            forceObj=[2000, 0, 0], # stronger poke for visible switching
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
        )

    p.stepSimulation()

    back  = safe_touch("BackLeg")
    front = safe_touch("FrontLeg")

    if i % 10 == 0:
        print(i, back, front, flush=True)

    time.sleep(DT)

p.disconnect()
