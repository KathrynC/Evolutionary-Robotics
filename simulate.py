import time
import pybullet as p

cid = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

for _ in range(240):
    p.stepSimulation()
    time.sleep(1/60)

input("Press Enter to quit...")
p.disconnect()
print("simulate.py OK")
