import time
import pybullet as p

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

i = 0
try:
    while True:
        p.stepSimulation()
        time.sleep(1/60)
        i += 1
        if i % 60 == 0:
            print(i)
except KeyboardInterrupt:
    print("\nStopped with Ctrl-C")
finally:
    p.disconnect()
