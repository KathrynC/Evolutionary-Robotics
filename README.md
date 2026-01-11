# Evolutionary Robotics workspace

## Activate environment
conda activate er

## Smoke test
python - <<'PY'
import time
import pybullet as p
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
for _ in range(120):
    p.stepSimulation()
    time.sleep(1/120)
p.disconnect()
print("smoke test OK")
PY
