import pybullet as p
import pybullet_data

cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
p.loadURDF("plane.urdf")

ids = p.loadSDF("box.sdf")
print("Loaded:", ids)

p.disconnect(cid)
