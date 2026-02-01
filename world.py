import os
from pathlib import Path
import pybullet as p

class WORLD:
    def __init__(self):
        self.planeId = p.loadURDF("plane.urdf")
        self.worldIds = []

        world_file = "world.sdf"
        # Load only if it's a real *world* SDF (contains a <world> element).
        # Override with FORCE_LOAD_WORLD_SDF=1 if you want to load it anyway.
        try:
            sdf_text = Path(world_file).read_text(errors="ignore")
        except Exception:
            sdf_text = ""

        if "<world" in sdf_text or os.getenv("FORCE_LOAD_WORLD_SDF", "0") == "1":
            self.worldIds = p.loadSDF(world_file)
        else:
            print(f"[WORLD] Skipping {world_file}: no <world> element (prevents stray cube). "
                  f"Set FORCE_LOAD_WORLD_SDF=1 to force-load.")
