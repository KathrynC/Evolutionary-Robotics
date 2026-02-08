"""
world.py

Role:
    Set up the simulation environment in PyBullet.

What it loads:
    - A ground plane URDF ("plane.urdf") to provide a stable floor.
    - Optionally, an SDF file ("world.sdf") if it appears to define a real <world>.

Safety behavior:
    Some projects keep a "world.sdf" that is not a true <world> file (e.g., a standalone model),
    which can spawn unexpected objects like a stray cube. To avoid surprises, this module checks
    whether the file contains a <world> element before loading it.

Overrides:
    FORCE_LOAD_WORLD_SDF=1  -> load world.sdf regardless of whether it contains <world>.

Ludobots role:
  - Provides the World class used by the simulator to load world assets (plane.urdf + world.sdf).
  - Part of the 'world + robot' split introduced around E. Joints.

Beyond Ludobots (this repo):
  - (Document any additional world features: friction knobs, obstacles, cameras, etc.)
"""

import os
from pathlib import Path

import pybullet as p


class WORLD:
    """PyBullet world setup: ground plane plus optional SDF world content."""

    def __init__(self):
        """Create the world.

        Side effects:
            - Loads "plane.urdf" into the physics world.
            - Optionally loads "world.sdf" when it contains a <world> element (or forced by env).
        """
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
            if os.getenv('SIM_DEBUG','0') == '1':
                print(
                f"[WORLD] Skipping {world_file}: no <world> element (prevents stray cube). "
                f"Set FORCE_LOAD_WORLD_SDF=1 to force-load."
                )
