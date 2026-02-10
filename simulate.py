"""
simulate.py

CLI entrypoint for running a single simulation episode.

Usage:
    python3 simulate.py

Notes:
    - This file intentionally does very little: it constructs SIMULATION and calls Run().
    - Most configuration is controlled via constants.py and environment variables
      that are read inside simulation.py (e.g., HEADLESS, SIM_STEPS, MAX_FORCE,
      SLEEP_TIME, GAIT_MODE, GAIT_VARIANT_PATH, TELEMETRY*).

Ludobots role:
  - Canonical early-course runner used to step the physics engine.
  - Loads world + robot assets and advances the simulation loop.

Beyond Ludobots (this repo):
  - May be retained as a wrapper/legacy runner if simulation.py is canonical (verify).
"""

from simulation import SIMULATION


if __name__ == "__main__":
    simulation = SIMULATION()
    simulation.Run()
