"""simulate.py

CLI entrypoint for running a single simulation episode.

Usage:
    python3 simulate.py

Notes:
    - This file intentionally does very little: it constructs SIMULATION and calls Run().
    - Most configuration is controlled via constants.py and environment variables
      (e.g., HEADLESS, SIM_STEPS, MAX_FORCE, SLEEP_TIME, GAIT_MODE, GAIT_VARIANT_PATH).
"""

from pathlib import Path
import os

from simulation import SIMULATION
from tools.telemetry.logger import TelemetryLogger


if __name__ == "__main__":
    simulation = SIMULATION()
    simulation.Run()
