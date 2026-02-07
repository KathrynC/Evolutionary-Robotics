from pathlib import Path
import os
from simulation import SIMULATION

from tools.telemetry.logger import TelemetryLogger

if __name__ == "__main__":
    simulation = SIMULATION()
    simulation.Run()
