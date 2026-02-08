"""analyze.py

Quick plotting helper for touch sensor traces saved as .npy arrays.

Expected inputs:
    - data/backLegSensorValues.npy
    - data/frontLegSensorValues.npy

These arrays are typically produced by your simulation/sensing pipeline and contain one
sensor value per timestep (length ~ constants.SIM_STEPS).

Usage:
    python3 analyze.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Load saved touch sensor time-series
back = np.load("data/backLegSensorValues.npy")
front = np.load("data/frontLegSensorValues.npy")

# Plot with a slightly thicker back-leg line for visibility
plt.plot(back, label="BackLeg", linewidth=3)
plt.plot(front, label="FrontLeg")
plt.legend()
plt.show()
