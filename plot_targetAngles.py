import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI windows
import matplotlib.pyplot as plt

path = os.path.join("data", "targetAngles.npy")
if not os.path.exists(path):
    raise SystemExit("Missing data/targetAngles.npy (run simulate.py first).")

y = np.load(path)

plt.figure()
plt.plot(y)
plt.xlabel("timestep")
plt.ylabel("targetAngles (radians)")
plt.title("Motor command vector: targetAngles")
out = os.path.join("data", "targetAngles.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("WROTE", out)
