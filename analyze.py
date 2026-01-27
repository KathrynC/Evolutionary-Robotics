import numpy as np
import matplotlib.pyplot as plt

back = np.load("data/backLegSensorValues.npy")
front = np.load("data/frontLegSensorValues.npy")

plt.plot(back, label="BackLeg", linewidth=3)
plt.plot(front, label="FrontLeg")
plt.legend()
plt.show()
