"""
search.py

Random search over synaptic weights.

Repeatedly generates a random brain (via generate.py) and evaluates it
(via simulate.py in headless mode). Prints the displacement (DX) for each
trial and reports the best at the end.

Ludobots role:
  - Module K: Random Search
  - Automates the generate -> simulate loop.

Usage:
    python3 search.py
"""

import os

NUM_TRIALS = 5

for i in range(NUM_TRIALS):
    os.system("python3 generate.py")
    print(f"\n--- Trial {i} ---", flush=True)
    os.system("HEADLESS=1 python3 simulate.py")
