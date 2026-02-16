import os
from solution import SOLUTION
import constants as c

for i in range(5):
    parent = SOLUTION(i)
    parent.Evaluate("DIRECT")
    print(i, parent.fitness)

# Clean up brain files
for i in range(5):
    brainFile = "brain" + str(i) + ".nndf"
    if os.path.exists(brainFile):
        os.remove(brainFile)
