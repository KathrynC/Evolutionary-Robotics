import sys
from simulation import SIMULATION

directOrGUI = sys.argv[1]
solutionID = sys.argv[2]

simulation = SIMULATION(directOrGUI, solutionID)
simulation.Run()

fitness = simulation.Get_Fitness()

print(fitness)

f = open("fitness" + str(solutionID) + ".txt", "w")
f.write(str(fitness))
f.close()
