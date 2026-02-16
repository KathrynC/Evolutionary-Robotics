import copy
import os
from solution import SOLUTION
import constants as c

class PARALLEL_HILL_CLIMBER:
    def __init__(self):
        self.nextAvailableID = 0
        self.parents = {}
        for i in range(c.populationSize):
            self.parents[i] = SOLUTION(self.nextAvailableID)
            self.nextAvailableID += 1

    def Evolve(self):
        self.Evaluate(self.parents)
        for currentGeneration in range(c.numberOfGenerations):
            self.Evolve_For_One_Generation()

    def Evolve_For_One_Generation(self):
        self.Spawn()
        self.Mutate()
        self.Evaluate(self.children)
        self.Print()
        self.Select()

    def Spawn(self):
        self.children = {}
        for i in self.parents:
            self.children[i] = copy.deepcopy(self.parents[i])
            self.children[i].Set_ID(self.nextAvailableID)
            self.nextAvailableID += 1

    def Mutate(self):
        for i in self.children:
            self.children[i].Mutate()

    def Evaluate(self, solutions):
        for i in solutions:
            solutions[i].Start_Simulation("DIRECT")
        for i in solutions:
            solutions[i].Wait_For_Simulation_To_End()

    def Print(self):
        for i in self.parents:
            print("parent:", self.parents[i].fitness, "child:", self.children[i].fitness, end="  ")
        print()

    def Select(self):
        for i in self.parents:
            if self.children[i].fitness < self.parents[i].fitness:
                self.parents[i] = self.children[i]
                self.parents[i].Set_ID(self.parents[i].myID)

    def Show_Best(self):
        bestFitness = None
        bestParent = None
        for i in self.parents:
            if bestFitness is None or self.parents[i].fitness < bestFitness:
                bestFitness = self.parents[i].fitness
                bestParent = i
        print("Best fitness:", bestFitness)
        self.parents[bestParent].Evaluate("GUI")

    def Clean_Up(self):
        for filename in os.listdir("."):
            if filename.startswith("brain") and filename.endswith(".nndf"):
                os.remove(filename)
            if filename.startswith("fitness") and filename.endswith(".txt"):
                os.remove(filename)
