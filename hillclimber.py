import copy
import os
from solution import SOLUTION
import constants as c

class HILL_CLIMBER:
    def __init__(self):
        self.parent = SOLUTION(0)

    def Evolve(self):
        self.parent.Evaluate("DIRECT")
        for currentGeneration in range(c.numberOfGenerations):
            self.Evolve_For_One_Generation()

    def Evolve_For_One_Generation(self):
        self.Spawn()
        self.Mutate()
        self.child.Evaluate("DIRECT")
        self.Print()
        self.Select()

    def Spawn(self):
        self.child = copy.deepcopy(self.parent)
        self.child.Set_ID(1)

    def Mutate(self):
        self.child.Mutate()

    def Print(self):
        print("parent fitness:", self.parent.fitness, "child fitness:", self.child.fitness)

    def Select(self):
        if self.child.fitness < self.parent.fitness:
            self.parent = self.child
            self.parent.Set_ID(0)

    def Show_Best(self):
        self.parent.Evaluate("GUI")

    def Clean_Up(self):
        for filename in os.listdir("."):
            if filename.startswith("brain") and filename.endswith(".nndf"):
                os.remove(filename)
            if filename.startswith("fitness") and filename.endswith(".txt"):
                os.remove(filename)
