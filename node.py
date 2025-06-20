import numpy as np

class Node:
    def __init__(self, minError):
        self.in0 = 0
        self.in1 = 0
        self.out0 = np.array([], dtype=int)
        self.out1 = np.array([], dtype=int)
        self.minError = minError
        self.decOut = 0
        self.cameFrom = 0