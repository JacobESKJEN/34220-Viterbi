class Node:
    def __init__(self, minError):
        self.in0 = 0
        self.in1 = 0
        self.in0Val = []
        self.in1Val = []
        self.minError = minError
        self.decOut = 0
        self.cameFrom = 0