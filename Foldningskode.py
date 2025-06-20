from sympy import *
import numpy as np
from collections import deque

class Foldningskode:
    def __init__(self, *generatorer):
        self.generator_list = generatorer
        self.antal_generatorer = len(self.generator_list)

        self.G = np.array(self.generator_list)
        self.n = self.G.shape[0]
        self.M = self.G.shape[1]-1
        print("n =", self.n, "M =", self.M)

        self.T = np.zeros((2, self.n, 2**self.M), dtype=int)

    def flip_bits(self, b):
        t = ""
        for i in b:
            if i == "1":
                t += "0"
            else:
                t += "1"
        return t

    def compare_bitstring(self, b1, b2):
        b = ""
        if type(b1) == type([]):
            b1 = "".join(str(x) for x in b1)
        if type(b2) == type([]):
            b2 = "".join(str(x) for x in b2)
        for i in range(len(b1)):
            if b1[i] == b2[i]:
                b += "1"
            else:
                b += "0"
        return b

    def determine_weight(self, b):
        n = 0
        for i in b:
            if i=="1":
                n+=1
        return n


    def opretTabelT(self):
        b = "000"
        updated = [False for _ in range(2**self.M)]
        for i in range(2**self.M):
            b = format(i, "b")
            b = "0"*(self.M-len(b))+b
            b = self.flip_bits(b)
            
            for m in self.generator_list:
                if not updated[i]:
                    self.T[0][0][i] = str(self.determine_weight(self.compare_bitstring(m, "0"+b)) % 2) # for første bit er 0 (og man sætter forbindelsen)
                    self.T[0][1][i] = str(self.determine_weight(self.compare_bitstring(m, "1"+b)) % 2) # for første bit er 1 (og man sætter forbindelsen)
                    updated[i] = True
                else:
                    self.T[0][0][i] = str(self.T[0][0][i]) + str(self.determine_weight(self.compare_bitstring(m, "0"+b)) % 2) # for første bit er 0 (og man sætter forbindelsen)
                    self.T[0][1][i] = str(self.T[0][1][i]) + str(self.determine_weight(self.compare_bitstring(m, "1"+b)) % 2) # for første bit er 1 (og man sætter forbindelsen)

                self.T[1][0][i] = int("0"+b, 2) # for første bit er 0 (og man sætter tilstanden)
                self.T[1][1][i] = int("1"+b, 2) # for første bit er 1 (og man sætter tilstanden)

    def encode(self, sequence):
        tilstand = "0"*self.M
        result = ""
        for i in sequence:
            result += "0"*(self.n-len(str(self.T[0][int(i)][int(tilstand, 2)]))) + str(self.T[0][int(i)][int(tilstand, 2)])
            tilstand = tilstand[1::] + str(i)
            
        return result
