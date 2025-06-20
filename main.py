from node import *
from Foldningskode import Foldningskode
import numpy as np

def intToBin(intInput, length):
    binary = format(intInput, "b")
    binary = "0"*(length-len(binary))+binary
    result = []
    for item in binary:
        result.append(int(item))
    return result

def main():
    # Encoded message
    encoded = [0, 1, 0, 1, 1, 1, 0, 0, 1, 1]

    # Initiate generator
    G = np.array([[1, 1, 0, 1],
                [1, 1, 1, 1]])

    #foldningskode = Foldningskode(["1101", "1111"])
    #kodet_bitstreng = foldningskode.encode("0101110011")
    #print(kodet_bitstreng)
    G_HEIGHT, G_WIDTH = G.shape
    M = G_WIDTH-1
    print(G_WIDTH, G_HEIGHT)
    # Initiate trellis
    trellis = [[Node(len(encoded)) for _ in range(2**M)] for _ in range(len(encoded)//G_HEIGHT + 1)]
    

if __name__ == "__main__":
    main()