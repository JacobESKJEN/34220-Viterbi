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
    # Encoded message (message was: [1 0 0 1 0])
    encoded = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

    # Initiate generator
    G = np.array([[1, 1, 0, 1],
                [1, 1, 1, 1]])

    #foldningskode = Foldningskode(["1101", "1111"])
    #kodet_bitstreng = foldningskode.encode("0101110011")
    #print(kodet_bitstreng)
    G_HEIGHT, G_WIDTH = G.shape
    print(G_WIDTH, G_HEIGHT)
    print(intToBin(3, 5))
    # Initiate trellis
    trellis = [[Node(len(encoded)) for _ in range(2**(G_WIDTH-1))] for _ in range(len(encoded)//G_HEIGHT + 1)]
    

if __name__ == "__main__":
    main()