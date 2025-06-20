from node import *
import numpy as np

def main():
    # Encoded message
    encoded = [0, 1, 0, 1, 1, 1, 0, 0, 1, 1]

    # Initiate generator
    G = np.array([[1, 1, 0, 1],
                [1, 1, 1, 1]])
    G_HEIGHT, G_WIDTH = G.shape
    print(G_WIDTH, G_HEIGHT)

    # Initiate trellis
    trellis = [[Node(len(encoded)) for _ in range(2**(G_WIDTH-1))] for _ in range(len(encoded)//G_HEIGHT + 1)]


    pass

if __name__ == "__main__":
    main()