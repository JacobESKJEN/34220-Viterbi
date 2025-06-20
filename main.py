from node import *
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
    #encoded = [1,1,1,1,1,0,0,0,0,1,0,0,0,1,1,1]

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
    for column_index in range(len(trellis)-1):
        for node_index in range(len(trellis[column_index])):
            node = trellis[column_index][node_index]
            tilstand = intToBin(node_index, M)
            tilstand_0 = np.concatenate(([0], tilstand)) # Til bestemmelse af forbindelse.
            tilstand_1 = np.concatenate(([1], tilstand)) # Til bestemmelse af forbindelse.
            output0 = np.array([])
            output1 = np.array([])
            for generator in G:
                shared1s_for_0 = np.bitwise_and(tilstand_0, generator)
                shared1s_for_1 = np.bitwise_and(tilstand_1, generator)
                output0 = np.append(output0, np.sum(shared1s_for_0)%2)
                output1 = np.append(output1, np.sum(shared1s_for_1)%2)
            node.out0 = output0
            node.out1 = output1
            node.in0 = node_index>>1
            node.in1 = (2**M+node_index)>>1
    
    # Trellis search
    trellis[0][0].minError = 0
    for column_index in range(len(trellis)-1):
        for node_index in range(len(trellis[column_index])):
            node = trellis[column_index][node_index]
            toDecode = encoded[column_index*G_HEIGHT:(column_index+1)*G_HEIGHT]
            errors0 = np.sum((toDecode + node.out0) % 2)
            errors1 = np.sum((toDecode + node.out1) % 2)

            if trellis[column_index+1][node.in0].minError > node.minError + errors0:
                trellis[column_index+1][node.in0].minError = node.minError + errors0
                trellis[column_index+1][node.in0].cameFrom = node_index
                trellis[column_index+1][node.in0].decOut = 0

            if trellis[column_index+1][node.in1].minError > node.minError + errors1:
                trellis[column_index+1][node.in1].minError = node.minError + errors1
                trellis[column_index+1][node.in1].cameFrom = node_index
                trellis[column_index+1][node.in1].decOut = 1
    
    # Backtrack
    currentNode = trellis[len(trellis)-1][0]
    for node in trellis[len(trellis)-1][:]:
        if node.minError < currentNode.minError:
            currentNode = node
    
    output = []
    layer = len(trellis)-1
    while currentNode.cameFrom != -1:
        print(currentNode.cameFrom)
        output.append(currentNode.decOut)
        layer -= 1
        currentNode = trellis[layer][currentNode.cameFrom]
    
    print(output[::-1])

    

if __name__ == "__main__":
    main()