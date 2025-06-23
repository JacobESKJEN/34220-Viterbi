from node import *
import numpy as np
from commpy import channelcoding

def intToBin(intInput, length): # Didn't have to be reversed anyway
    binary = format(intInput, "b")
    binary = "0"*(length-len(binary))+binary
    result = []
    for item in binary:
        result.append(int(item))
    return result

def generatorMatrixToIntsReversed(G): # Useful for comparing our decoder's results with the built-in channelcoding decoder
    # INPUTS:
    # G: Generator matrix
    # OUTPUTS:
    # output: Array containing decimal values corresponding to each row of G reversed

    G_HEIGHT, G_WIDTH = G.shape
    output = np.array([], dtype=int)

    for i in range(G_HEIGHT):
        intValue = 0
        for j in range(G_WIDTH):
            intValue += G[i][j]*2**j
        output = np.append(output, intValue)

    return [output] # For some reason this has to be wrapped in another array for channelCoding to accept it

def viterbiEncoder(message, G):
    # INPUTS:
    # message: An array of binary values to encode
    # G:       Generator matrix
    # OUTPUTS:
    # encoded: An array of binary values

    G_HEIGHT, G_WIDTH = G.shape
    M = G_WIDTH - 1

    print(f'G_HEIGHT: {G_HEIGHT}, G_WIDTH: {G_WIDTH}, M: {M}')

    shiftRegister = np.zeros(M, dtype=int)
    encoded = np.zeros(0, dtype=int)

    for i in range(len(message)):
        andOperand = np.concatenate(([message[i]], shiftRegister))

        for j in range(G_HEIGHT):
            andResult = np.sum(andOperand * G[j]) % 2
            encoded = np.append(encoded, andResult)
        
        shiftRegister = np.roll(shiftRegister, 1)
        shiftRegister[0] = message[i]
    
    return encoded

def viterbiDecode(trellis, encoded, G):
    G_HEIGHT, G_WIDTH = G.shape
    M = G_WIDTH - 1

    # Trellis search
    trellis[0][0].minError = 0
    for column_index in range(len(trellis)-1):
        for node_index in range(len(trellis[column_index])):
            node = trellis[column_index][node_index]
            if node == 0:
                continue
            toDecode = encoded[column_index*G_HEIGHT:(column_index+1)*G_HEIGHT]
            #if column_index==1 and node_index == 2:
            #    print("For 0 vej", toDecode, node.out0, (toDecode + node.out0) % 2)
            #    print("For 1 vej", toDecode, node.out1, (toDecode + node.out1) % 2)

            #print("toDecode:  ", toDecode)
            #print("node.out0: ", node.out0)
            
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
    
    """trellis_distances = np.zeros((len(trellis),len(trellis[0])))
    for column_index in range(len(trellis)):
        for node_index in range(len(trellis[column_index])):
            trellis_distances[column_index, node_index] = trellis[column_index][node_index].minError

    print(trellis_distances.T)"""

    # Backtrack
    currentNode = trellis[len(trellis)-1][0]
    for node in trellis[len(trellis)-1][:]:
        if node.minError < currentNode.minError:
            currentNode = node
    
    output = []
    layer = len(trellis)-1
    while currentNode.cameFrom != -1:
        #print(layer, trellis[layer].index(currentNode))
        #print(currentNode.cameFrom)
        output.append(currentNode.decOut)
        layer -= 1
        currentNode = trellis[layer][currentNode.cameFrom]
    #print(layer, trellis[layer].index(currentNode))
    output = output[::-1]
    return output

def main():
    # Message to encode
    message = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0]

    # Initiate generator
    """ G = np.array([[1, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 0]])  """
    G = np.array([[1,1,1,1,0,0,1],[1,0,1,1,0,1,1]])
    encoded = viterbiEncoder(message, G)
    for i in range(len(encoded)):
        if np.random.rand() > 0.99: # 1 procent error
            encoded[i] = (encoded[i]+1)%2

    #foldningskode = Foldningskode(["1101", "1111"])
    #kodet_bitstreng = foldningskode.encode("0101110011")
    #print(kodet_bitstreng)
    G_HEIGHT, G_WIDTH = G.shape
    M = G_WIDTH-1
    print("M", M)
    #print(G_WIDTH, G_HEIGHT)
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
            #print(format(node_index, "b"))
            #print(tilstand)
            for generator in G:
                shared1s_for_0 = np.bitwise_and(tilstand_0, generator)
                shared1s_for_1 = np.bitwise_and(tilstand_1, generator)
                output0 = np.append(output0, np.sum(shared1s_for_0)%2)
                output1 = np.append(output1, np.sum(shared1s_for_1)%2)
            node.out0 = output0
            node.out1 = output1
            node.in0 = node_index>>1 #(node_index<<1) & ((2**M)-1) 
            node.in1 = (2**M+node_index)>>1#((node_index<<1)+1) & ((2**M)-1) #

    decodedUsingMethod = viterbiDecode(trellis, encoded, G)    

    ## Viterbi decoder from channelcoding:
    trellisUsingPackage = channelcoding.convcode.Trellis(np.array([M]),np.array(generatorMatrixToIntsReversed(G))) # has to be in the reverse order of how it's written in G
    decodedUsingPackage = channelcoding.convcode.viterbi_decode(encoded.copy(),trellisUsingPackage,tb_depth=None,decoding_type='hard')

    print(f'Message: ------------------- {np.array(message, dtype=int)}')
    print(f'Decoded using our decoder: - {np.array(decodedUsingMethod, dtype=int)}')
    print(f'Decoded using channelcoding: {decodedUsingPackage}')


if __name__ == "__main__":
    main()