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

def findMetric(received, expected):
    # INPUTS:
    # received: array of received values
    # expected: array of expected values (with values {0, 1})
    # OUTPUT:
    # 'the amount of errors' between the two (not actual amount of errors but simplified number)

    metricValues = np.zeros(len(expected))
    expected = (expected - 0.5) * 2

    for i in range(len(expected)):
        if(received[i] >= 0 and expected[i] == -1):
            metricValues[i] = np.abs(received[i])
        
        if(received[i] < 0 and expected[i] == 1):
            metricValues[i] = np.abs(received[i])

    return np.sum(metricValues)

def addNoise(ratioInDB, array):
    # INPUTS:
    # ratioInDB: signal to noise ratio expressed in dB
    # array:     the signal to add noise to (with values {0, 1})
    # OUTPUTS:
    # an array with added noise (with decimal values centered around {-1, 1})

    sigma = np.sqrt(0.5*10**(-ratioInDB/10))
    noise = sigma*np.random.randn(len(array))

    array = (array - 0.5) * 2

    return array + noise

def puncture(encodedMessage, puncturePattern):
    # INPUTS:
    # encodedMessage:  The encoded message to puncture
    # puncturePattern: The pattern to be used for puncturing
    # OUTPUTS:
    # The encoded message but punctured

    output = np.array([])
    puncturePattern = (puncturePattern.T).flatten()

    for i in range(len(encodedMessage)):
        if(puncturePattern[i % len(puncturePattern)] == 1):
            output = np.append(output, encodedMessage[i])

    return output

def patchPunctures(puncturedMessage, puncturePattern):
    # INPUTS:
    # puncturedMessage: The punctures message to patch
    # puncturePattern:  The pattern used for puncturing
    # OUTPUTS:
    # The encoded message but with punctures patched

    output = np.array([])
    puncturePattern = (puncturePattern.T).flatten()

    punctureIndex = 0
    patternIndex = 0

    while punctureIndex < len(puncturedMessage):
        if(puncturePattern[patternIndex % len(puncturePattern)] == 1):
            output = np.append(output, puncturedMessage[punctureIndex])
            punctureIndex += 1
        else:
            output = np.append(output, 0)
        
        patternIndex += 1

    return output

def viterbiDecode(trellis, encoded, G, start_column=1):
    G_HEIGHT, G_WIDTH = G.shape
    M = G_WIDTH - 1

    # Trellis search
    #for i in range(len(trellis[0])):
    #    trellis[0][i].minError = 0
   
    for column_index in range(start_column, len(trellis)-1):
        for node_index in range(len(trellis[column_index])):
            node = trellis[column_index][node_index]
            if node == 0:
                continue
            toDecode = encoded[column_index*G_HEIGHT:(column_index+1)*G_HEIGHT]
            
            errors0 = findMetric(toDecode, node.out0)
            errors1 = findMetric(toDecode, node.out1)
            
            if trellis[column_index+1][node.in0].minError > node.minError + errors0:
                trellis[column_index+1][node.in0].minError = node.minError + errors0
                trellis[column_index+1][node.in0].cameFrom = node_index
                trellis[column_index+1][node.in0].decOut = 0

            if trellis[column_index+1][node.in1].minError > node.minError + errors1:
                trellis[column_index+1][node.in1].minError = node.minError + errors1
                trellis[column_index+1][node.in1].cameFrom = node_index
                trellis[column_index+1][node.in1].decOut = 1
    
    trellis_distances = np.zeros((len(trellis),len(trellis[0])))
    for column_index in range(len(trellis)):
        for node_index in range(len(trellis[column_index])):
            trellis_distances[column_index, node_index] = trellis[column_index][node_index].minError

    #print(trellis_distances.T)

    # Backtrack
    currentNode = trellis[len(trellis)-1][0]
    for node in trellis[len(trellis)-1]:
        if node.minError < currentNode.minError:
            currentNode = node
    
    output = []
    layer = len(trellis)-1
    while currentNode.cameFrom != -1:
        output.append(currentNode.decOut)
        layer -= 1
        currentNode = trellis[layer][currentNode.cameFrom]
    
    output = output[::-1]
    return output, trellis

def main():
    # Message to encode
    #message = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0]
    message_length = 1000
    message = [np.random.randint(0, 2, dtype=int) for _ in range(message_length)]

    # Initiate generator
    #G = np.array([[1, 1, 1, 1],
    #            [1, 0, 1, 1],
    #            [1, 1, 1, 0]])
    #G = np.array([[1,0,1],[1,1,1]])
    G = np.array([[1,1,1,1,0,0,1],[1,0,1,1,0,1,1]])
    G_HEIGHT, G_WIDTH = G.shape

    puncturePattern = np.array([[1, 0], [1, 1]]) # NOTE: Breaks if it does not have the same "height" as G
    
    encoded = viterbiEncoder(message, G)
    chars_to_remove = 5 # Remove 4 first characters
    for _ in range(chars_to_remove): # Remove 4 first characters
        message.pop(0)
        for __ in range(G_HEIGHT):
            encoded = np.delete(encoded, 0)

    ratioInDB = 2
    encodedWithNoise = addNoise(ratioInDB, (encoded - 0.5)*2)
    encodedWithNoiseAndPunctures = puncture(encodedWithNoise, puncturePattern.copy())
    
    M = G_WIDTH-1
    L = 9 * M
    
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

    decoded = []

    for t in range(len(trellis[0])):
        trellis[0][t].minError = 0
        trellis[0][t].cameFrom = -1
    
    # Trellis search
    #for i in range((len(encoded)//G_HEIGHT + 1)//L-1):
    i = 0
    #encoded = encodedWithNoise
    encoded = patchPunctures(encodedWithNoiseAndPunctures, puncturePattern.copy()) # Fixing punctures
    while i*L + 2*L < (len(encoded)//G_HEIGHT + 1):
        print(i*L, i*L+2*L)
        window_trellis = trellis[i*L:i*L+2*L]  
        start_column_index = L if i > 0 else 1
        print("Start",start_column_index)
        for column_index in range(start_column_index,len(window_trellis)):
            for node_index in range(len(window_trellis[column_index])):
                window_trellis[column_index][node_index].minError = len(encoded)
        
        for j in range(len(window_trellis[0])):
            window_trellis[0][j].cameFrom = -1

        part_of_encoded = encoded[i*L*G_HEIGHT:(i*L+2*L)*G_HEIGHT]
        
        output, updated_trellis = viterbiDecode(window_trellis, part_of_encoded, G, start_column=start_column_index-1)
        trellis[i*L:i*L+2*L] = updated_trellis
        
        decoded += output[0:L]
        print("Decoded",len(decoded), L)
        print(decoded)
        print(message[:i*L+L])
        print(decoded == message[:i*L+L])
        i += 1
    
    output = decoded

    ## Viterbi decoder from channelcoding:
    trellisUsingPackage = channelcoding.convcode.Trellis(np.array([M]),np.array(generatorMatrixToIntsReversed(G))) # has to be in the reverse order of how it's written in G
    decodedUsingPackage = channelcoding.convcode.viterbi_decode(encoded.copy(),trellisUsingPackage,tb_depth=None,decoding_type='soft')

    print(f'Message: ------------------- {np.array(message, dtype=int)}')
    print(f'Decoded using our decoder: - {np.array(decoded, dtype=int)}')
    print(f'Decoded using channelcoding: {decodedUsingPackage}')
    print("Decoded message correct?:", message[0:len(output)] == output)
    print("Decoded same as channelcoding?:",(decodedUsingPackage[0:len(output)] == output).all())
    print("Channel coding correct?:", (decodedUsingPackage == message).all())
    print(len(output), len(message), L)


if __name__ == "__main__":
    main()