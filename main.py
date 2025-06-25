from node import *
import numpy as np
from commpy import channelcoding
import imageio.v3 as iio
from jpeg_compression_cycle import *
from Huffman_minVar import *
import matplotlib.pyplot as plt

def intToBin(intInput, length): # Didn't have to be reversed anyway
    binary = format(intInput, "b")
    binary = "0"*(length-len(binary))+binary
    result = []
    for item in binary:
        result.append(int(item))
    return result
    """for item_i in range(len(binary)):
        if binary[item_i-1] == "-":
            continue
        item = binary[item_i]
        if item != "-":
            result.append(int(item))
        else:
            result[0] = result[0]
            result.append(int(item + binary[item_i+1]))
    print(result, int("".join(str(x) for x in result), 2))
    return result"""

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
    encoded = []
    #encoded = np.zeros(0, dtype=int)

    for i in range(len(message)):
        andOperand = np.concatenate(([message[i]], shiftRegister))
        for j in range(G_HEIGHT):
            andResult = np.sum(andOperand * G[j]) % 2
            encoded.append(andResult)
        
        shiftRegister = np.roll(shiftRegister, 1)
        shiftRegister[0] = message[i]
    print("Encoded")
    return np.array(encoded)

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
    print("Adding noise")
    sigma = np.sqrt(0.5*10**(-ratioInDB/10))
    noise = sigma*np.random.randn(len(array))

    array = (array - 0.5) * 2

    return array + noise, noise

def puncture(encodedMessage, puncturePattern):
    # INPUTS:
    # encodedMessage:  The encoded message to puncture
    # puncturePattern: The pattern to be used for puncturing
    # OUTPUTS:
    # The encoded message but punctured

    output = []
    puncturePattern = (puncturePattern.T).flatten()
    for i in range(len(encodedMessage)):
        if(puncturePattern[i % len(puncturePattern)] == 1):
            output.append(encodedMessage[i])
    print("Punctured")
    return np.array(output)

def patchPunctures(puncturedMessage, puncturePattern):
    # INPUTS:
    # puncturedMessage: The punctures message to patch
    # puncturePattern:  The pattern used for puncturing
    # OUTPUTS:
    # The encoded message but with punctures patched

    #output = np.array([])
    output = []
    puncturePattern = (puncturePattern.T).flatten()

    punctureIndex = 0
    patternIndex = 0

    while punctureIndex < len(puncturedMessage):
        if(puncturePattern[patternIndex % len(puncturePattern)] == 1):
            output.append(puncturedMessage[punctureIndex])
            punctureIndex += 1
        else:
            output.append(0)
        
        patternIndex += 1

    return np.array(output)

def viterbiSearchAndBacktrack(trellis, encoded, G, start_column=1):
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

def viterbiDecode(G, encodedWithNoiseAndPunctures):
    G_HEIGHT, G_WIDTH = G.shape

    puncturePattern = np.array([[1, 0], [1, 1]]) # NOTE: Breaks if it does not have the same "height" as G
    
    M = G_WIDTH-1
    L = 9 * M
    
    encoded = patchPunctures(encodedWithNoiseAndPunctures, puncturePattern.copy()) # Fixing punctures
    print("Punctures fixed")
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

    print("Trellis initiated")
    decoded = []

    # Trellis search
    for t in range(len(trellis[0])):
        trellis[0][t].minError = 0
        trellis[0][t].cameFrom = -1

    print("Backtracking")
    i = 0
    while i*L + 2*L < (len(encoded)//G_HEIGHT + 1):
        window_trellis = trellis[i*L:i*L+2*L]  
        start_column_index = L if i > 0 else 1
        #print("Start",start_column_index)
        for column_index in range(start_column_index,len(window_trellis)):
            for node_index in range(len(window_trellis[column_index])):
                window_trellis[column_index][node_index].minError = len(encoded)
        
        for j in range(len(window_trellis[0])):
            window_trellis[0][j].cameFrom = -1

        part_of_encoded = encoded[i*L*G_HEIGHT:(i*L+2*L)*G_HEIGHT]
        
        output, updated_trellis = viterbiSearchAndBacktrack(window_trellis, part_of_encoded, G, start_column=start_column_index-1)
        trellis[i*L:i*L+2*L] = updated_trellis
        
        decoded += output[0:L]
        i += 1
    return decoded
    
def encodeHuffman(huffmantable, toEncode):
    result = []
    n = 0
    sorted_uniques = np.unique(np.array(toEncode))
    if len(toEncode.shape) == 2:
        for i in range(len(toEncode)):
            for j in range(len(toEncode[i])):
                index = np.where(sorted_uniques == toEncode[i][j])
                n+=1
                #print(n/(len(toEncode)*len(toEncode[i])))
                result.append(huffmantable[index][0][1])
    else:
        for i in range(len(toEncode)):
            index = np.where(sorted_uniques == toEncode[i])
            
            n+=1
            #print(n/(len(toEncode)*len(toEncode[i])))
            result.append(huffmantable[index][0][1])
            if i < 20:
                #print(index, huffmantable[index], toEncode[i])
                pass
    return "".join(str(x) for x in result)

def main():
    img = iio.imread("rsc/I52.png")
    height, width = img.shape[:2]
    qf = .5
    img_jpeg, compressed = jpeg_compression_cycle(img, qf)

    #decompressed = decode_jpeg(compressed[0], compressed[1], compressed[2], .1)
    #print(compressed)
    combined = np.concatenate((compressed[0].flatten(),compressed[1].flatten(), compressed[2].flatten()))
    #print(combined)
    #print("Length should be:", combined.shape)
    chars, counts = np.unique(combined, return_counts=True)
     
    freq = counts / sum(counts)
    huffmantable, huffmantree = buildHuffDictMV(chars, freq)
    
    huffman_coded = encodeHuffman(huffmantable, combined)
    print(len(huffman_coded))

    # Initiate generator
    #G = np.array([[1, 1, 1, 1],
    #            [1, 0, 1, 1],
    #            [1, 1, 1, 0]])
    G = np.array([[1,0,1],[1,1,1]])
    #G = np.array([[1,1,1,1,0,0,1],[1,0,1,1,0,1,1]])
    #message_length = 1000
    #message = [np.random.randint(0, 2, dtype=int) for _ in range(message_length)]
    message = [int(x) for x in huffman_coded]
   
    message_length=len(message)
    G_HEIGHT, G_WIDTH = G.shape
    M = G_WIDTH - 1
    L = 9 * M

    puncturePattern = np.array([[1, 0], [1, 1]]) # NOTE: Breaks if it does not have the same "height" as G
    
    encoded = viterbiEncoder(message, G)
    chars_to_remove = 0 # Remove 4 first characters
    for _ in range(chars_to_remove): # Remove 4 first characters
        message.pop(0)
        for __ in range(G_HEIGHT):
            encoded = np.delete(encoded, 0)

    ratioInDB = 10
    encodedWithPunctures = puncture(encoded, puncturePattern.copy())
    encodedWithNoiseAndPunctures, noisePattern = addNoise(ratioInDB, (encodedWithPunctures - 0.5)*2)

    message_with_noise = np.round(message + noisePattern[:len(message)]) % 2
    print(len(message), len(message_with_noise))

    output = viterbiDecode(G, np.append(encodedWithNoiseAndPunctures, np.ones(3*L))) # Smider L 0'er på enden, så man laver viterbidekodning af hele billedet
    print("Output", len(output), "Message(noise)", len(message_with_noise))


    ## Viterbi decoder from channelcoding:
    #trellisUsingPackage = channelcoding.convcode.Trellis(np.array([M]),np.array(generatorMatrixToIntsReversed(G))) # has to be in the reverse order of how it's written in G
    #decodedUsingPackage = channelcoding.convcode.viterbi_decode(encoded.copy(),trellisUsingPackage,tb_depth=None,decoding_type='soft')

    #print(f'Message: ------------------- {np.array(message, dtype=int)}')
    #print(f'Decoded using our decoder: - {np.array(output, dtype=int)}')
    #print(f'Decoded using channelcoding: {decodedUsingPackage}')
    print("Decoded message correct?:", message[0:len(output)] == output)
    #print("Decoded same as channelcoding?:",(decodedUsingPackage[0:len(output)] == output).all())
    #print("Channel coding correct?:", (decodedUsingPackage == message).all())

    decompressed_combined = decodehuff(huffmantree, "".join(str(x) for x in output))
    print(decompressed_combined[:25], combined[:25])
    print(len(decompressed_combined), len(combined))
    decompressed_luminence_flat = decompressed_combined[:width*height]
    decompressed_cb_flat = decompressed_combined[width*height: int(5/4 * width*height)]
    decompressed_cr_flat = decompressed_combined[int(5/4 *width*height): int(6/4 * width*height)+1]
    decompressed_luminence = np.zeros((height, width))
    decompressed_cb = np.zeros((height//2, width//2))
    decompressed_cr = np.zeros((height//2, width//2))
    for y in range(height):
        for x in range(width):
            decompressed_luminence[y][x] = decompressed_luminence_flat[width*y+x]
    for y in range(height//2):
        for x in range(width//2):
            decompressed_cb[y][x] = decompressed_cb_flat[(width//2)*y+x] 
            decompressed_cr[y][x] = decompressed_cr_flat[(width//2)*y+x] 
    
    decompressed_conv_code = decode_jpeg(decompressed_luminence, decompressed_cb, decompressed_cr, qf)


    decompressed_combined = decodehuff(huffmantree, "".join(str(x) for x in message_with_noise))
    
    decompressed_luminence_flat = decompressed_combined[:width*height]
    decompressed_cb_flat = decompressed_combined[width*height: int(5/4 * width*height)]
    decompressed_cr_flat = decompressed_combined[int(5/4 *width*height): int(6/4 * width*height)+1]
    print("Lengths:", len(decompressed_combined), len(decompressed_luminence_flat), len(decompressed_cb_flat), len(decompressed_cr_flat),len(decompressed_luminence_flat) + len(decompressed_cb_flat) + len(decompressed_cr_flat))
    decompressed_luminence = np.zeros((height, width))
    decompressed_cb = np.zeros((height//2, width//2))
    decompressed_cr = np.zeros((height//2, width//2))
    for y in range(height):
        for x in range(width):
            decompressed_luminence[y][x] = decompressed_luminence_flat[width*y+x]
    for y in range(height//2):
        for x in range(width//2):
            decompressed_cb[y][x] = decompressed_cb_flat[(width//2)*y+x] 
            #print(len(decompressed_cr_flat), (width//2)*y+x)
            decompressed_cr[y][x] = decompressed_cr_flat[(width//2)*y+x] 

    decompressed_no_conv_code = decode_jpeg(decompressed_luminence, decompressed_cb, decompressed_cr, qf)


    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(img)
    ax[1].imshow(decompressed_no_conv_code)
    ax[2].imshow(decompressed_conv_code)
    plt.show(block=True)


if __name__ == "__main__":
    main()