import numpy as np

def quantCoder(x, s):
    buffer = 0
    d = []
    result = []
    for i in x:
        d.append(i-buffer)
        result.append(np.ceil((i-buffer)/s))
        buffer += s*np.ceil((i-buffer)/s)
    return result

def quantDecoder(d, s):
    result = []
    buffer = 0
    for i in d:
        buffer += s*i
        result.append(int(buffer))
    return result