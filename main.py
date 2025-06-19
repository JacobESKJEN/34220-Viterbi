from quantization import *
from Huffman_minVar import *
import yuvio
import numpy as np

def intra_predict(current_frame, blocksize, quant_step):
    frame = np.pad(current_frame, ((1,0),(1,0)), "constant", constant_values=(128, 128))


def inter_predict(current_frame, reference_frame, blocksize, quant_step):
    pass


def encode(current_frame, blocksize, quant_step, immType="I", refererence_frame=None):
    pass

def main():
    pass

if __name__ == "__main__":
    main()