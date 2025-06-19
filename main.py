from quantization import *
from Huffman_minVar import *
from scipy.fftpack import dctn, idctn
import matplotlib.pyplot as plt
import yuvio
import numpy as np

def intra_predict(current_frame, blocksize, quant_step):
    frame = np.pad(current_frame, ((1,0),(1,0)), "constant", constant_values=(128, 128))
    # Loop gennem hver blok og bestem prædiktionen ud fra at tile den tidligere række og søjle tage gennemsnittet
    # af deres sum til hver pixel indenfor blokken. Bestem derefter residualerne.
    DCT = np.zeros(current_frame.shape)
    quant_diff_DCT = np.zeros(current_frame.shape)
    decoded = np.zeros(current_frame.shape)
    for irow in range(1, current_frame.shape[0], blocksize):
        for icol in range(1, current_frame.shape[1], blocksize):
            prevcol = frame[irow:irow+blocksize,icol-1].reshape(-1, 1)
            prevrow = frame[irow-1,icol:icol+blocksize].reshape(1, -1)
            pred = 0.5 * ( np.tile(prevcol, (1, blocksize)) + np.tile(prevrow, (blocksize, 1)) )
            residuals = frame[irow:irow+blocksize,icol:icol+blocksize] - pred

            # Blokvis DCT af residualer
            dct_block = dctn(residuals, 2, norm="ortho")
            dct_block[len(dct_block)//2:, :] = 0
            dct_block[:, len(dct_block)//2:] = 0
            DCT[irow-1:irow+blocksize-1, icol-1:icol+blocksize-1] = dct_block

            quant_DCT_block = [quantCoder(row, quant_step) for row in DCT[irow-1:irow+blocksize-1, icol-1:icol+blocksize-1]]
            
            quant_diff_DCT[irow-1:irow+blocksize-1, icol-1:icol+blocksize-1] = np.array(quant_DCT_block)

            # Decode and return decoded
            unquant_DCT_block = [quantDecoder(row, quant_step) for row in quant_DCT_block]

            block = idctn(unquant_DCT_block, 2, norm="ortho")

            decoded[irow-1:irow+blocksize-1, icol-1:icol+blocksize-1] = block + pred

    return quant_diff_DCT, decoded


def inter_predict(current_frame, reference_frame, blocksize, quant_step):
    pass


def encode(current_frame, blocksize, quant_step, immType="I", refererence_frame=None):
    pass

def main():
    blocksize = 16
    quant_step = 4
    img = yuvio.imread("rsc/pa_25fps.yuv", 768, 432, "yuv420p", index=50)
    img_y = img.y.astype(int)

    diff, img_decoded = intra_predict(img_y, blocksize, quant_step) 

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img_y, cmap="grey")
    ax[1].imshow(img_decoded, cmap="grey")

    chars, counts = np.unique(img_y, return_counts = True)
    print(len(np.unique(img_y)))
    print(len(np.unique(diff)))
    freq = counts / sum(counts)
    chars = chars.tolist()
    huffmantable, huffmantree = buildHuffDictMV(chars, freq)
    img_y_encoded = "".join(huffmantable[chars.index(value)][1] for value in img_y.flatten())

    chars, counts = np.unique(diff, return_counts = True)
    freq = counts / sum(counts)
    chars = chars.tolist()
    huffmantable, huffmantree = buildHuffDictMV(chars, freq)
    diff_encoded = "".join(huffmantable[chars.index(value)][1] for value in diff.flatten())
    print(len(np.unique(img_y)))
    print(len(np.unique(diff)))
    print("Original:", 25*len(img_y_encoded)/1000000, "Mbit/S")
    print("Compressed:", 25*len(diff_encoded)/1000000, "Mbit/S" )

    plt.show(block=True)

if __name__ == "__main__":
    main()