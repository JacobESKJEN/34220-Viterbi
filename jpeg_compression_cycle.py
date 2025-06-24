import numpy as np
import math as m
from scipy.fftpack import dctn, idctn
from skimage.util import apply_parallel
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.transform import rescale

q_max = 255
qm_y = np.array([[16, 11, 10, 16, 124, 140, 151, 161], [12, 12, 14, 19, 126, 158, 160, 155], 
                [14, 13, 16, 24, 140, 157, 169, 156], [14, 17, 22, 29, 151, 187, 180, 162], 
                [18, 22, 37, 56, 168, 109, 103, 177], [24, 35, 55, 64, 181, 104, 113, 192],
                [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 199]]) 

qm_c = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], 
                [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99], 
                [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]])

def quantizeqm(block, qm, qf):
    outblock = np.round(block/(qm*qf))
    return outblock

def unquantizeqm(block, qm, qf):
    outblock = block*(qm*qf)
    return outblock


def jpeg_compression_cycle(inputim, qf):
    ## Simulates the JPEG compression cycle
    # Set the compression ratio by modifying the compression factor qf (qf is between [0, 100])

    # Quantization Factor
    if qf < 50:
        q_scale = m.floor(5000 / qf )
    else:
        q_scale = 200 - 2*qf

    # RGB to YCbCr
    ycc = rgb2ycbcr(inputim)

    # Downsample chroma
    cb = rescale(ycc[ : , : , 1] +0.5 , (0.5, 0.5), preserve_range=True)
    cr = rescale(ycc[ : , : , 2] +0.5, (0.5, 0.5), preserve_range=True)
    y = ycc[: , : , 0]
    
    # DCT, with scaling before quantization
    yDCTBlk = apply_parallel(dctn, y, (8,8), extra_keywords={'norm': 'ortho'}, compute=True)* q_max
    cbDCTBlk = apply_parallel(dctn, cb, (8,8), extra_keywords={'norm': 'ortho'}, compute=True)* q_max
    crDCTBlk = apply_parallel(dctn, cr, (8,8), extra_keywords={'norm': 'ortho'}, compute=True)* q_max

    # Quantize DCT coefficients
    yDCTBlkQM = apply_parallel(quantizeqm, yDCTBlk, (8,8), extra_arguments=(qm_y, q_scale/100), compute=True)
    cbDCTBlkQM = apply_parallel(quantizeqm, cbDCTBlk, (8,8), extra_arguments=(qm_c, q_scale/100), compute=True)
    crDCTBlkQM = apply_parallel(quantizeqm, crDCTBlk, (8,8), extra_arguments=(qm_c, q_scale/100), compute=True)

    # Dequantize DCT coefficients
    yDCTBlkQMiQM = apply_parallel(unquantizeqm, yDCTBlkQM, (8,8), extra_arguments=(qm_y, q_scale/100), compute=True)
    cbDCTBlkQMiQM = apply_parallel(unquantizeqm, cbDCTBlkQM, (8,8), extra_arguments=(qm_c, q_scale/100), compute=True)
    crDCTBlkQMiQM = apply_parallel(unquantizeqm, crDCTBlkQM, (8,8), extra_arguments=(qm_c, q_scale/100), compute=True)
    
    # Inverse DCT with scaling before
    yDCTBlkQMiQMiDCT = apply_parallel(idctn, yDCTBlkQMiQM/q_max, (8,8), extra_keywords={'norm': 'ortho'}, compute=True)
    cbDCTBlkQMiQMiDCT = apply_parallel(idctn, cbDCTBlkQMiQM/q_max, (8,8), extra_keywords={'norm': 'ortho'}, compute=True)
    crDCTBlkQMiQMiDCT = apply_parallel(idctn, crDCTBlkQMiQM/q_max, (8,8), extra_keywords={'norm': 'ortho'}, compute=True)

    # Upsample chroma
    cbrec = rescale(cbDCTBlkQMiQMiDCT, (2,2), mode='edge', preserve_range=True)
    crrec = rescale(crDCTBlkQMiQMiDCT, (2,2), mode='edge', preserve_range=True)
    yrec = yDCTBlkQMiQMiDCT
    
    # Concatenate the channels
    ycbcrrec = np.dstack((yrec , cbrec -0.5 , crrec -0.5))
    rgb = 255*ycbcr2rgb(ycbcrrec)
    rgb[rgb>255] =255
    rgb[rgb<0] =0
    jpeg_result = rgb.astype('uint8')

    print(jpeg_result)

    return jpeg_result