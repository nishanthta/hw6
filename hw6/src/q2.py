# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
import matplotlib.pyplot as plt

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    B = None
    L = None
    u, s, vt = np.linalg.svd(I, full_matrices=False)
    k = 3 #set the required rank
    s[k:] = 0
    # now, M = u s vt is the reconstruction for M with rank 3

    return vt[:k,], u[:k,]


if __name__ == "__main__":

    # Put your main code here
    I, originalL, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    albedos, normals = estimateAlbedosNormals(B)
    # normals = enforceIntegrability(normals, s)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))

    ax[0].imshow(albedoIm, cmap='gray')
    ax[1].imshow(normalIm, cmap='gray')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()
    plt.clf()

    surface = estimateShape(normals, s)
    # plotSurface(surface)
