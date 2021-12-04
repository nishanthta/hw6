# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import cv2, os

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    ht, width = res[0], res[1]
    x, y = np.meshgrid(np.arange(ht), np.arange(width))
    c = (ht / 2., width / 2.)
    real_x = (x - c[0]) * pxSize + center[0]
    real_y = (y - c[1]) * pxSize + center[1]
    real_z_sq = rad ** 2 - real_x ** 2 - real_y ** 2
    neg = real_z_sq < 0
    real_z_sq[neg] = 0.
    real_z = np.sqrt(real_z_sq)

    n = np.stack((real_x, real_y, real_z), axis=2)
    n = n.reshape((ht*width, -1))
    n = (n.T / np.linalg.norm(n, axis=1).T).T
    image = np.dot(n, light).reshape((width, ht))
    return image



def loadData(path = "hw6/data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None
    for i in range(7):
        img = cv2.imread(os.path.join(path, 'input_{}.tif'.format(i + 1)))
        if img is None:
            print('Empty')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ht, width = gray.shape[0], gray.shape[1]
        gray = gray.reshape((1,ht*width))
        if i == 0:
            I = np.zeros((7, ht*width))
        I[i,:] = gray
    L = np.load(os.path.join(path, 'sources.npy')).T
    s = (ht, width)
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    temp = np.dot(L,L.T)
    temp = np.linalg.inv(temp).dot(L)
    B = temp.dot(I)
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    # albedos = None
    # normals = None
    albedos = np.linalg.norm(B, axis=0)
    normals = B / (albedos + 1e-6)
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None

    ht, width = s[0], s[1]

    albedoIm = np.reshape((albedos / np.max(albedos)), s)
    normals = (normals + 1.) / 2.
    normalIm = np.reshape((normals).T, (ht, width, 3))
    
    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    dx = normals[0, :]/(-1.*normals[2, :] + 1e-6)
    dy = normals[1, :]/(-1.*normals[2, :] + 1e-6)

    dx, dy = np.reshape(dx, s), np.reshape(dy, s)

    return integrateFrankot(dx, dy)


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    pass


if __name__ == '__main__':

    # Put your main code here
    center = np.asarray([0., 0., 0.])
    rad = 7.5
    lights = np.asarray([[1, 1, 1]/np.sqrt(3), [1, -1, 1] /
                         np.sqrt(3), [-1, -1, 1]/np.sqrt(3)])
    pxSize = 7e-3
    res = np.asarray([3840, 2160])
    for i in range(len(lights)):
        image = renderNDotLSphere(center, rad, lights[i], pxSize, res)
        # plt.imshow(image)
        # plt.show()
        # plt.clf()

    I, L, s = loadData()
    u, v, vh = np.linalg.svd(I, full_matrices=False)

    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    plt.imshow(albedoIm, cmap='gray')
    plt.show()
    pass
