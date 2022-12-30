import PIL.Image
import numpy as np
import random
from scipy import ndimage
import scipy
import scipy.stats as ss


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """
    Generate an anisotropic Gaussian kernel.  If l1 = l2, will get an isotropic Gaussian kernel.
    :param ksize: e.g., 15, kernel size
    :param theta: [0,  pi], rotation angle range
    :param l1: [0.1,50], scaling of eigenvalues
    :param l2: [0.1,l1], scaling of eigenvalues
    :return: k: kernel

    Code adapted from https://github.com/cszn/BSRGAN
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    """
    Code adapted from https://github.com/cszn/BSRGAN
    :param mean:
    :param cov:
    :param size:
    :return:
    """
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


def fspecial_gaussian(hsize, sigma):
    """
    Code adapted from https://github.com/cszn/BSRGAN
    :param hsize:
    :param sigma:
    :return:
    """
    hsize = [hsize, hsize]
    size = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-size[1], size[1]+1), np.arange(-size[0], size[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


def fspecial_laplacian(alpha):
    """
    Code adapted from https://github.com/cszn/BSRGAN
    :param alpha:
    :return:
    """
    alpha = max([0, min([alpha,1])])
    h1 = alpha/(alpha+1)
    h2 = (1-alpha)/(alpha+1)
    h = [[h1, h2, h1], [h2, -4/(alpha+1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial(filter_type, *args, **kwargs):
    """
    Code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    """
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)


def generate_kernel(sigma=2.4, kernel_size=21, l1=2, l2=5, theta=np.pi, gauss_type='isotropic'):

    if gauss_type == 'random':
        if random.random() < 0.5:
            gauss_type = 'anisotropic'
        else:
            gauss_type = 'isotropic'

    if gauss_type == 'anisotropic':
        k = anisotropic_Gaussian(ksize=kernel_size, theta=theta, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', kernel_size, sigma)
    return k


def bsrgan_blur(img, sigma=2.4, kernel_size=21, l1=2, l2=5, theta=np.pi, gauss_type='isotropic'):  # TODO: confirm license is ok to use
    """
    Code adapted from https://github.com/cszn/BSRGAN
    :param img: Input image
    :param sigma: Isotropic gaussian kernel width
    :param kernel_size: Blur kernel size (square)
    :param l1: anisotropic gaussian axis scaling
    :param l2: anisotropic gaussian axis scaling
    :param theta: # Anisotropic rotation (in radians)
    :param gauss_type: Type of Gaussian to use ('isotropic', 'anisotropic', 'random')
    :return: blurred image
    """

    k = generate_kernel(sigma=sigma, kernel_size=kernel_size, l1=l1, l2=l2, theta=theta, gauss_type=gauss_type)

    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img, k


# TODO: - find out which types of kernels can be used - consider Gaussian, anisotropic etc etc

if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt

    img_in = Image.open('../../../../Data/set5_samples/hr/baby.png')
    img = np.array(img_in)

    blur_im, kernel = bsrgan_blur(img, gauss_type='anisotropic')

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_in)
    ax[1].imshow(blur_im)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()

    plt.figure()
    plt.imshow(kernel)
    plt.show()
