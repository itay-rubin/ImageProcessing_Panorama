from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve as sconvolve
import numpy as np
from skimage.color import rgb2gray
import imageio


NORMALIZE_FACTOR = 255
GRAYSCALE = 1
MIN_DIM = 16
COLS = 1
ROWS = 0
SHORTEST_FILTER = 1


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation):
    """
    The function gets an image file path and a representation type,
    normalizes it and converts it if needed.
    :param filename: The image file path - String.
    :param representation: int - 1 for grayscale, 2 for rgb.
    :return: The new image file (matrix)
    """
    image = imageio.imread(filename)

    if image.dtype != np.float64:
        image = image.astype(np.float64)
        image /= NORMALIZE_FACTOR

    if representation == GRAYSCALE:
        image = rgb2gray(image)

    return image


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Building a Gaussian pyramid from a given image with maximum max_levels
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    :return: array with the Gaussian pyramid in a descending resolution order, the filter:a row vector of shape
             (1, filter_size)
    """

    filter_vec = get_filter(filter_size)
    pyr = [im]
    for i in range(max_levels - 1):

        if pyr[i].shape[ROWS] // 2 <= MIN_DIM or pyr[i].shape[COLS] // 2 <= MIN_DIM:
            break
        pyr.append(reduce(pyr[i], filter_vec))

    return pyr, filter_vec


def get_filter(filter_size):
    """
    Creating a binomial filter in a given size
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    :return: a row vector of shape (1, filter_size)
    """
    if filter_size == SHORTEST_FILTER:
        return np.expand_dims(np.ones(SHORTEST_FILTER), ROWS)

    ones = np.ones(2)
    filter_vec = ones
    for i in range(filter_size - 2):
        filter_vec = np.convolve(filter_vec, ones)
    normed_filter = filter_vec / np.sum(filter_vec)
    return np.expand_dims(normed_filter, ROWS)


def blur(im, filter_vec):
    """
    Blurring the image with given filter
    :param im: a grayscale image with double values in [0,1]
    :param filter_vec: a row vector of shape (1, filter_size)
    :return: the blurred grayscale image
    """
    row_blurred = sconvolve(im, filter_vec, mode='nearest')
    col_blurred = sconvolve(np.transpose(row_blurred), filter_vec)
    return np.transpose(col_blurred)


def reduce(im, filter_vec):
    """
    Reducing the image shape by half in each axis after blurring with given filter
    :param im: a grayscale image with double values in [0,1]
    :param filter_vec: a row vector of shape (1, filter_size)
    :return: the reduced grayscale image
    """
    blurred = blur(im, filter_vec)
    return blurred[::2, ::2]

