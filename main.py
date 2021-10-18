# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np
import skimage.color

YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],[0.596, -0.275, -0.321],[0.212, -0.523, 0.311]])
RGB_MATRIX = np.array([[1, 0.956, 0,621],[1, -0.272, -0.647],[1, -1.106, 1.703]])
GRAY_SCALE = 1
RGB = 2
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
x = np.hstack([np.repeat(np.arange(0,50,2),10)[None, :], np.array([255]*6)[None, :]])
grad = np.tile(x, (256,1))

# filename - the filename of an image on disk (could be grayscale or RGB).
# representation - representation code, either 1 or 2 defining whether the output should be a:
# grayscale image (1)
# or an RGB image (2).
# NOTE: If the input image is grayscale, we won’t call it with represen- tation = 2.
def read_image(filename, representation):
    """

    :param filename:
    :param representation:
    :return:
    """
    tempImage  = plt.imread(filename)
    resultImage = tempImage
    if representation == GRAY_SCALE:
        resultImage = skimage.color.rgb2gray(tempImage)
    elif representation == RGB:
        resultImage = tempImage

    return resultImage



def imdisplay(filename, representation):
    """
    filename and representation are the same as those defined in read_image’s interface.

    :param filename:
    :param representation:
    :return:
    """
    imageToShow = read_image(filename, representation)
    plt.figure()
    plt.imshow(imageToShow, cmap = "gray")
    plt.show()


def rgb2yiq(imRGB):
    """

    :param imRGB:  An RGB image
    :return:
    """
    result = np.dot(imRGB, YIQ_MATRIX.T)
    return result


def yiq2rgb(imYIQ):
    """

    :param imYIQ:
    :return:
    """
    result = np.dot(imYIQ, RGB_MATRIX.T)
    return result


def histogram_equalize(im_orig):
    """
    Given an image, this function will perform an histogram equalization.
    NOTE: If an RGB image is given, the following equalization procedure should only operate on the Y channel of
    :param im_orig: is the input grayscale or RGB float64 image with values in [0, 1].
    :return: a list [im_eq, hist_orig, hist_eq] where:
            - im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1]
            - hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
            - hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """

    resultArray = []

    pass

if __name__ == '__main__':
    imdisplay("/Users/tzlilovadia/Desktop/pic.png", 1)
