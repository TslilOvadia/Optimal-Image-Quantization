### Ex-1.
### Submitted by Tzlil Ovadia, ID: 311317689


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


def read_image(filename, representation):
    """
    filename - the filename of an image on disk (could be grayscale or RGB).
    representation - representation code, either 1 or 2 defining whether the output should be a:
    grayscale image (1)
    or an RGB image (2).
    NOTE: If the input image is grayscale, we won’t call it with represen- tation = 2.
    :param filename: String - the address of the image we want to read
    :param representation: Int - as described above
    :return: an image in the correct representation
    """
    tempImage  = plt.imread(filename)
    resultImage = tempImage
    if representation == GRAY_SCALE:
        resultImage = skimage.color.rgb2gray(tempImage)
    elif representation == RGB:
        resultImage = tempImage

    return resultImage


def TEST_imdisplay(imageToShow):
    """
    For self-testing purpose.
    :param imageToShow: an image Object recieved from read_image() func.
    """
    plt.figure()
    plt.imshow(imageToShow, cmap = "gray")
    plt.show()

def imdisplay(filename, representation):
    """
    filename and representation are the same as those defined in read_image’s interface.
    :param filename:
    :param representation:
    """
    imageToShow = read_image(filename, representation)
    plt.figure()
    plt.imshow(imageToShow, cmap = "gray")
    plt.show()


# 3.4.1 - Transforming an RGB image to YIQ color space
def rgb2yiq(imRGB):
    """
    Transforming an RGB image to YIQ color space
    :param imRGB:  An RGB image
    :return:
    """
    result = np.dot(imRGB, YIQ_MATRIX.T)
    return result


# 3.4.2 - Transforming an YIQ image to RGB color space
def yiq2rgb(imYIQ):
    """
    Transforming an YIQ image to RGB color space
    :param imYIQ:
    :return:
    """
    result = np.dot(imYIQ, RGB_MATRIX.T)
    return result


def getNumOfPixel(image):
    """
    Helper function to get number of pixels in image (assuming that image is valid)
    :param image: an image file read with read_image() function
    :return: Integer, ∑(pixels)
    """
    return len(image)*len(image[0])

def checkIfNormalizedValid(histogram):
    """

    :param histogram:
    :return:
    """
    return histogram.max() == 255 and histogram.min() == 0



# 3.5 - Histogram equalization

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

    # Step No. 1 - Compute the image histogram:
    hist_orig,bins = np.histogram(im_orig.flatten(), 256, [0, 1])
    # Step No. 2 - Compute the cumulative histogram:
    hist_cdf = np.array(hist_orig.cumsum(), dtype=np.float)
    # Step No. 3 - Normalize the cumulative histogram:
    N = getNumOfPixel(im_orig)
    hist_cdf /= N
    # Step No 4. - Multiply the normalized histogram by the maximal gray level value (Z-1):
    hist_cdf *= 255
    # Step No 5. - Verify that the minimal value is 0 and that the maximal is Z-1, otherwise
    # stretch the result linearly in the range [0,Z-1]:
    # print(f"max GC: {hist_cdf.max()} \n min GC: {hist_cdf.min()}")
    if not checkIfNormalizedValid(hist_cdf):
        #Linear Stretch here
        pass
    im_orige = np.array(im_orig.flatten())
    im_orige = np.floor(im_orige*255)
    # LUT:
    lookUpTable = np.floor((hist_cdf - hist_cdf.min())/(hist_cdf[255]-hist_cdf.min())*255)
    flat_im_eq = lookUpTable[np.array(im_orige, dtype=int)]
    #
    im_eq = np.reshape(np.asarray(flat_im_eq), im_orig.shape)
    # print(T_k)
    TEST_imdisplay(im_eq)
    hist_eq, bins_eq = np.histogram(im_eq.flatten(), 256, [0, 1])

    return hist_orig, im_eq, hist_eq




if __name__ == '__main__':
    imdisplay("/Users/tzlilovadia/Desktop/test.png", 1)
    histogram_equalize(read_image("/Users/tzlilovadia/Desktop/test.png", 1))