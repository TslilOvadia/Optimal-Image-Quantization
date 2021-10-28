### Ex-1.
### Submitted by Tzlil Ovadia, ID: 311317689

import matplotlib.pyplot as plt
import numpy as np
import skimage.color


YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],[0.596, -0.275, -0.321],[0.212, -0.523, 0.311]])
RGB_MATRIX = np.array([[1, 0.956, 0,621],[1, -0.272, -0.647],[1, -1.106, 1.703]])
GRAY_SCALE = 1
RGB = 2
GRAY_SCALE_LEVELS = 256


# x = np.hstack([np.repeat(np.arange(0,50,2),10)[None, :], np.array([255]*6)[None, :]])
# grad = np.tile(x, (256,1))

# 3.2 - Read Image
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
    if representation != RGB and representation != GRAY_SCALE:
        return "Invalid Input. You may use representation <- {1, 2}"
    tempImage  = plt.imread(filename)[:,:,:3]
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

# 3.3 - Display Image
def imdisplay(filename, representation):
    """
    filename and representation are the same as those defined in read_image’s interface.
    :param filename:
    :param representation:
    """
    if representation != GRAY_SCALE and representation != RGB:
        return "Invalid Input. You may use representation <- {1, 2}"
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
    r,g,b = imRGB[:,:,0], imRGB[:,:,1], imRGB[:,:,2]
    y = YIQ_MATRIX[0][0]*r + YIQ_MATRIX[0][1]*g + YIQ_MATRIX[0][2]*b
    i = YIQ_MATRIX[1][0]*r + YIQ_MATRIX[1][1]*g + YIQ_MATRIX[1][2]*b
    q = YIQ_MATRIX[2][0]*r + YIQ_MATRIX[2][1]*g + YIQ_MATRIX[2][2]*b
    result = imRGB
    result[:,:,0] = y
    result[:,:,1] = i
    result[:, :, 2] = q
    return result
# 3.4.2 - Transforming an YIQ image to RGB color space
def yiq2rgb(imYIQ):
    """
    Transforming an YIQ image to RGB color space
    :param imYIQ:
    :return:
    """
    y,i,q = imYIQ[:,:,0], imYIQ[:,:,1], imYIQ[:,:,2]
    r = RGB_MATRIX[0][0]*y + RGB_MATRIX[0][1]*i + RGB_MATRIX[0][2]*q
    g = RGB_MATRIX[1][0]*y + RGB_MATRIX[1][1]*i + RGB_MATRIX[1][2]*q
    b = RGB_MATRIX[2][0]*y + RGB_MATRIX[2][1]*i + RGB_MATRIX[2][2]*q
    result = imYIQ
    result[:,:,0] = r
    result[:,:,1] = g
    result[:, :, 2] = b
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



def checkImageFormat(image):
    """

    :param image: an Image
    :return:
    """
    if len(image.shape) == 2:
        return GRAY_SCALE
    elif len(image.shape) == 3:
        return RGB
    else:
        return 0


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

    """ 
    Important Things To Complete:
    1. Ask if it is possible to make use of skimage.color.rgb2yiq and yiq2rgb
    2. Figure out how to perform a linear stretch
    3. Add some end-cases where the algorithm could fail e.g where we have only 2-color image.
    4. Remove unnecessary junk from code
    5. Make sure that the values returned are by the requested format
    """

    # Step No. 0 - Check if the given image is an RGB/GrayScale Format:
    swappedToYIQ = False # Used to help us know if yiq2rgb is needed
    color_im = None
    N = getNumOfPixel(im_orig)
    format = checkImageFormat(im_orig)
    if format == RGB:
        color_im = rgb2yiq(im_orig)
        # Take The Y Comp.
        im_orig = color_im[:,:,0]

        swappedToYIQ = True

    # Step No. 1 - Compute the image histogram:
    hist_orig,bins = np.histogram(im_orig, 256)
    # Step No. 2 - Compute the cumulative histogram:
    hist_cdf = np.array(hist_orig.cumsum(), dtype=np.float)
    # Step No. 3 - Normalize the cumulative histogram:
    hist_cdf /= N
    # Step No 4. - Multiply the normalized histogram by the maximal gray level value (Z-1):
    hist_cdf *= 255
    # Step No 5. - Verify that the minimal value is 0 and that the maximal is Z-1, otherwise
    # stretch the result linearly in the range [0,Z-1]:

    flat = np.array(im_orig.flatten())
    flat = np.floor(flat*255)
    # LUT:
    lookUpTable = np.floor((hist_cdf - hist_cdf.min())/(hist_cdf[255]-hist_cdf.min())*255)
    flat_im_eq = lookUpTable[np.array(flat, dtype=int)]
    #
    im_eq = np.reshape(np.asarray(flat_im_eq), im_orig.shape)
    if swappedToYIQ:
        color_im[:,:,0] = im_eq/255

        im_eq = yiq2rgb(color_im).astype(np.float64)
    hist_eq, bins_eq = np.histogram(flat_im_eq, 256)

    return im_eq, hist_orig, hist_eq

def initQuants(hist_seg):
    quants = []
    for i in range(len(hist_seg)-1):
        quants.append(int((hist_seg[i]+(hist_seg[i+1]-hist_seg[i])/2)))
    return quants

def updateSegmentIndex(q1, q2):
    """
    Helper function used in qunatize() func in order to update the indices of z array
    :param q1:
    :param q2:
    :return: updated value of z_index inside z array.
    """
    return (q1+q2)/2

def updateQuantIndex(start_idx, stop_idx, histogram):
    """
    Helper function used in qunatize() func in order to update the quants indices.
    :param seg_i: Is an range which represents a segment of 256 array.
    :param histogram:
    :return: updated index of the current quant value which minimize the error
    """
    seg_i_arr = np.array(range(int(start_idx),int(stop_idx+1))) ##
    hist_seg_i = histogram[range(int(start_idx),int(stop_idx+1))] ##
    enumrtator = sum(list(map(lambda z, h_z: z * h_z,seg_i_arr , hist_seg_i)))
    denomenator = sum(hist_seg_i)

    if denomenator == 0:
        return 0


    return enumrtator/denomenator



def check_if_updated(before, after):
    """
    Given 2 array versions, we check if the array was updated
    :param before: array before iterations
    :param after: array after iterations
    :return: Boolean value - True if any updates were made, False otherwise.
    """
    if before == after:
        return False
    return True

# 3.6 Optimal image quantization
def quantize (im_orig, n_quant, n_iter):
    """
    :param im_orig: is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1])
    :param n_quant: is the number of intensities your output im_quant image should have.
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: [im_quant, error] where:
             im_quant: is the quantized output image. (float64 image with values in [0, 1]).
             error: is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
    """
    # Setting the relevant variables for the algorithm:
    error = []
    histogram,bins = np.histogram(im_orig, bins = 256)
    hist_cdf = np.array(histogram.cumsum(), dtype=np.float64) #make more elegant
    N = getNumOfPixel(im_orig)
    delta = N/n_quant
    z = [-1]
    # Initialize the segments array we want to start with:
    for q in range(1,n_quant):
        z.append(np.where(hist_cdf >= q * delta)[0][0])
    z.append(255)
    # Initialize the values of the initial quants which we will update
    quants = initQuants(z)
    # Iterate through the quants and z items, and update the values to reduce the error
    for iteration in range(n_iter):
        # quants_updated = True
        # z_updated = True
        # quants_prev = quants
        # z_prev = z
        # Computing q - the values to which each of the segments’ intensities will map.
        #               q is also a one dimensional array, containing n_quant elements:

        for q in range(len(quants)):
            quants[q] = updateQuantIndex(z[q],z[q+1], histogram)
        # quants_updated = check_if_updated(quants_prev,quants)
        # Computing z - the borders which divide the histograms into segments.
        #           z is an array with shape (n_quant+1,). The first and last elements are 0 and 255 respectively:
        for z_i in range(1, len(z)-2):
            z[z_i] = updateSegmentIndex(quants[z_i-1], quants[z_i])
        # z_updated = check_if_updated(z_prev,z)

        #Loop for the errors calculations:
        error_i = 0
        for i in range(n_quant):
            g = np.arange(int(z[i]), int(z[i+1]))
            error_i  += sum((quants[i] - g)**2 * histogram[g])

        error.append(error_i)
    plt.plot(error)
    plt.show()
    for z_i in range(len(z)-1):
        im_orig[ (z[z_i] < im_orig) & ( im_orig <= z[z_i+1])] = int(quants[z_i])
    im_quant = im_orig

    return [im_quant, error]
