# import all necessary libraries
import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog


def load_image(file):
    '''
    Method to load an image using matplotlib.image
    :param file: the path of the file
    :return: numpy array of image
    '''
    return mpimg.imread(file)


def convert_color(img, cspace):
    '''
    Function to convert the color space of the image
    :param img: The image to convert
    :param cspace: the color space to convert to
    :return: the converted image
    '''

    # convert the image based on the color space specified
    if cspace == 'RGB':
        cspace_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif cspace == 'BGR':
        cspace_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif cspace == 'YUV':
        cspace_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif cspace == 'HSV':
        cspace_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        cspace_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        cspace_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        cspace_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
        cspace_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        cspace_image = np.copy(img)

    return cspace_image


def bin_spatial(img, size=(32, 32)):
    '''
    Function to get the spatial feature vector
    :param img: The image to extract features from
    :param size: the new spatial size
    :return: the spatial feature vector
    '''
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()

    # stack all color channels and return
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    Function to get a histogram of colors.
    :param img: the image to extract the feature from
    :param nbins: number of bins in the histogram
    :param bins_range: the range of numbers in the image
    :return: the extracted feature
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    '''
    Function to get the Histogram of Gradients for an image
    :param img: the image to extract feature from
    :param orient: the HOG orientations
    :param pix_per_cell: the number of pixels per cell
    :param cell_per_block: the number of cells per block
    :param vis: boolean to indicate the HOG of the image should be returned
    :param feature_vec: boolean to indicate the features must be vectorized
    :return: the HOG features and the image
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features
