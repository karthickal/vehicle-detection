# import all necessary libraries
from utils import *

def get_feature(img, cspace='RGB', spatial_size=(64, 64), hist_bins=32, orient=9,
                pix_per_cell=8, cell_per_block=2, hog_channels=0,
                spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    Function to extract the desired features from an image
    :param img: the image to extract features from
    :param cspace: the colorspace to convert the image to
    :param spatial_size: the spatial size for the feature
    :param hist_bins: the number of bins for the histogram feature
    :param orient: the HOG orientation
    :param pix_per_cell: the number of pixels per cell
    :param cell_per_block: the number of cells per block
    :param hog_channels: the channels of the image from which HOG has to be extracted
    :param spatial_feat: boolean to indicate if spatial features should be extracted
    :param hist_feat: boolean to indicate if color histogram features should be extracted
    :param hog_feat: boolean to indicate if HOG features should be extracted
    :return: the final feature vector
    '''

    # convert the image
    image = convert_color(img, cspace)

    # get the spatial features
    if spatial_feat == True:
        spatial_features = bin_spatial(image, size=spatial_size)

    # get the histogram of colors
    if hist_feat == True:
        hist_features = color_hist(image, nbins=hist_bins)

    # get the histogram of features
    if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = []

        for channel in hog_channels:
            hog_features.append(get_hog_features(image[:, :, channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)

    return np.concatenate((spatial_features, hist_features, hog_features))
