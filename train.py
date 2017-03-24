# import all necessary libraries
import glob
import os
import time
from extractor import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
import pickle

# set the parameter constanst so it is easier to train
CSPACE = 'YCrCb'  # RGB, HSV, LUV, HLS, YUV, YCrCb
SPATIAL_BIN_SIZE = (16, 16)
HIST_NBINS = 16
HOG_ORIENT = 9
HOG_PIX_PER_CELL = 8
HOG_CELL_PER_BLOCK = 2
HOG_CHANNEL = (0, 1, 2)
INCL_SPATIAL = True
INCL_HIST = True
INCL_HOG = True

# set the model to be used
MODEL = 'SVC'  # SVC, DT, NB

# set the file names and data directories
VEHICLES_DIR = './data/vehicles'
NON_VEHICLES_DIR = './data/non-vehicles'
PICKLE_FILE = 'svc.p'


def get_features_filename():
    '''
    Method to use the training constants to construct a filename to store the features and parameters and return it
    :return: the filename for the features pickle file
    '''
    return 'features_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.p'.format(CSPACE, SPATIAL_BIN_SIZE[0], HIST_NBINS, HOG_ORIENT,
                                                             HOG_PIX_PER_CELL,
                                                             HOG_CELL_PER_BLOCK, HOG_CHANNEL, INCL_SPATIAL,
                                                             INCL_HIST,
                                                             INCL_HOG)


def get_model_filename():
    '''
    Method to use the training constants to construct a filename to store the model and return it
    :return: the filename for the model pickle file
    '''
    return 'model_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.p'.format(MODEL, SPATIAL_BIN_SIZE[0], HIST_NBINS, HOG_ORIENT,
                                                          HOG_PIX_PER_CELL,
                                                          HOG_CELL_PER_BLOCK, HOG_CHANNEL, INCL_SPATIAL, INCL_HIST,
                                                          INCL_HOG)


def get_model():
    '''
    Method to get the saved model from the saved file
    :return: the saved classifier
    '''
    model_file = get_model_filename()
    if os.path.exists(model_file):
        model_data = pickle.load(open(model_file, "rb"))
        return model_data['model']
    else:
        print('Could not find model file')
        raise


def get_train_parameters():
    '''
    Method to get the training parameters from the saved file
    :return: the saved params as a dictionary
    '''
    features_file = get_features_filename()

    if os.path.exists(features_file):

        # set all features in a dictionary
        features = pickle.load(open(features_file, "rb"))
        features_data = {}
        features_data['cspace'] = features['cspace']
        features_data['hog_orient'] = features['hog_orient']
        features_data['hog_pix_per_cell'] = features['hog_pix_per_cell']
        features_data['hog_cell_per_block'] = features['hog_cell_per_block']
        features_data['spatial_size'] = features['spatial_size']
        features_data['hist_bins'] = features['hist_bins']
        features_data['hog_channel'] = features['hog_channel']
        features_data['incl_spat'] = features['incl_spat']
        features_data['incl_hog'] = features['incl_hog']
        features_data['incl_hist'] = features['incl_hist']

        # return the features dict
        return features_data, features['scaler']
    else:
        print('Could not find features file')
        raise


def load_features(sample_size=False):
    '''
    Method to load feature vectors for the training data
    :param sample_size: parameter to limit the sample size due to space constraints
    :return: the feature vector of the cars and non-cars data
    '''

    # read in cars and notcars images from the corresponding folders
    cars = []
    notcars = []
    cars_images = glob.glob(VEHICLES_DIR)
    for folder in cars_images:
        cars += glob.glob('{}/*/*.png'.format(folder))

    not_cars_images = glob.glob(NON_VEHICLES_DIR)
    for folder in not_cars_images:
        notcars += glob.glob('{}/*/*.png'.format(folder))

    # restrict the count of data if sample size is restricted
    if sample_size:
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]
    else:
        sample_size = len(cars) + len(notcars)

    # get the features filename
    features_file = get_features_filename()

    # if features exist  check if it can be reused
    if os.path.exists(features_file):
        char = input('A features pickle already exists, do you want to reuse (y/n):')
        if char == 'Y' or char == 'y':
            features = pickle.load(open(features_file, "rb"))
            return features['features_train'], features['labels_train'], features['features_test'], \
                   features['labels_test'], features['scaler']

    car_features = []
    non_car_features = []

    print('total car images count is', len(cars))
    print('total non car images count is', len(notcars))

    # extract the features for each image
    t1 = time.time()
    for file in cars:
        img = load_image(file)
        car_features.append(get_feature(img, cspace=CSPACE,
                                        spatial_size=SPATIAL_BIN_SIZE, hist_bins=HIST_NBINS,
                                        orient=HOG_ORIENT, pix_per_cell=HOG_PIX_PER_CELL,
                                        cell_per_block=HOG_CELL_PER_BLOCK,
                                        hog_channels=HOG_CHANNEL, spatial_feat=INCL_SPATIAL,
                                        hist_feat=INCL_HIST, hog_feat=INCL_HOG))
    car_features = shuffle(car_features)

    for file in notcars:
        img = load_image(file)
        non_car_features.append(get_feature(img, cspace=CSPACE,
                                            spatial_size=SPATIAL_BIN_SIZE, hist_bins=HIST_NBINS,
                                            orient=HOG_ORIENT, pix_per_cell=HOG_PIX_PER_CELL,
                                            cell_per_block=HOG_CELL_PER_BLOCK,
                                            hog_channels=HOG_CHANNEL, spatial_feat=INCL_SPATIAL,
                                            hist_feat=INCL_HIST, hog_feat=INCL_HOG))
    t2 = time.time()
    print(round(t2 - t1, 2), 'seconds to extract all features...')

    # stack the features together
    X = np.vstack((car_features, non_car_features)).astype(np.float64)

    # scale the features
    X_scaler = StandardScaler().fit(X)
    X_scaled = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=rand_state)

    print('Parameters: ', CSPACE, ' Color Space', HOG_ORIENT, ' HOG Orientations', HOG_PIX_PER_CELL,
          ' HOG pixels per cell', HOG_CELL_PER_BLOCK, ' HOG cells per block', SPATIAL_BIN_SIZE, ' Spatial Size',
          HIST_NBINS, ' Histogram Bins')
    print('Feature vector length is:', len(X_train[0]))

    # save the parameters and features
    try:
        with open(features_file, 'wb') as pfile:
            pickle.dump(
                {
                    'features_train': X_train,
                    'labels_train': y_train,
                    'features_test': X_test,
                    'labels_test': y_test,
                    'scaler': X_scaler,
                    'cspace': CSPACE,
                    'hog_orient': HOG_ORIENT,
                    'hog_pix_per_cell': HOG_PIX_PER_CELL,
                    'hog_cell_per_block': HOG_CELL_PER_BLOCK,
                    'spatial_size': SPATIAL_BIN_SIZE,
                    'hist_bins': HIST_NBINS,
                    'hog_channel': HOG_CHANNEL,
                    'incl_spat': INCL_SPATIAL,
                    'incl_hog': INCL_HOG,
                    'incl_hist': INCL_HIST
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', features_file, ':', e)
        raise

    # return the features and labels for training and validation
    return X_train, y_train, X_test, y_test, X_scaler


def train_classifier(sample_size=False):
    '''
    Method to train the classifier from the feature vector
    :param sample_size: restrict sample size during space constraints
    :return: the trained classifier
    '''

    # get the feature vector
    X_train, y_train, X_test, y_test, scaler = load_features(sample_size)

    # get the model filename used to save
    model_file = get_model_filename()
    model = None

    # if model exist check if it can be reused
    if os.path.exists(model_file):
        char = input('A model pickle already exists, do you want to reuse (y/n):')
        if char == 'Y' or char == 'y':
            model_data = pickle.load(open(model_file, "rb"))
            model = model_data['model']

    # create a new model
    if not model:
        if MODEL == 'SVC':
            model = LinearSVC()
        elif MODEL == 'DT':
            model = DecisionTreeClassifier()
        elif MODEL == 'NB':
            model = GaussianNB()

        # Check the training time for the SVC
        t = time.time()
        print('Training model ...')

        # train the classifier
        model.fit(X_train, y_train)

        t2 = time.time()
        print(round(t2 - t, 2), 'seconds to train model...')

        # save the model for later use
        try:
            with open(model_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'model': model,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', model_file, ':', e)
            raise

    # Check the score of the model
    print('Test Accuracy of the model = ', round(model.score(X_test, y_test), 4))

    # return the model and the train parameters
    return model, sample_size, get_train_parameters()
