import numpy as np
import cv2

from skimage.feature import hog

from code.io import glob_file_paths

# Internals

def _apply_color_transform(image, c_space = None):

    if c_space is None:
        return image

    if c_space == 'HSV':
        conversion = cv2.COLOR_BGR2HSV
    elif c_space == 'LUV':
        conversion = cv2.COLOR_RGB2LUV
    elif c_space == 'HLS':
        conversion = cv2.COLOR_BGR2HLS
    elif c_space == 'YUV':
        conversion = cv2.COLOR_BGR2YUV
    elif c_space == 'YCrCb':
        conversion = cv2.COLOR_BGR2YCrCb

    image = cv2.cvtColor(image, conversion)

    return image


def _hog_features(image, orientations, pix_per_cell, cell_per_block, visualize = False, feature_vector = True):

    tmp = hog(image, orientations = orientations, 
                          pixels_per_cell = (pix_per_cell, pix_per_cell),
                          cells_per_block = (cell_per_block, cell_per_block), 
                          transform_sqrt = True,
                          visualize = visualize,
                          feature_vector = feature_vector,
                          block_norm= 'L2-Hys')

    if visualize:
        features = None
        hog_image = tmp[1]
    else:
        features = tmp
        hog_image = None

    return features, hog_image



def _extract_hog_features(image, orientations, pix_per_cell, cell_per_block, n_channels, feature_vector = True):

    features = list(map(lambda ch:_hog_features(image[:,:,ch], orientations, pix_per_cell, cell_per_block, feature_vector = feature_vector)[0], range(n_channels)))

    features = np.ravel(features)

    return features

def _extract_spatial_features(image, spatial_size):

    # ravel() flattens the image to a vector
    features = cv2.resize(image, (spatial_size, spatial_size)).ravel()

    return features

def _extract_histogram_features(image, n_bins, n_channels):

    def _hist(channel, n_bins):
        
        h = np.histogram(channel, bins = n_bins, range = (0, 256))[0]

        return h

    # Loop over each color channel in image and create a histogram for each
    histograms = list(map(lambda channel:_hist(image[:, :, channel], n_bins), range(n_channels)))

    # Create a single histogram vector
    features = np.concatenate(histograms)

    return features

# Externals


def extract_hog_image(image, orientations, pix_per_cell, cell_per_block):

    hog_image = _hog_features(image, orientations, pix_per_cell, cell_per_block, visualize = True)[1]

    return hog_image


def extract_image_features(image, config, n_channels, feature_vector = True):

    image = _apply_color_transform(image, config["color_space"])

    f_spatial = _extract_spatial_features(image, config["spatial_size"])

    f_hist = _extract_histogram_features(image, config["histogram_bins"], n_channels)

    f_hog = _extract_hog_features(image, config["orientations"], config["pix_per_cell"], config["cell_per_block"], n_channels, feature_vector = feature_vector)

    features = np.concatenate((f_spatial, f_hist, f_hog))

    return features


def extract_features(folder_path, config):

    # Glob the file paths instead of images to save some memory
    file_paths = glob_file_paths(folder_path)

    n_images = len(file_paths)

    image = cv2.imread(file_paths[0])

    n_channels = image.shape[2]

    # Pre-allocate a numpy array

    n_features_per_image = len(extract_image_features(image, config, n_channels))
    
    features = np.zeros((n_images, n_features_per_image), dtype = np.float)

    for i in range(n_images):

        image = cv2.imread(file_paths[i])

        features[i] = extract_image_features(image, config, n_channels)

        if i % 100 == 0:
            print('INFO:extract_image_features():' + folder_path +': Image ' + str(i) + ' of ' + str(n_images))


    return features