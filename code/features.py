import numpy as np
import cv2

from skimage.feature import hog

from code.io import glob_file_paths

# Helper

def _map_color_transform(c_space = None):

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

    return conversion


class FeatureExtractor:
    def __init__(self, config):
        
        self.conversion = _map_color_transform(config["color_space"])
        self.orientations = config["orientations"]
        self.pix_per_cell = config["pix_per_cell"]
        self.cell_per_block = config["cell_per_block"]
        self.spatial_size = config["spatial_size"]
        self.histogram_bins = config["histogram_bins"]

        self.n_channels = 3

    # HOG features

    def _hog_worker(self, image):

        features = hog(image, orientations = self.orientations, 
                              pixels_per_cell = (self.pix_per_cell, self.pix_per_cell),
                              cells_per_block = (self.cell_per_block, self.cell_per_block), 
                              transform_sqrt = True,
                              visualize = False,
                              feature_vector = True,
                              block_norm= 'L2-Hys')

        hog_image = None

        return features, hog_image


    def _extract_hog_features(self, image, cpu_pool = None):

        if cpu_pool is None:
            features = list(map(lambda ch:self._hog_worker(image[:,:,ch])[0], range(self.n_channels)))
            features = np.ravel(features)
        else:
            tmp = cpu_pool.map(self._hog_worker, [image[:,:,0], image[:,:,1], image[:,:,2]])
            features = np.concatenate((tmp[0][0], tmp[1][0], tmp[2][0]))

        return features

    # Spatial features

    def _extract_spatial_features(self, image):

        # ravel() flattens the image to a vector
        features = cv2.resize(image, (self.spatial_size, self.spatial_size)).ravel()

        return features

    # Histogram features

    def _hist_worker(self, channel):
        
        h = np.histogram(channel, bins = self.histogram_bins, range = (0, 256))[0]

        return h

    def _extract_histogram_features(self, image, cpu_pool = None):

        # Loop over each color channel in image and create a histogram for each
        if cpu_pool is None:
            histograms = list(map(lambda channel:self._hist_worker(image[:, :, channel]), range(self.n_channels)))
        else:
            histograms = cpu_pool.map(self._hist_worker, [image[:,:,0], image[:,:,1], image[:,:,2]])


        features = np.concatenate(histograms)
        

        return features

    # Externals


    def extract_image_features(self, image, cpu_pool = None):

        #image = _apply_color_transform(image, self.color_space)
        image = cv2.cvtColor(image, self.conversion)

        f_spatial = self._extract_spatial_features(image)

        f_hist = self._extract_histogram_features(image, cpu_pool = cpu_pool)

        f_hog = self._extract_hog_features(image, cpu_pool = cpu_pool)

        features = np.concatenate((f_spatial, f_hist, f_hog))

        return features


    def extract_features(self, folder_path, cpu_pool = None):

        # Glob the file paths instead of images to save some memory
        file_paths = glob_file_paths(folder_path)

        n_images = len(file_paths)

        image = cv2.imread(file_paths[0])

        n_channels = image.shape[2]

        # Pre-allocate a numpy array

        n_features_per_image = len(self.extract_image_features(image, cpu_pool = cpu_pool))
        
        features = np.zeros((n_images, n_features_per_image), dtype = np.float)

        for i in range(n_images):

            image = cv2.imread(file_paths[i])

            features[i] = self.extract_image_features(image, cpu_pool = cpu_pool)

            if i % 100 == 0:
                print('INFO:FeatureExtractor.extract_features():' + folder_path +': Image ' + str(i) + ' of ' + str(n_images))


        return features