import cv2
import numpy as np

from glob import glob

from os.path import join as path_join

def glob_file_paths(folder_path, n_max_samples = None, file_ext = '.png'):

    pattern = path_join(folder_path, '**', '*' + file_ext)

    file_paths = glob(pattern, recursive = True)

    n_samples = len(file_paths)

    if n_max_samples is not None and (n_samples > n_max_samples):
        print('INFO:glob_file_paths(): Picking out ' + str(n_max_samples) + ' random samples from a total set of ' + str(n_samples) + ' samples!')
        
        file_paths = random.sample(file_paths, n_max_samples)

    return file_paths

def glob_images(folder_path, n_max_images = 50):

    # Try looking for PNG images first
    file_paths = glob_file_paths(folder_path, n_max_samples = n_max_images)

    # Look for JPG if no PNG images were found
    if len(file_paths) == 0:
        file_paths = glob_file_paths(folder_path, n_max_samples = n_max_images, file_ext = '.jpg')

    n_images = len(file_paths)

    image_shape = cv2.imread(file_paths[0]).shape

    n_rows = image_shape[0]
    n_cols = image_shape[1]

    if len(image_shape) > 2:
        n_channels = 3
    else:
        n_channels = 1

    images = np.zeros((n_images, n_rows, n_cols, n_channels), dtype = 'uint8')

    for i in range(n_images):
        images[i] = cv2.imread(file_paths[i])

    return images

