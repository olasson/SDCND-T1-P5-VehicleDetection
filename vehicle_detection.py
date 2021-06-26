
import argparse

import numpy as np

from code.config import load_config, config_is_valid
from code.misc import file_exists, folder_is_empty
from code.io import glob_images, save_data, load_data
from code.plots import plot_images
from code.features import extract_features
from code.model import prepare_data, svc_train, svc_accuracy
from code.detect import VehicleDetector
from code.draw import draw_bounding_boxes

ERROR_PREFIX = 'ERROR_MAIN: '
INFO_PREFIX = 'INFO_MAIN: '

FOLDER_PATH_VEHICLES = './data/vehicles'
FOLDER_PATH_NON_VEHICLES = './data/non-vehicles'

KEY_VEHICLES = 'features_vehicles'
KEY_NON_VEHICLES = 'features_non_vehicles'

KEY_SVC = 'svc'
KEY_SCALER = 'scaler'


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = 'Advanced Lane Lines')

    # Images

    parser.add_argument(
        '--images',
        type = str,
        nargs = '?',
        default = '',
        help = 'Folder path to a set of images. Path to a folder containing more folders containing images is valid.',
    )

    parser.add_argument(
        '--detect',
        action = 'store_true',
        help = 'Run the vehicle detector on the set of images from --images.'
    )

    # Show

    parser.add_argument(
        '--show',
        action = 'store_true',
        help = 'Shows a set or subset of images'
    )

    parser.add_argument(
        '--n_max_images',
        type = int,
        default = 50,
        help = 'The maximum number of images that can be shown in the image plot.'
    )

    parser.add_argument(
        '--n_max_cols',
        type = int,
        default = 3,
        help = 'The maximum number of columns in the image plot.'
    )

    # Videos

    parser.add_argument(
        '--video',
        type = str,
        nargs = '?',
        default = '',
        help = 'File path to a folder video file to run the pipeline on.',
    )

    # Config

    parser.add_argument(
        '--config',
        type = str,
        nargs = '?',
        default = './config/config1.json',
        help = 'Path to a .json file containing project config.',
    ) 

    args = parser.parse_args()

    # Init config

    file_path_config = args.config

    config = load_config(file_path_config)

    if not config_is_valid(config):
        print(ERROR_PREFIX + 'Incorrectly structured config file at: ' + file_path_config + '! Check code/config.py for more info!')
        exit()

    # Init paths

    folder_path_images = args.images

    file_path_video = args.video

    file_path_features = config["pickled_features"]
    file_path_model = config["pickled_model"]

    # Init values

    n_max_images = args.n_max_images
    n_max_cols = args.n_max_cols

    # Init flags

    flag_show_images = args.show

    flag_run_on_images = args.detect 

    flag_features_are_extracted = file_exists(file_path_features)
    flag_model_exists = file_exists(file_path_model)

    # Checks

    if flag_show_images and folder_is_empty(folder_path_images):
        print(ERROR_PREFIX + 'You are trying to show a set of images, but the folder: ' + folder_path_images + ' is empty!')
        exit()

    if flag_run_on_images and folder_is_empty(folder_path_images):
        print(ERROR_PREFIX + 'You are trying to run the pipeline on a set of images, but the folder: ' + folder_path_images + ' is empty!')
        exit()

    # Show

    if flag_show_images:

        print(INFO_PREFIX + 'Showing images from folder: ' + folder_path_images)

        images = glob_images(folder_path_images, n_max_images = n_max_images)

        plot_images(images, title_fig_window = folder_path_images, n_max_cols = n_max_cols)

        exit()

    # Extract

    if not flag_model_exists:

        if flag_features_are_extracted:
            print(INFO_PREFIX + 'Loading features!')
            features_vehicles, features_non_vehicles = load_data(file_path_features, KEY_VEHICLES, KEY_NON_VEHICLES)
        else:
            print(INFO_PREFIX + 'Extracting features!')
            features_vehicles = extract_features(FOLDER_PATH_VEHICLES, config)
            features_non_vehicles = extract_features(FOLDER_PATH_NON_VEHICLES, config)

            print(INFO_PREFIX + 'Saving features!')
            save_data(file_path_features, features_vehicles, features_non_vehicles, KEY_VEHICLES, KEY_NON_VEHICLES)

    # Model

    if flag_model_exists:
        print(INFO_PREFIX + 'Loading model!')
        svc, X_scaler = load_data(file_path_model, KEY_SVC, KEY_SCALER)
    else:
        print(INFO_PREFIX + 'Preparing data!')
        X_train, y_train, X_test, y_test, X_scaler = prepare_data(features_vehicles, features_non_vehicles, random_state = 314)

        print(INFO_PREFIX + 'Training model!')
        svc = svc_train(X_train, y_train)

        accuracy = svc_accuracy(svc, X_test, y_test)
        print(INFO_PREFIX  + 'Model accuracy: ', accuracy)

        print(INFO_PREFIX + 'Saving model!')
        save_data(file_path_model, svc, X_scaler, KEY_SVC, KEY_SCALER)


    if flag_run_on_images:

        print(INFO_PREFIX + 'Runing pipeline on images from folder: ' + folder_path_images)

        images = glob_images(folder_path_images, n_max_images = n_max_images)

        n_rows, n_cols, n_channels = images[0].shape
        
        n_images = len(images)

        images_result = np.zeros((n_images, n_rows, n_cols, n_channels), dtype = np.uint8)
        titles = []

        vehicle_detector = VehicleDetector(svc, X_scaler, n_rows, n_cols, config, buffer_size = 0)

        for i in range(n_images):

            bounding_boxes = vehicle_detector.detect(images[i])

            images_result[i] = draw_bounding_boxes(images[i], bounding_boxes, fill = True)
            titles.append("Number of vehicles detected: " + str(len(bounding_boxes)))

            print(INFO_PREFIX + 'Processed image ' + str(i + 1) + ' of ' + str(n_images) + '!')

        plot_images(images_result, titles, title_fig_window = folder_path_images, n_max_cols = n_max_cols)




















