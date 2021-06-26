
import cv2

import numpy as np

from scipy.ndimage.measurements import label

from code.features import extract_image_features

from collections import deque

def _image_region_search(image_region, v_min, h_min, scale, cells_per_step, config, svc, scaler):

    if scale != 1.0:

        if scale > 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        image_region = cv2.resize(image_region, (np.int(image_region.shape[1] / scale), np.int(image_region.shape[0] / scale)), interpolation = interpolation)


    n_hblocks = (image_region.shape[1] // config["pix_per_cell"]) - config["cell_per_block"] + 1
    n_vblocks = (image_region.shape[0] // config["pix_per_cell"]) - config["cell_per_block"] + 1

    n_blocks_per_window = (config["window"] // config["pix_per_cell"]) - config["cell_per_block"] + 1


    h_steps = (n_hblocks - n_blocks_per_window) // cells_per_step + 1
    v_steps = (n_vblocks - n_blocks_per_window) // cells_per_step + 1

    windows = []
    predictions = []

    for h_step in range(h_steps):
        for v_step in range(v_steps):
            h_pos = h_step * cells_per_step
            v_pos = v_step * cells_per_step

            window_min_h = h_pos * config["pix_per_cell"]
            window_min_v = v_pos * config["pix_per_cell"]

            image_window = image_region[window_min_v:window_min_v + config["window"] , window_min_h:window_min_h + config["window"]]

            if image_window.shape[0] < config["window"]  or image_window.shape[1] < config["window"] :
                image_window = cv2.resize(image_window, (config["window"], config["window"]), interpolation = cv.INTER_LINEAR)

            features = extract_image_features(image_window, config, 3)

            features = scaler.transform(features.reshape(1, -1))

            prediction = svc.predict(features)[0]

            window_scale = np.int(config["window"] * scale)

            top_left = (np.int(window_min_h * scale) + h_min, np.int(window_min_v * scale) + v_min)

            bottom_right = (top_left[0] + window_scale, top_left[1] + window_scale)

            windows.append((top_left, bottom_right))

            predictions.append(prediction)


    return windows, predictions


def _image_search(image, search_parameters, config, svc, scaler):

    windows = []
    predictions = []

    for v_min, v_max, h_min, h_max, scale, cells_per_step in search_parameters:
        
        image_region = image[v_min:v_max, h_min:h_max, :]

        _windows, _predictions = _image_region_search(image_region, v_min, h_min, scale, cells_per_step, config, svc, scaler)

        windows.append(_windows)
        predictions.append(_predictions)

    # Flatten lists
    windows = [item for sublist in windows for item in sublist]
    predictions = [item for sublist in predictions for item in sublist]

    return windows, predictions

def _make_heatmap(windows, predictions, n_rows, n_cols):

    heatmap = np.zeros((n_rows, n_cols), dtype = np.float)

    n_samples = len(windows)


    for i in range(n_samples):

        if predictions[i] == 1:
            window = windows[i]
            heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += (1.5 * sum(predictions))

    return heatmap


def _bounding_boxes(heatmap, min_width, min_height):

    labels = label(heatmap)

    bounding_boxes = []

    for car_n in range(1, labels[1] + 1):

        tmp = (labels[0] == car_n).nonzero()

        nonzero_x = np.array(tmp[1])
        nonzero_y = np.array(tmp[0])

        top_left = (np.min(nonzero_x), np.min(nonzero_y))
        bottom_right = (np.max(nonzero_x), np.max(nonzero_y))

        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]

        if (width >= min_width) and (height >= min_height):
            bounding_boxes.append((top_left, bottom_right))


    return bounding_boxes


class VehicleDetector:

    def __init__(self, svc, scaler, n_rows, n_cols, config, buffer_size = 8):
        
        self.svc = svc
        self.scaler = scaler

        self.n_rows = n_rows
        self.n_cols = n_cols

        n_rows_min = int(n_rows / 1.8)
        n_cols_min = 100

        self.search_parameters  = [(n_rows_min, (n_rows_min + 100), n_cols // 4,  n_cols,   1, 1),
                                   (n_rows_min, (n_rows_min + 130), n_cols // 3,  n_cols,   1, 2),
                                   (n_rows_min, (n_rows_min + 200), n_cols // 2,  n_cols, 1.5, 2),
                                   (n_rows_min, (n_rows_min + 250), n_cols_min,   n_cols,   2, 1)]

        self.config = config
        
        self.heatmap_buffer = deque(maxlen = buffer_size)

    def detect(self, image):

        windows, predictions = _image_search(image, self.search_parameters, self.config, self.svc, self.scaler)

        heatmap = _make_heatmap(windows, predictions, self.n_rows, self.n_cols)

        self.heatmap_buffer.append(heatmap)

        if len(self.heatmap_buffer) > 1:
            heatmap = np.average(self.heatmap_buffer, axis = 0)

        bounding_boxes = _bounding_boxes(heatmap, (0.8 * self.config["window"]), (0.5 * self.config["window"]))

        return bounding_boxes