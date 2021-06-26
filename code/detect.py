
import cv2

import numpy as np

from code.features import extract_image_features

def _image_region_search(image_region, v_min, h_min, scale, cells_per_step, config, svc, scaler):

    if scale != 1.0:

        if scale > 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv.INTER_LINEAR

        image_region = cv2.resize(image_region, (np.int(image_region.shape[0] / scale), np.int(image_region.shape[1] / scale)), interpolation = interpolation)

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

            window_image = image_region[window_min_v:window_min_v + config["window"] , window_min_h:window_min_h + config["window"]]

            features = extract_image_features(window_image, config, 3)

            features = scaler.transform(features.reshape(1, -1))

            prediction = svc.predict(features)[0]

            window_scale = np.int(config["window"] * scale)

            top_left = (np.int(window_min_h * scale) + h_min, np.int(window_min_v * scale) + v_min)

            bottom_right = (top_left[0] + window_scale, top_left[1] + window_scale)

            windows.append((top_left, bottom_right))

            predictions.append(prediction)


    return windows, predictions


def _image_search(image, regions, config, svc, scaler):

    windows = []
    predictions = []

    for v_min, v_max, h_min, h_max, scale, cells_per_step in regions:
        
        image_region = image[v_min:v_max, h_min:h_max, :]

        _windows, _predictions = _image_region_search(image_region, v_min, h_min, scale, cells_per_step, config, svc, scaler)

        windows.append(_windows)
        predictions.append(_predictions)

    # Flatten lists
    windows = [item for sublist in windows for item in sublist]
    predictions = [item for sublist in predictions for item in sublist]

    return windows, predictions

