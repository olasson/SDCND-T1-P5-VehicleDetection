
import cv2

import numpy as np


def _draw_rectangle(image_tmp, window, color, thickness, fill = False):

    cv2.rectangle(image_tmp, window[0], window[1], color, thickness)

    if fill:
        fill_color = (color[0] //3, color[1] //3, color[2] //3)
        cv2.rectangle(image_tmp, window[0], window[1], fill_color, -1)


def draw_windows(image, windows, predictions, debug = False):

    image_tmp = np.zeros_like(image)

    for i in range(len(windows)):

        if predictions[i] == 1:

            _draw_rectangle(image_tmp, windows[i], (0, 255, 0), 3, fill = debug)
        else:
            if debug:
                _draw_rectangle(image_tmp, windows[i], (0, 0, 100), 1)

        
    image = cv2.addWeighted(image, 1, image_tmp, 0.7, 0)

    return image

def draw_bounding_boxes(image, bounding_boxes, fill = False):

    image_tmp = np.zeros_like(image)

    for bb in bounding_boxes:
        _draw_rectangle(image_tmp, bb, (255, 0, 0), 5, fill = fill)

    image = cv2.addWeighted(image, 1, image_tmp, 0.7, 0)

    return image