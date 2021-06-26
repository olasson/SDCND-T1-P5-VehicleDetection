
import cv2

import numpy as np


def _draw_window(image_tmp, window, color, thickness, fill = False):

    cv2.rectangle(image_tmp, window[0], window[1], color, thickness)

    if fill:
        cv2.rectangle(image_tmp, window[0], window[1], color, -1)


def draw_windows(image, windows, predictions, debug = False):

    image_tmp = np.zeros_like(image)

    for i in range(len(windows)):

        if predictions[i] == 1:

            _draw_window(image_tmp, windows[i], (0, 255, 0), 3, fill = True)
        else:
            if debug:
                _draw_window(image_tmp, windows[i], (0, 0, 100), 1)

        
    image = cv2.addWeighted(image, 1, image_tmp, 0.7, 0)

    return image




"""
def draw_rectangles(image, rectangles, color, thickness, fill = False):

    #rectangles = np.array(rectangles)

    if len(rectangles) == 0:
        return image

    image_tmp = np.zeros_like(image)

    #rectangles = [rectangles]

    for rectangle in rectangles:

        cv2.rectangle(image_tmp, rectangle[0], rectangle[1], color, thickness)

    if fill:

        for rectangle in rectangles:

            cv2.rectangle(image_tmp, rectangle[0], rectangle[1], (0, 50, 0), -1)

    image = cv2.addWeighted(image, 1, image_tmp, 0.7, 0)

    return image
"""