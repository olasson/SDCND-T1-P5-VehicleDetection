
import cv2

import numpy as np


def draw_rectangles(image, rectangles, color, thickness, fill = False):

    rectangles = np.array(rectangles)

    if len(rectangles) == 0:
        return image

    image_tmp = np.zeros_like(image)

    for rectangle in rectangles:

        cv2.rectangle(image_tmp, rectangle[0], rectangle[1], color, thickness)

    if fill:

        for rectangle in rectangles:

            cv2.rectangle(image_tmp, rectangle[0], rectangle[1], (0, 50, 0), -1)

    image = cv2.addWeighted(image, 1, image_tmp, 0.7, 0)

    return image