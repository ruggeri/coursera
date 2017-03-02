import cv2
import numpy as np

import lib.config as config

# Mask constants
MASK_HEIGHT_RATIO = 0.35
MASK_X_RATIO = 0.2
MASK_EXTRA_WINDOW = 0.01

def build_mask(height, width, extra_space):
    min_y = height * (1 - MASK_HEIGHT_RATIO)
    min_y *= (1 - MASK_EXTRA_WINDOW) if extra_space else 1.0
    min_y = int(min_y)

    min_x = width/2 * (1 - MASK_X_RATIO)
    min_x *= (1 - MASK_EXTRA_WINDOW) if extra_space else 1.0
    min_x = int(min_x)

    max_x = width/2 * (1 + MASK_X_RATIO)
    max_x *= (1 + MASK_EXTRA_WINDOW) if extra_space else 1.0
    max_x = int(max_x)

    poly = np.array([
        (0, height),
        (min_x, min_y),
        (max_x, min_y),
        (width, height)
    ])

    return poly

# Inspired by code from Udacity.
def apply_mask(image, poly):
    mask = np.zeros_like(image)

    dimensions = len(image.shape)
    if dimensions == 3:
        num_channels = image.shape[2]
        mask_color = (255,) * 3
    else:
        # Monochrome
        mask_color = 255

    # All black except polygon, which is painted white.
    cv2.fillPoly(mask, [poly], mask_color)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
