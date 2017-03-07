import cv2
import numpy as np

# Mask constants
MASK_HEIGHT_RATIO = 0.35
MASK_X_RATIO = 0.2
MASK_EXTRA_WINDOW = 0.01

def build_mask_polygon(height, width, extra_space):
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

# Builds a mask where the polygon is entirely white and the rest is
# entirely black.
masks = {}
def build_mask(image_shape, poly):
    # Try to reuse previously allocated mask.
    if image_shape not in masks:
        mask = np.zeros_like(image)
        masks[image_shape] = mask
    else:
        # TODO: Still dumb, don't need to refill mask constantly.
        mask = masks[image_shape]
        mask *= 0

    if dimensions == 3:
        num_channels = image_shape[2]
        mask_color = (255,) * num_channels
    else:
        # Monochrome
        mask_color = 255

    # Paint the polygon white.
    cv2.fillPoly(mask, [poly], mask_color)

    return mask

def perform_mask(image, height, width, extra_space, dst = None):
    poly = build_mask_polygon(
        height = height, width = width, extra_space = extra_space
    )
    mask = build_mask(image.shape, poly)
    masked_image = cv2.bitwise_and(image, mask, dst = dst)
    return masked_image
