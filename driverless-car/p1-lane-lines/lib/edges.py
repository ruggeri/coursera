import cv2
from lib.mask import build_mask, apply_mask
import matplotlib.pyplot as plt

import lib.config as config

# Converts image to grayscale and blurs.
BLUR_PIXELS = 11
def preprocess_image(image):
    height, width, depth = image.shape

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.GaussianBlur(
        gray_image, (BLUR_PIXELS, BLUR_PIXELS), 0
    )

    # Mask to an appropriate window in front of the car.
    # I'll make this a little wider than the next mask.
    mask = build_mask(
        height = height, width = width, extra_space = True
    )
    masked_image = apply_mask(gray_image, mask)

    # Display intermediate result of blur/mask.
    config.LAST_MASKED_IMAGE = masked_image
    if config.DEBUG_MODE:
        plt.figure()
        plt.imshow(masked_image, cmap="gray")

    return masked_image

# Canny Constants
CANNY_MIN_CUTOFF = 4
CANNY_MAX_CUTOFF = 64

# Runs edge detection, then applies another mask.
def detect_edges(masked_image):
    height, width = masked_image.shape

    # Run edge detection.
    edges = cv2.Canny(masked_image, CANNY_MIN_CUTOFF, CANNY_MAX_CUTOFF)

    # Need to mask again, else will detect edges of mask :-P
    # This mask is a little narrower to cutoff those edges.
    mask = build_mask(
        height = height, width = width, extra_space = False
    )
    masked_edges = apply_mask(edges, mask)

    # Display intermediate result of edge detection.
    config.LAST_MASKED_EDGES = masked_edges
    if config.DEBUG_MODE:
        plt.figure()
        plt.imshow(masked_edges, cmap="gray")

    return masked_edges
