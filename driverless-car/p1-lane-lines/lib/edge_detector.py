import cv2
import numpy as np

from .mask import perform_mask

# Gray and Blur Constants
BLUR_PIXELS = 11
# Canny Constants
CANNY_MIN_CUTOFF = 4
CANNY_MAX_CUTOFF = 64

class EdgeDetector:
    def __init__(self, shape, logger):
        self.gb_image = np.zeros(shape[0:2], dtype=np.uint8)
        self.edge_image = np.zeros(shape[0:2], dtype=np.uint8)
        self.logger = logger

    def perform_gray_and_blur(self, image):
        gb_image = self.gb_image

        # Gray the image and blur it.
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, dst = gb_image)
        cv2.GaussianBlur(
            gb_image, (BLUR_PIXELS, BLUR_PIXELS), 0, dst = gb_image
        )

        # Mask to an appropriate window in front of the car.
        height, width = gb_image.shape
        perform_mask(
            gb_image,
            height = height, width = width, extra_space = True,
            dst = gb_image
        )

        self.logger.log_image(
            "EdgeDetector/gray_and_blur_image", gb_image
        )

        return gb_image

    def detect_edges(self):
        edge_image = self.edge_image

        # Run edge detection.
        cv2.Canny(
            self.gb_image,
            CANNY_MIN_CUTOFF, CANNY_MAX_CUTOFF,
            edges = edge_image
        )

        # Need to mask again, else will detect edges of mask :-P
        # This mask is a little narrower to cutoff those edges.
        height, width = edge_image.shape
        perform_mask(
            edge_image,
            height = height, width = width, extra_space = False,
            dst = edge_image
        )

        self.logger.log_image("EdgeDetector/edge_image", edge_image)

        return edge_image

    def run(self, image):
        self.perform_gray_and_blur(image)
        self.detect_edges()
        return self.edge_image
