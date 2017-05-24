import cv2
import numpy as np

from .line import Line

# Detect lines in the image, averaging and extending them.
class BasicDetector:
    # Hough Constants
    RHO_ACCURACY = 1
    THETA_ACCURACY = (1 * np.pi) / 180
    THRESHOLD = 5
    MIN_LINE_LENGTH = 5
    MAX_LINE_GAP = 2

    def __init__(self, logger):
        self.logger = logger

    def run(self, edge_image):
        height, width = edge_image.shape

        # Detect lines.
        lines = cv2.HoughLinesP(
            edge_image,
            rho = self.RHO_ACCURACY,
            theta = self.THETA_ACCURACY,
            threshold = self.THRESHOLD,
            minLineLength = self.MIN_LINE_LENGTH,
            maxLineGap = self.MAX_LINE_GAP
        )

        if lines is None:
            raise Exception("No lines detected!")

        # Unpack the lines from unneeded nesting.
        lines = [Line(line[0]) for line in lines]

        self.logger.log_lines("BasicDetector/lines", lines)

        return lines
