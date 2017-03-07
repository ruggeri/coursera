# Detect lines in the image, averaging and extending them.
class BasicDetector:
    # Hough Constants
    RHO_ACCURACY = 1
    THETA_ACCURACY = (2 * np.pi) / 180
    THRESHOLD = 10
    MIN_LINE_LENGTH = 10
    MAX_LINE_GAP = 2

    def __init__(self, edge_image, logger):
        self.edge_image = edge_image
        self.lines = None

    def run(self):
        height, width = self.edge_image.shape

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
        self.lines = [line[0] for line in lines]

        logger.log_lines("BasicDetector/lines", self.lines)

        return self.lines
