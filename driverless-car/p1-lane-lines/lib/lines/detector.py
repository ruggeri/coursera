detector = None
class Detector:
    HORIZON_RATIO = 0.35

    @staticmethod
    def run(edges_image, logger):
        if detector is None:
            detector = Detector(edges_image.shape, logger)

        return detector.run(edges_image)

    def __init__(self, image_shape, logger):
        self.image_shape = image_shape
        self.logger = logger
        self.line_history = LineHistory()

    def min_max_height(self):
        self.min_height = int(
            self.image_shape[0] * (1 - self.HORIZON_RATIO)
        )
        self.max_height = self.image_shape[1]

        return (min_height, max_height)

    def run(edges_image):
        basic_detector = BasicDetector(edge_image, self.logger)
        basic_lines = basic_detector.run()

        lr_splitter = LeftRightSplitter(
            basic_lines, self.width, self.logger
        )
        left_lines, right_lines = lr_splitter.run(lines)

        extreme_line_filter = ExtremeLineFilter(
            self.line_history, logger
        )
        left_lines = extreme_line_filter.run(lines, "LEFT")
        right_lines = extreme_line_filter.run(lines, "RIGHT")

        line_averager = LineAverager(
            *self.min_max_height(), self.logger
        )
        left_line = line_averager.run(left_lines, "LEFT")
        right_line = line_averager.run(right_lines, "RIGHT")

        smoother = Smoother(self.line_history, self.logger)
        left_line = smoother.run(left_line, "LEFT")
        right_line = smoother.run(right_line, "RIGHT")

        extender = Extender(*self.min_max_height())
        left_line = extender.run(left_line)
        right_line = extender.run(right_line)

        line_history.add(left_line, "LEFT")
        line_history.add(right_line, "RIGHT")

        return (left_line, right_line)
