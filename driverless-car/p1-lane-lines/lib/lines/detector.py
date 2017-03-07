from .averager import LineAverager
from .basic_detector import BasicDetector
from .extender import Extender
from .extreme_line_filter import ExtremeLineFilter
from .history import LineHistory
from .left_right_splitter import LeftRightSplitter
from .smoother import Smoother

class LineDetector:
    HORIZON_RATIO = 0.35

    def __init__(self, image_shape, logger):
        self.image_shape = image_shape
        self.logger = logger
        self.line_history = LineHistory()

    def min_max_height(self):
        self.min_height = int(
            self.image_shape[0] * (1 - self.HORIZON_RATIO)
        )
        self.max_height = self.image_shape[1]

        return (self.min_height, self.max_height)

    def run(self, edge_image):
        basic_detector = BasicDetector(self.logger)
        basic_lines = basic_detector.run(edge_image)

        lr_splitter = LeftRightSplitter(
            self.image_shape[1], self.logger
        )
        left_lines, right_lines = lr_splitter.run(basic_lines)

        extreme_line_filter = ExtremeLineFilter(
            self.line_history, self.logger
        )
        left_lines = extreme_line_filter.run(left_lines, "LEFT")
        right_lines = extreme_line_filter.run(right_lines, "RIGHT")

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

        self.line_history.add(left_line, "LEFT")
        self.line_history.add(right_line, "RIGHT")

        return (left_line, right_line)
