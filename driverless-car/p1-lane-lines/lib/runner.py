from .edge_detector import EdgeDetector
from .drawer import Drawer
from .lines.detector import LineDetector
from .logger import Logger

class Runner:
    def __init__(self, shape):
        self.logger = Logger()
        self.edge_detector = EdgeDetector(shape, self.logger)
        self.line_detector = LineDetector(shape, self.logger)
        self.drawer = Drawer(shape)

    def run(self, image):
        edge_image = self.edge_detector.run(image)
        left_line, right_line = self.line_detector.run(edge_image)

        self.drawer.draw_transparent_lines(
            image, [left_line, right_line], (255, 0, 0), 20
        )

        self.logger.dump_lines(image, self.drawer)

        return image
