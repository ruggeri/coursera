class Runner:
    def __init__(self, shape):
        self.logger = Logger()
        self.edge_detector = EdgeDetector(shape, logger)
        self.line_detector = LineDetector(shape, logger)
        self.drawer = Drawer()

    def run(self, image):
        edge_image = self.edge_detector.run(image)
        left_line, right_line = self.line_detector.run(edge_image)

        self.drawer.draw_transparent_lines(
            image, [left_line, right_line], (255, 0, 0), 20
        )
