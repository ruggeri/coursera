CONFIG = [
    # Extend lines come first...
    #("OutlierToPrevLineFilter/rejected/LEFT", (150, 0, 105), 2),
    #("OutlierToPrevLineFilter/rejected/RIGHT", (150, 0, 105), 2),
    #("OutlierToPrevLineFilter/result/LEFT", (100, 200, 100), 2),
    #("OutlierToPrevLineFilter/result/RIGHT", (100, 200, 100), 2),
    ("LineAverager/result/LEFT", (255, 255, 0), 2),
    ("LineAverager/result/RIGHT", (255, 255, 0), 2),
    ("Smoother/result/LEFT", (255, 0, 255), 2),
    ("Smoother/result/RIGHT", (255, 0, 255), 2),

    # Then non-extended lines, to paint over.
    ("BasicDetector/lines", (0, 255, 0), 2),
    ("LeftRightSplitter/rejected", (0, 150, 105), 2),
    ("LowThetaFilter/rejected/LEFT", (0, 105, 150), 2),
    ("LowThetaFilter/rejected/RIGHT", (0, 105, 150), 2),
]

class Logger:
    def __init__(self):
        self.images = {}
        self.lines = {}

    def log_image(self, name, image):
        self.images[name] = image

    def log_lines(self, name, lines):
        self.lines[name] = lines

    def log_line(self, name, line):
        self.log_lines(name, [line])

    def dump_lines(self, image, drawer):
        for (name, color, thickness) in CONFIG:
            if name not in self.lines:
                print("nothing logged for {}".format(name))
                continue
            drawer.draw_lines(image, self.lines[name], color, thickness)
