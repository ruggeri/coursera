CONFIG = [
    ("BasicDetector/lines", (0, 255, 0), 2),
    #("LeftRightSplitter/left", (255, 255, 0), 2),
    #("LeftRightSplitter/right", (0, 255, 255), 2),
    #("LeftRightSplitter/rejected", (255, 0, 255), 2),
    #("ExtremeLineFilter/too_low_theta/LEFT", (0, 0, 255), 2),
    #("ExtremeLineFilter/too_low_theta/RIGHT", (0, 0, 255), 2),
    ("ExtremeLineFilter/result/LEFT", (0, 0, 255), 2),
    ("ExtremeLineFilter/result/RIGHT", (0, 0, 255), 2),
    ("LineAverager/result/LEFT", (255, 255, 0), 2),
    ("LineAverager/result/RIGHT", (255, 255, 0), 2),
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
