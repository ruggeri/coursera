import numpy as np

from .helpers import line_length

class LineAverager:
    def __init__(self, min_height, max_height, logger):
        self.min_height = min_height
        self.max_height = max_height
        self.logger = logger
        self.reset()

    def reset(self):
        self.xs = []
        self.ys = []
        self.ws = []

    def add_endpoint(self, x, y, w):
        self.xs.append(x)
        self.ys.append(y)
        self.ws.append(w)

    def add_line(self, line, weight):
        self.add_endpoint(line[0], line[1], weight)
        self.add_endpoint(line[2], line[3], weight)

    def add_lines(self, lines):
        # TODO: Probably want to knock down importance of too
        # different lines again?
        for line in lines:
            # I have chosen to give weight proportional to the length
            # of the original line in the image, regardless of whether
            # it has been extended.
            weight = line_length(line.orig_line)
            self.add_line(line, weight)

    def fit_average_line(self):
        # Fits a line of best fit.  TODO: All my lines start/end at
        # same y coordinats. So this is really just a weigheted
        # average of the xs, I believe.
        p = np.polyfit(self.ys, self.xs, 1, w = self.ws)

        # Calculates endpoints of best fit line
        min_x = int(p[0] * self.min_height + p[1])
        max_x = int(p[0] * self.max_height + p[1])
        return [min_x, self.min_height, max_x, self.max_height]

    def run(self, lines, side):
        self.reset()
        self.add_lines(lines)

        average_line = self.fit_average_line()

        self.logger.log_line(
            "LineAverager/result/{}".format(side), average_line
        )

        return average_line
