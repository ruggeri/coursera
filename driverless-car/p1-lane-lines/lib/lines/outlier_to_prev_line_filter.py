import numpy as np

from .helpers import distance_to_line

class OutlierToPrevLineFilter:
    MAX_DISTANCE = 25
    # The idea is that you don't want too many lines, but you want at
    # least two to average with.
    MIN_NUM_LINES = 2

    def __init__(self, line_history, logger):
        self.line_history = line_history
        self.logger = logger

    def run_with_distance(self, lines, side, max_distance):
        prev_line = self.line_history.get_prev_line(side)
        if prev_line is None:
            return ([], lines)

        rejected_lines = []
        good_lines = []
        for line in lines:
            p1 = (line[0], line[1])
            d1 = distance_to_line(p1, prev_line)
            p2 = (line[2], line[3])
            d2 = distance_to_line(p2, prev_line)

            if (d1 > self.MAX_DISTANCE) or (d2 > max_distance):
                rejected_lines.append(line)
            else:
                good_lines.append(line)

        return (rejected_lines, good_lines)

    def run(self, lines, side):
        max_distance = self.MAX_DISTANCE
        while True:
            rejected_lines, good_lines = self.run_with_distance(
                lines, side, max_distance
            )

            if len(good_lines) >= self.MIN_NUM_LINES:
                break
            else:
                max_distance *= 1.25

        self.logger.log_lines(
            "OutlierToPrevLineFilter/rejected/{}".format(side),
            rejected_lines
        )
        self.logger.log_lines(
            "OutlierToPrevLineFilter/result/{}".format(side),
            good_lines
        )

        return good_lines
