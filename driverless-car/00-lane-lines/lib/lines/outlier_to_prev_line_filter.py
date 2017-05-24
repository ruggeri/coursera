import numpy as np

from .helpers import distance_to_line

class OutlierToPrevLineFilter:
    MAX_NEAR_DISTANCE = 60
    MAX_FAR_DISTANCE = 30

    # The idea is that you don't want too many lines, but you want at
    # least two to average with.
    MIN_NUM_LINES = 2
    DISTANCE_GROWTH = 1.25

    def __init__(self, line_history, logger):
        self.line_history = line_history
        self.logger = logger

    def run_with_distance(self, lines, side, factor):
        prev_line = self.line_history.get_prev_line(side)
        if prev_line is None:
            return ([], lines)

        rejected_lines = []
        good_lines = []
        for line in lines:
            p_near = (line[0], line[1])
            p_far = (line[2], line[3])

            if p_near[1] < p_far[1]:
                p_near, p_far = p_far, p_near

            d_near = distance_to_line(p_near, prev_line)
            d_far = distance_to_line(p_far, prev_line)

            max_far_distance = self.MAX_FAR_DISTANCE * factor
            max_near_distance = self.MAX_NEAR_DISTANCE * factor
            if (d_far > max_far_distance) or (d_near > max_near_distance):
                rejected_lines.append(line)
            else:
                good_lines.append(line)

        return (rejected_lines, good_lines)

    def run(self, lines, side):
        distance_factor = 1.0
        while True:
            rejected_lines, good_lines = self.run_with_distance(
                lines, side, distance_factor
            )

            if len(good_lines) >= self.MIN_NUM_LINES:
                break
            else:
                distance_factor *= self.DISTANCE_GROWTH

        self.logger.log_lines(
            "OutlierToPrevLineFilter/rejected/{}".format(side),
            rejected_lines
        )
        self.logger.log_lines(
            "OutlierToPrevLineFilter/result/{}".format(side),
            good_lines
        )

        return good_lines
