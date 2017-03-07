import numpy as np

from .helpers import line_theta

class LowThetaFilter:
    THETA_MIN = 15 * ((2 * np.pi) / 360)

    def __init__(self, logger):
        self.logger = logger

    # Lines at a very great angle to the car's direction are not
    # likely to be useful. Lane lines go forward.
    def run(self, lines, side):
        rejected_lines = []
        good_lines = []

        for line in lines:
            theta = line_theta(line)
            if (abs(theta) < self.THETA_MIN):
                rejected_lines.append(line)
                continue
            else:
                good_lines.append(line)

        self.logger.log_lines(
            "LowThetaFilter/rejected/{}".format(side), rejected_lines
        )
        self.logger.log_lines(
            "LowThetaFilter/result/{}".format(side), good_lines
        )

        return good_lines
