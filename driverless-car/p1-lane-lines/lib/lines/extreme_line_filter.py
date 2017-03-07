import numpy as np

from .helpers import line_theta, line_length, weighted_median

class ExtremeLineFilter:
    # Ignore lines below this theta.
    THETA_MIN = 15 * ((2 * np.pi) / 360)

    # Penalize lines this much different than previous line.
    PREV_THETA_DIFF_THRESHOLD = 10 * ((2 * np.pi) / 360)
    TOO_DIFFERENT_PENALTY = 0.75

    # Ignore outlier lines.
    MEDIAN_THETA_DIFF_THRESHOLD = 10 * ((2 * np.pi) / 360)

    def __init__(self, line_history, logger):
        self.line_history = line_history
        self.logger = logger

    # Lines at a very great angle to the car's direction are not
    # likely to be useful. Lane lines go forward.
    def remove_low_theta_lines(self, lines, side):
        too_low_theta_lines = []
        good_lines = []

        for line in lines:
            theta = line_theta(line)
            if (abs(theta) < self.THETA_MIN):
                too_low_theta_lines.append(line)
                continue
            else:
                good_lines.append(line)

        self.logger.log_lines(
            "ExtremeLineFilter/too_low_theta/{}".format(side),
            too_low_theta_lines
        )

        return good_lines

    # Penalize lines that are too different from the line calculated in
    # the previous time step.
    def penalize_too_different_lines(self, lines_and_weights, side):
        # If this is the first time step, ignore this penalization
        # method.
        prev_line = self.line_history.get_prev_line(side)
        if prev_line is None:
            return lines_and_weights
        prev_line_theta = line_theta(prev_line)

        too_different_lines = []
        new_lines_and_weights = []

        for (line, weight) in lines_and_weights:
            theta = line_theta(line)
            theta_diff = abs(prev_line_theta - theta)

            should_penalize = (
                (theta_diff > self.PREV_THETA_DIFF_THRESHOLD)
            )

            if should_penalize:
                too_different_lines.append(line)
                weight *= (1 - self.TOO_DIFFERENT_PENALTY)

            new_lines_and_weights.append((line, weight))

        self.logger.log_lines(
            "ExtremeLineFilter/too_different/{}".format(side),
            too_different_lines
        )

        return new_lines_and_weights

    # Assign a line an initial weight based on its length.
    def assign_lines_initial_weights(self, lines):
        lines_and_weights = []
        for line in lines:
            weight = line_length(line)
            lines_and_weights.append((line, weight))

        return lines_and_weights

    # Calculate a weighted median theta value.
    def median_theta(self, lines_and_weights):
        thetas_and_weights = list(
            map(lambda p: (line_theta(p[0]), p[1]), lines_and_weights)
        )

        return weighted_median(thetas_and_weights)

    # Remove lines that are too different from the calculated median
    # theta.
    def remove_outlier_lines(self, median_theta, lines, side):
        outlier_lines = []
        good_lines = []

        for line in lines:
            theta = line_theta(line)
            theta_diff = abs(theta - median_theta)
            if theta_diff > self.MEDIAN_THETA_DIFF_THRESHOLD:
                # This is an outlier; skip it.
                outlier_lines.append(line)
                continue

            good_lines.append(line)

        self.logger.log_lines(
            "ExtremeLineFilter/outlier/{}".format(side), outlier_lines
        )

        return good_lines

    def run(self, lines, side):
        # Remove low theta lines.
        lines = self.remove_low_theta_lines(lines, side)

        # Calculate median theta.
        lines_and_weights = self.assign_lines_initial_weights(lines)
        lines_and_weights = self.penalize_too_different_lines(
            lines_and_weights, side
        )
        median_theta = self.median_theta(lines_and_weights)

        # Remove outliers.
        lines = self.remove_outlier_lines(median_theta, lines, side)

        return lines
