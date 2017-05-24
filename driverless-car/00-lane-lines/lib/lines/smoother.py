import numpy as np

# TODO: I think there's an opportunity to use the weights of previous
# lines, rather than just treat them all equal. Some past lines you're
# more sure of than others.
class Smoother:
    SMOOTH_CONSTANT = 3

    def __init__(self, line_history, logger):
        self.line_history = line_history
        self.logger = logger

    def run(self, line, side):
        prev_line = self.line_history.get_prev_line(side)

        if prev_line is None:
            return np.array(line)

        smoothed_line = np.array(line, dtype=np.float32)
        smoothed_line += (self.SMOOTH_CONSTANT * prev_line)
        smoothed_line /= (1 + self.SMOOTH_CONSTANT)

        self.logger.log_line(
            "Smoother/result/{}".format(side), smoothed_line
        )

        return smoothed_line.astype(np.int32)
