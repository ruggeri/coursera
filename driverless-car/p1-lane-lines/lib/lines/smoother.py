import numpy as np

# TODO: I think there's an opportunity to use the weights of previous
# lines, rather than just treat them all equal. Some past lines you're
# more sure of than others.
class Smoother:
    DECAY_CONSTANT = 0.5

    def __init__(self, line_history, logger):
        self.line_history = line_history
        self.logger = logger

    def run(self, line, side):
        prev_line = self.line_history.get_prev_line(side)

        if prev_line is None:
            return np.array(line)

        new_line = np.array(line)
        new_line += (self.DECAY_CONSTANT * prev_line)
        new_line /= (1 + self.DECAY_CONSTANT)

        logger.log_lines(f"Smoother/{side}/result", [new_line])

        return new_line
