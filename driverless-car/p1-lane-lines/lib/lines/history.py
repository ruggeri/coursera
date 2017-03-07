import numpy as np

class LineHistory:
    NUM_LINES_HISTORY = 5

    def __init__(self):
        self.left_lines = []
        self.right_lines = []

    def get_lines(self, side):
        if side == "LEFT":
            return self.left_lines
        elif side == "RIGHT":
            return self.right_lines
        else:
            raise Exception("Didn't ask for left or right lines")

    def get_prev_line(self, side):
        lines = self.get_lines(side)
        if len(lines) == 0:
            return None
        else:
            return lines[-1]

    def add(self, line, side):
        lines = self.get_lines(side)
        if len(lines) == self.NUM_LINES_HISTORY:
            lines.pop(0)
        lines.append(np.array(line))
