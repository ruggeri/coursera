# Splits lines into a left and right set.
class LeftRightSplitter:
    SAFETY_MARGIN = 0.025

    def __init__(self, width, logger):
        self.width = width
        self.logger = logger

    def run(self, lines):
        left_lines = []
        right_lines = []
        rejected_lines = []

        # Sometimes a line does extend past the middle; we'll filter
        # those out. In fact, we don't want any lines, too close to
        # the middle, even if they don't cross it.
        min_x = (self.width / 2) - (width * self.SAFETY_MARGIN)
        max_x = (self.width / 2) + (width * self.SAFETY_MARGIN)

        for line in self.lines:
            if line[0] < min_x and (line[2] < min_x):
                left_lines.append(line)
            elif (line[0] > max_x) and (line[2] > max_x):
                right_lines.append(line)
            else:
                rejected_lines.append(line)

        if (len(left_lines) == 0):
            raise Exception("Why no lines on left side?")
        if (len(right_lines) == 0):
            raise Exception("Why no lines on right side?")

        self.logger.log_lines("LeftRightSplitter/left", left_lines)
        self.logger.log_lines("LeftRightSplitter/right", right_lines)
        self.logger.log_lines("LeftRightSplitter/rejected", rejected_lines)

        return (left_lines, right_lines)
