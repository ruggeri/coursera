class LineAverager:
    def __init__(self, min_height, max_height, logger):
        self.min_height = min_height
        self.max_height = max_height

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
            self.add_line(line, line_length(line))

    def fit_average_line(self):
        # Fits a line of best fit.
        p = np.polyfit(self.ys, self.xs, 1, w = self.ws)

        # Calculates endpoints of best fit line
        min_x = p[0] * self.min_height + p[1]
        max_x = p[0] * self.max_height + p[1]
        return [min_x, self.min_height, max_x, self.max_height]

    def run(self, lines, side):
        self.add_lines(lines)

        average_line = self.fit_average_line()

        self.logger.log_line(
            f"LineAverager/{side}/result", average_line
        )

        return average_line
