# Extends a detected line all the way from `y=min_height` to
# `y=max_height`.
class Extender:
    def __init__(self, min_height, max_height):
        self.min_height = min_height
        self.max_height = max_height

    def run_line(self, line):
        x1, y1 = line[0], line[1]
        x2, y2 = line[2], line[3]

        # Make (x1, y1) the more distant point, and (x2, y2) the
        # closer one.
        if y1 > y2:
            (x2, y2), (x1, y1) = (x1, y1), (x2, y2)

        # The following logic is technically incorrect for lines
        # parallel to either axis.
        if x1 == x2:
            print("Endpoints have same x value: {}".format(line))
            return None
        if y1 == y2:
            print("Endpoints have same y value: {}".format(line))
            return None

        m = (y2 - y1) / (x2 - x1)

        # Calculate new endpoints.
        y0 = self.min_height
        x0 = int(x1 + (y0 - y1) / m)
        y3 = self.max_height
        x3 = int(x1 + ((y3 - y1) / m))

        new_line = line.copy()
        new_line.current_line = [x0, y0, x3, y3]
        return new_line

    def run(self, lines):
        extended_lines = []
        for line in lines:
            new_line = self.run_line(line)
            if new_line is not None:
                extended_lines.append(new_line)

        return extended_lines
