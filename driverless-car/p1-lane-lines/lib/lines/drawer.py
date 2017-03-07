# Draws a set of lines into an overlay.
class Drawer:
    DEFAULT_LINE_THICKNESS = 20

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.overlay = np.zeros((height, width), dtype=np.uint8)

    def draw_lines(self, lines, thickness = None):
        if thickness is None:
            thickness = self.DEFAULT_LINE_THICKNESS

        self.overlay *= 0

        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(
                self.overlay, (x1, y1), (x2, y2), 255, thickness
            )

        return self.overlay
