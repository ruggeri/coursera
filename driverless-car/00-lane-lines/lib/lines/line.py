class Line:
    def __init__(self, orig_line, current_line = None):
        self.orig_line = orig_line
        if current_line is None:
            current_line = orig_line
        self.current_line = self.orig_line

    def copy(self):
        return Line(self.orig_line, self.current_line)

    def __getitem__(self, idx):
        return self.current_line[idx]
