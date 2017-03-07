class Logger:
    def __init__(self):
        pass

    def log_image(self, name, image):
        pass

    def log_lines(self, name, lines):
        pass

    def log_line(self, name, line):
        self.log_lines(name, [line])
