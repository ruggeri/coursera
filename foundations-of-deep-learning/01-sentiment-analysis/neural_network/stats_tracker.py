from collections import deque, namedtuple

Result = namedtuple("Result", "true_pos true_neg false_pos false_neg")

class StatsTracker:
    DEFAULT_MAXLEN = 100

    def __init__(self, maxlen = DEFAULT_MAXLEN):
        self.events = deque(maxlen = maxlen)
        self.times = deque(maxlen = maxlen)

    def examples_per_second(self):
        return len(self.times) / sum(self.times)

    def log_example(self, output, target, time):
        if target == 0:
            if (output < 0.5):
                self.events.append("TRUE_NEG")
            else:
                self.events.append("FALSE_POS")
        else:
            if (output < 0.5):
                self.events.append("FALSE_NEG")
            else:
                self.events.append("TRUE_POS")

        self.times.append(time)

    def error_rate(self):
        results = self.results()
        num_wrong = results.false_pos + results.false_neg
        num = num_wrong + (results.true_pos + results.true_neg)
        return num_wrong / num

    def results(self):
        result_l = list([0, 0, 0, 0])

        for result in self.events:
            if result == "TRUE_POS":
                result_l[0] += 1
            elif result == "TRUE_NEG":
                result_l[1] += 1
            elif result == "FALSE_POS":
                result_l[2] += 1
            elif result == "FALSE_NEG":
                result_l[3] += 1
            else:
                raise "WTF?"

        return Result(*result_l)
