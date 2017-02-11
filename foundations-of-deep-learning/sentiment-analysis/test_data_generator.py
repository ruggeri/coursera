import numpy as np
import random

class TestDataGenerator:
    THRESHOLD = 0.25

    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions
        self.threshold = self.THRESHOLD

    def gen_samples(self, num_samples):
        inputs = []
        targets = []

        for i in range(num_samples):
            (input_v, target) = self.gen_sample()
            inputs.append(input_v)
            targets.append(target)

        return (inputs, targets)

    def gen_sample(self):
        target = random.choice([0, 1])
        input_v = np.zeros((self.num_dimensions, 1))
        for i in range(self.num_dimensions):
            x = random.random()
            if (i % 2 == 0) and (target == 0):
                input_v[i] = 1 if x > self.threshold else 0
            elif (i % 2 == 0) and (target == 1):
                input_v[i] = 0 if x > self.threshold else 1
            elif (i % 2 == 1) and (target == 0):
                input_v[i] = 0 if x > self.threshold else 1
            elif (i % 2 == 1) and (target == 1):
                input_v[i] = 1 if x > self.threshold else 0

        return (input_v, target)
