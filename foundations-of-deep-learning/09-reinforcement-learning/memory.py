from collections import deque
import config
import random

class Memory():
    def __init__(self):
        self.examples = deque(maxlen = config.MEMORY_LEN)

    def add_example(self, example):
        self.examples.append(example)

    def training_batch(self):
        examples = random.sample(
            self.examples,
            min(len(self.examples), config.MINIBATCH_SIZE)
        )
        return examples[:config.MINIBATCH_SIZE]

    def num_points(self):
        return len(self.examples)
