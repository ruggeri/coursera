from math import ceil

class Batcher:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def run(self, inputs, targets):
        batches = []
        num_batches = ceil(len(inputs) / self.batch_size)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size
            end_idx = min(end_idx, len(inputs))

            batch_inputs = inputs[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]

            batches.append((batch_inputs, batch_targets))

        return batches
