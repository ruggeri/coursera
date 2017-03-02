import sys
from .batcher import Batcher
from .evaluator import Evaluator
from .ff_neural_network import FFNeuralNetwork
from .trainer import Trainer

class Runner:
    DEFAULT_NUM_HIDDEN_UNITS = 10
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_BATCH_SIZE = 1
    VALIDATION_SET_RATIO = 0.10

    def __init__(self,
                 inputs,
                 targets,
                 num_hidden_units = None,
                 learning_rate = None,
                 batch_size = None,
                 validation_set_ratio = None):
        # A world of defaults!
        if num_hidden_units is None:
            num_hidden_units = self.DEFAULT_NUM_HIDDEN_UNITS
        if learning_rate is None:
            learning_rate = self.DEFAULT_LEARNING_RATE
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE
        if validation_set_ratio is None:
            validation_set_ratio = self.VALIDATION_SET_RATIO

        num_input_units = len(inputs[0])
        self.nn = FFNeuralNetwork(num_input_units, num_hidden_units)
        self.trainer = Trainer(self.nn, learning_rate)
        self.batcher = Batcher(batch_size)
        self.evaluator = Evaluator(self.nn)

        (self.train_set, self.validation_set) = \
          Evaluator.split_dataset(validation_set_ratio, inputs, targets)

        self.epoch_num = 0

    def run_epoch(self):
        batches = self.batcher.run(*self.train_set)
        for b_num, (b_inputs, b_targets) in enumerate(batches):
            self.trainer.train_batch(b_inputs, b_targets)
            error_rate = self.trainer.stats.error_rate()
            eps = self.trainer.stats.examples_per_second()
            strings = [f"\r\x1b[KEpoch {self.epoch_num}",
                       f"Batch #{b_num}.",
                       f"Train Error rate: {error_rate:.3f}",
                       f"Speed: {int(eps):5}ex/sec"]
            sys.stdout.write("\t".join(strings))

        result = self.evaluator.run(*self.validation_set)
        strings = [f"\r\x1b[K>>>Epoch {self.epoch_num}",
                   f"CE: {result.cross_entropy:.3f}",
                   f"Valid Error rate: {result.error_rate:.3f}<<<"]
        print("\t".join(strings))

        self.epoch_num += 1

# TODO: I would like to see the performance difference from using
# sparse arrays.
