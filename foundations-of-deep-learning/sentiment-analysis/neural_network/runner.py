from neural_network.batcher import Batcher
from neural_network.evaluator import Evaluator
from neural_network.ff_neural_network import FFNeuralNetwork
from neural_network.trainer import Trainer

class Runner:
    DEFAULT_NUM_HIDDEN_UNITS = 10
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_BATCH_SIZE = 200
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

    def run_epoch(self):
        batches = self.batcher.run(*self.train_set)
        for idx, (batch_inputs, batch_targets) in enumerate(batches):
            print(f"Training batch #{idx}")
            self.trainer.train_batch(batch_inputs, batch_targets)

            errors = self.evaluator.run(*self.validation_set)
            print(errors)
