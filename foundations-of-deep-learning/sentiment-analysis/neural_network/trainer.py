from neural_network.back_propagator import BackPropagator
import numpy as np

class Trainer:
    def __init__(self, neural_network, learning_rate):
        self.nn = neural_network
        self.bp = BackPropagator(self.nn)

        # Arrays for collecting GD step
        self.output_bias_step = 0.0
        self.output_weights_step = \
          np.zeros(self.nn.hidden_to_output_weights.shape)
        self.hidden_bias_step = np.zeros(self.nn.hidden_layer.shape)
        self.hidden_weights_step = \
          np.zeros(self.nn.input_to_hidden_weights.shape)

        self.learning_rate = learning_rate

    def perform_update(self, num_examples):
        scaled_learning_rate = self.learning_rate / num_examples

        self.nn.output_bias += \
          scaled_learning_rate * self.output_bias_step
        self.nn.hidden_to_output_weights += \
          scaled_learning_rate * self.output_weights_step
        self.nn.hidden_bias += \
          scaled_learning_rate * self.hidden_bias_step
        self.nn.input_to_hidden_weights += \
          scaled_learning_rate * self.hidden_weights_step

    def reset_steps(self):
        self.output_bias_step *= 0
        self.output_weights_step *= 0
        self.hidden_bias_step *= 0
        self.hidden_weights_step *= 0

    def train_batch(self, batch_inputs, batch_targets):
        self.reset_steps()

        input_target_pairs = zip(batch_inputs, batch_targets)
        for (input_v, target) in input_target_pairs:
            self.train_on_example(input_v, target)

        # Adjust weights!
        self.perform_update(len(batch_inputs))

    def train_on_example(self, input_v, target):
        output = self.nn.run(input_v)
        self.bp.backward_propagate(output, target)

        # Subtract so that we *decrease* the CE.
        self.output_bias_step -= self.bp.output_input_derivative
        self.output_weights_step -= self.bp.output_weights_derivative
        self.hidden_bias_step -= self.bp.hidden_input_derivative
        self.hidden_weights_step -= self.bp.hidden_weights_derivative
