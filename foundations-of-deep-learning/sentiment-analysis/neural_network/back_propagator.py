from neural_network.functions import ce_derivative, sigmoid_derivative
import numpy as np

class BackPropagator:
    def __init__(self, neural_network):
        self.nn = neural_network

        self.output_input_derivative = 0.0
        self.output_weights_derivative = \
          np.zeros(self.nn.hidden_to_output_weights.shape)
        self.hidden_input_derivative = \
          np.zeros(self.nn.hidden_layer.shape)
        self.hidden_weights_derivative = \
          np.zeros(self.nn.input_to_hidden_weights.shape)

    def backward_propagate(self, output, target):
        error_derivative = ce_derivative(output, target)
        self.output_input_derivative = \
          error_derivative * sigmoid_derivative(output)

        # Propagate to hidden to output weights
        self.output_weights_derivative.fill(1)
        self.output_weights_derivative *= self.nn.hidden_layer.T
        self.output_weights_derivative *= self.output_input_derivative

        # Propagate to hidden inputs
        sigmoid_derivative(
            self.nn.hidden_layer, out=self.hidden_input_derivative
        )
        self.hidden_input_derivative *= \
          self.nn.hidden_to_output_weights.T
        self.hidden_input_derivative *= self.output_input_derivative

        # Propagate to input to hidden weights.
        self.hidden_weights_derivative.fill(1)
        self.hidden_weights_derivative *= self.nn.input_layer.T
        self.hidden_weights_derivative *= self.hidden_input_derivative
