import numpy as np
from .functions import sigmoid

class FFNeuralNetwork:
    def __init__(self, num_input_units, num_hidden_units):
        self.input_layer = np.zeros((num_input_units, 1))
        self.input_to_hidden_weights = np.random.normal(
            0.0,
            num_hidden_units ** -0.5,
            size=(num_hidden_units, num_input_units)
        )
        self.hidden_bias = np.random.normal(
            0.0,
            num_hidden_units ** -0.5,
            size=(num_hidden_units, 1)
        )
        self.hidden_layer = np.zeros((num_hidden_units, 1))
        self.hidden_to_output_weights = \
          np.random.normal(size=(1, num_hidden_units))
        self.output_bias = np.random.normal()

    def run(self, input_v):
        self.input_layer[:] = input_v

        # Compute hidden layer
        self.input_to_hidden_weights.dot(self.input_layer,
                                         out=self.hidden_layer)
        self.hidden_layer += self.hidden_bias
        sigmoid(self.hidden_layer)

        # Compute output layer
        output = self.hidden_to_output_weights.dot(self.hidden_layer)
        output += self.output_bias
        sigmoid(output)

        return float(output)
