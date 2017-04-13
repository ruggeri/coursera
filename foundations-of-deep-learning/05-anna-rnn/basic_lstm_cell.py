import numpy as np
import tensorflow as tf

class BasicLSTMCell:
    def __init__(self, num_lstm_units, num_input_units):
        self.num_lstm_units = num_lstm_units
        self.num_input_units = num_input_units

        total_num_units = num_input_units + num_lstm_units

        def weights_and_biases(name, subname = "multiplier"):
            weights = glorot_weights(
                f"lstm_cell/{name}_{subname}/weights",
                total_num_units,
                num_lstm_units,
            )
            biases = tf.Variable(
                tf.zeros([num_lstm_units]),
                name = f"lstm_cell/{name}_{subname}/biases"
            )
            return (weights, biases)

        (self.forget_multiplier_weights, self.forget_multiplier_biases) = (
            weights_and_biases("forget")
        )
        # They find it helpful to bias the forgetting to 1.0 initially.
        self.forget_multiplier_biases += 1

        (self.write_multiplier_weights, self.write_multiplier_biases) = (
            weights_and_biases("write")
        )
        (self.write_value_weights, self.write_value_biases) = (
            weights_and_biases("write", "value")
        )

        (self.read_multiplier_weights, self.read_multiplier_biases) = (
            weights_and_biases("read")
        )

    def __call__(self, ipt, prev_state_and_prev_output):
        prev_state, prev_output = prev_state_and_prev_output
        concat_input = tf.concat([prev_output, ipt], 1)

        def calc_value(weights, biases, activation = tf.sigmoid):
            return activation(
                tf.matmul(concat_input, weights) + biases
            )

        forget_multiplier = calc_value(
            self.forget_multiplier_weights,
            self.forget_multiplier_biases
        )
        write_multiplier = calc_value(
            self.write_multiplier_weights,
            self.write_multiplier_biases
        )
        write_value = calc_value(
            self.write_value_weights,
            self.write_value_biases,
            activation = tf.tanh
        )
        read_multiplier = calc_value(
            self.read_multiplier_weights,
            self.read_multiplier_biases
        )

        next_state = (
            (forget_multiplier * prev_state) +
            (write_multiplier * write_value)
        )
        next_output = tf.tanh(read_multiplier * next_state)

        return (next_output, (next_state, next_output))

def glorot_weights(name, num_input_units, num_output_units):
    limit = np.sqrt(6 / (num_input_units + num_output_units))

    return tf.Variable(
        tf.random_uniform(
            [num_input_units, num_output_units],
            minval = -limit,
            maxval = limit,
        ),
        name = name
    )
