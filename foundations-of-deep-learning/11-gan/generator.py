from collections import namedtuple
import config
import helper
import numpy as np
import tensorflow as tf
import trainer

Parameters = namedtuple("Parameters", [
    "hidden_weights1",
    "hidden_biases1",
    "output_weights",
    "output_biases",
    "variable_scope",
])

Generator = namedtuple("Generator", [
    "parameters",
    "class_label",
    "z",
    "generated_x",
    "loss",
    "train_op",
    "summary",
])

SUMMARY_KEY = "GENERATOR_SUMMARIES"

def summary_histogram(tensor, name = None):
    tf.summary.histogram(
        name or tensor.name,
        tensor,
        collections = [SUMMARY_KEY]
    )

def summary_scalar(tensor, name = None):
    tf.summary.scalar(
        name or tensor.name,
        tensor,
        collections = [SUMMARY_KEY]
    )

def parameters(network_configuration):
    nc = network_configuration

    with tf.variable_scope("generator_vars") as variable_scope:
        # Hidden layer
        hidden_weights1 = tf.Variable(
            helper.glorot_uniform_initializer(
                nc.num_classes + nc.num_z_dims,
                nc.num_generator_hidden_units,
            ),
            name = "hidden_weights1"
        )
        summary_histogram(hidden_weights1)

        hidden_biases1 = tf.Variable(
            config.DEFAULT_BIAS * np.ones(
                nc.num_generator_hidden_units, dtype = np.float32
            ),
            name = "hidden_biases1"
        )
        summary_histogram(hidden_biases1)

        # Output layer
        output_weights = tf.Variable(
            helper.glorot_uniform_initializer(
                nc.num_generator_hidden_units,
                nc.num_x_dims
            ),
            name = "output_weights"
        )
        summary_histogram(output_weights)

        output_biases = tf.Variable(
            config.DEFAULT_BIAS * np.ones(nc.num_x_dims),
            name = "output_biases",
            dtype = tf.float32
        )
        summary_histogram(output_biases)

    return Parameters(
        hidden_weights1 = hidden_weights1,
        hidden_biases1 = hidden_biases1,
        output_weights = output_weights,
        output_biases = output_biases,
        variable_scope = variable_scope,
    )

def apply_parameters(one_hot_class_label, z, generator_parameters):
    gp = generator_parameters

    # Perform fc hidden layer
    with tf.name_scope("hidden"):
        h1_input = tf.concat(
            [one_hot_class_label, z],
            axis = 1
        )
        h1 = tf.matmul(
            h1_input, gp.hidden_weights1
        ) + gp.hidden_biases1
        h1_output = helper.leaky_relu(h1)
        summary_histogram(h1_output, name = "h1_output")

    # Perform fc output layer
    with tf.name_scope("generated_x"):
        generated_x = tf.matmul(
            h1_output, gp.output_weights
        ) + gp.output_biases
        generated_x = tf.nn.tanh(generated_x)
        summary_histogram(generated_x, name = "generated_x")

    return generated_x

def generator(
        network_configuration,
        generator_parameters,
        discriminator_parameters):
    nc, gp = network_configuration, generator_parameters

    with tf.name_scope("generator"):
        # Placeholders
        with tf.name_scope("placeholders"):
            class_label = tf.placeholder(
                tf.int64, [None], name = "class_label"
            )
            summary_histogram(class_label)

            one_hot_class_label = tf.one_hot(
                class_label, nc.num_classes
            )

            z = tf.placeholder(
                tf.float32, [None, nc.num_z_dims], name = "z"
            )
            summary_histogram(z)

        generated_x = apply_parameters(
            one_hot_class_label = one_hot_class_label,
            z = z,
            generator_parameters = generator_parameters,
        )

        loss, train_op = trainer.build_for_generator(
            one_hot_class_label = one_hot_class_label,
            generated_x = generated_x,
            discriminator_parameters = discriminator_parameters,
            variable_scope = gp.variable_scope
        )
        summary_scalar(loss)

    return Generator(
        parameters = generator_parameters,
        class_label = class_label,
        z = z,
        generated_x = generated_x,
        loss = loss,
        train_op = train_op,
        summary = tf.summary.merge_all(SUMMARY_KEY)
    )
