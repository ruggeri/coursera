from collections import namedtuple
import discriminator
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
    "class_label",
    "z",
    "generated_x",
    "loss",
    "train_op",
])

def parameters(network_configuration):
    nc = network_configuration

    with tf.variable_scope("generator_vars") as variable_scope:
        # Hidden layer
        num_inputs = nc.num_classes + nc.num_z_dims
        num_hidden_units = nc.num_generator_hidden_units
        hidden_bound1 = helper.glorot_bound(
            num_inputs,
            num_hidden_units,
        )
        hidden_weights1 = tf.Variable(
            tf.random_uniform(
                [num_inputs, num_hidden_units],
                minval = -hidden_bound1,
                maxval = +hidden_bound1,
            ),
            name = "hidden_weights1"
        )
        hidden_biases1 = tf.Variable(
            0.1 * np.ones(num_hidden_units, dtype = np.float32),
            name = "hidden_biases1"
        )

        # Output layer
        output_bound = helper.glorot_bound(num_hidden_units, 1)
        output_weights = tf.Variable(
            tf.random_uniform(
                [num_hidden_units, nc.num_x_dims],
                minval = -output_bound,
                maxval = +output_bound,
            ),
            name = "output_weights"
        )
        output_biases = tf.Variable(
            0.1,
            name = "output_biases"
        )

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

    # Perform fc output layer
    with tf.name_scope("generated_x"):
        generated_x = tf.matmul(
            h1_output, gp.output_weights
        ) + gp.output_biases
        generated_x = tf.nn.tanh(generated_x)

    return generated_x

def generator(
        network_configuration,
        generator_parameters,
        discriminator_parameters):
    nc, gp = network_configuration, generator_parameters

    with tf.name_scope("generator"):
        # Placeholders
        class_label = tf.placeholder(
            tf.int64, [None], name = "class_label"
        )
        one_hot_class_label = tf.one_hot(class_label, nc.num_classes)
        z = tf.placeholder(
            tf.float32, [None, nc.num_z_dims], name = "z"
        )

        generated_x = apply_parameters(
            one_hot_class_label = one_hot_class_label,
            z = z,
            generator_parameters = generator_parameters,
        )

        prediction_logits, prediction = discriminator.apply_parameters(
            one_hot_class_label = one_hot_class_label,
            x = generated_x,
            discriminator_parameters = discriminator_parameters,
        )

        # NB: Rather than explicitly try to make the discriminator
        # maximize, we minimize the "wrong" loss, because the
        # gradients are stronger to learn from.
        _, loss, train_op = trainer.build(
            prediction_logits = prediction_logits,
            prediction = prediction,
            authenticity_label = tf.ones_like(prediction_logits),
            variable_scope = gp.variable_scope
        )

    return Generator(
        class_label = class_label,
        z = z,
        generated_x = generated_x,
        loss = loss,
        train_op = train_op
    )
