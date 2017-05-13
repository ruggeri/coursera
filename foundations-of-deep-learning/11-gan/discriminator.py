from collections import namedtuple
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

Discriminator = namedtuple("Discriminator", [
    "class_label",
    "x",
    "authenticity_label",
    "prediction",
    "loss",
    "accuracy",
    "train_op",
])

def parameters(network_configuration):
    nc = network_configuration

    with tf.variable_scope("discriminator_vars") as variable_scope:
        num_inputs = nc.num_classes + nc.num_x_dims
        num_hidden_units = nc.num_discriminator_hidden_units
        hidden_stddev1 = helper.xavier_stddev(
            num_inputs,
            num_hidden_units,
        )
        hidden_weights1 = tf.Variable(
            tf.truncated_normal(
                [num_inputs, num_hidden_units],
                stddev = hidden_stddev1,
            ),
            name = "hidden_weights1"
        )
        hidden_biases1 = tf.Variable(
            0.1 * np.ones(num_hidden_units, dtype = np.float32),
            name = "hidden_biases1"
        )

        output_stddev = helper.xavier_stddev(num_hidden_units, 1)
        output_weights = tf.Variable(
            tf.truncated_normal(
                [num_hidden_units, 1],
                stddev = output_stddev
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

def apply_parameters(one_hot_class_label, x, discriminator_parameters):
    dc = discriminator_parameters

    # Perform fc hidden layer
    with tf.name_scope("hidden"):
        h1_input = tf.concat(
            [one_hot_class_label, x],
            axis = 1
        )
        h1 = tf.matmul(
            h1_input, dc.hidden_weights1
        ) + dc.hidden_biases1
        h1_output = helper.leaky_relu(h1)

    # Perform fc output layer
    with tf.name_scope("prediction"):
        prediction_logits = tf.reshape(
            tf.add(
                tf.matmul(h1_output, dc.output_weights),
                dc.output_biases
            ),
            (-1,),
            name = "prediction_logits"
        )
        prediction = tf.nn.sigmoid(
            prediction_logits, name = "prediction"
        )

    return (prediction_logits, prediction)

def discriminator(
        network_configuration,
        discriminator_parameters):
    nc, dp = network_configuration, discriminator_parameters

    with tf.name_scope("discriminator"):
        # Placeholders
        with tf.name_scope("placeholders"):
            class_label = tf.placeholder(
                tf.int64, [None], name = "class_label"
            )
            one_hot_class_label = tf.one_hot(
                class_label, nc.num_classes
            )
            x = tf.placeholder(
                tf.float32, [None, nc.num_x_dims], name = "x"
            )
            authenticity_label = tf.placeholder(
                tf.int64, [None], name = "authenticity_label"
            )

        # Predictions
        prediction_logits, prediction = apply_parameters(
            one_hot_class_label = one_hot_class_label,
            x = x,
            discriminator_parameters = discriminator_parameters,
        )

        # Evaluation and training operations
        accuracy, loss, train_op = trainer.build(
            prediction_logits = prediction_logits,
            prediction = prediction,
            authenticity_label = authenticity_label,
            variable_scope = dp.variable_scope,
        )

    return Discriminator(
        class_label = class_label,
        x = x,
        authenticity_label = authenticity_label,
        prediction = prediction,
        loss = loss,
        accuracy = accuracy,
        train_op = train_op,
    )
