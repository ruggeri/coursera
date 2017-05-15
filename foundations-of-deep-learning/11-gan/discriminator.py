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

Discriminator = namedtuple("Discriminator", [
    "parameters",
    "all_x",
    "all_class_label",
    "all_authenticity_label",
    "loss",
    "accuracy",
    "train_op",
])

def parameters(network_configuration):
    nc = network_configuration

    with tf.variable_scope("discriminator_vars") as variable_scope:
        hidden_weights1 = tf.Variable(
            helper.glorot_uniform_initializer(
                nc.num_classes + nc.num_x_dims,
                nc.num_discriminator_hidden_units
            ),
            name = "hidden_weights1"
        )
        hidden_biases1 = tf.Variable(
            config.DEFAULT_BIAS * np.ones(
                nc.num_discriminator_hidden_units, dtype = np.float32
            ),
            name = "hidden_biases1"
        )

        output_weights = tf.Variable(
            helper.glorot_uniform_initializer(
                nc.num_discriminator_hidden_units, 1
            ),
            name = "output_weights"
        )
        output_biases = tf.Variable(
            config.DEFAULT_BIAS,
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
            all_class_label = tf.placeholder(
                tf.int64, [None], name = "all_class_label"
            )
            one_hot_all_class_label = tf.one_hot(
                all_class_label, nc.num_classes
            )

            all_x = tf.placeholder(
                tf.float32, [None, nc.num_x_dims], name = "all_x"
            )

            all_authenticity_label = tf.placeholder(
                tf.int64, [None], name = "all_authenticity_label"
            )

        # Predictions
        with tf.name_scope("predictions"):
            all_prediction_logits, all_prediction = apply_parameters(
                one_hot_class_label = one_hot_all_class_label,
                x = all_x,
                discriminator_parameters = discriminator_parameters,
            )

        # Evaluation and training operations
        accuracy, loss, train_op = trainer.build_for_discriminator(
            all_prediction_logits = all_prediction_logits,
            all_prediction = all_prediction,
            all_authenticity_label = all_authenticity_label,
            variable_scope = dp.variable_scope,
        )

    return Discriminator(
        parameters = discriminator_parameters,
        all_x = all_x,
        all_class_label = all_class_label,
        all_authenticity_label = all_authenticity_label,
        loss = loss,
        accuracy = accuracy,
        train_op = train_op,
    )
