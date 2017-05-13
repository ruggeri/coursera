from collections import namedtuple
import config
import numpy as np
import re
import tensorflow as tf

Graph = namedtuple("Graph", [
    "class_label",
    # Generator input/output
    "z",
    "generator_x",
    # Discriminator input/output
    "discriminator_x",
    "authenticity_label",
    "discriminator_output",
    # Discriminator training
    "discriminator_loss",
    "discriminator_accuracy",
    "train_discriminator_op",
    # Generator training
    "generator_loss",
    "train_generator_op",
])

def leaky_relu(input, name):
    return tf.maximum(
        input,
        config.LEAKAGE * input,
        name = name
    )

def generator(z_dims, num_hidden_units, x_dims, one_hot_class_label):
    z = tf.placeholder(tf.float32, [None, z_dims], name = "z")
    with tf.variable_scope("generator_vars"):
        h1 = tf.layers.dense(
            inputs = tf.concat([one_hot_class_label, z], axis = 1),
            units = num_hidden_units,
            activation = None,
        )
        h1 = leaky_relu(h1, "h1")
        generator_x = tf.layers.dense(
            inputs = h1,
            units = x_dims,
            activation = tf.nn.tanh,
            name = "generator_x"
        )

    return (z, generator_x)

def glorot_bound(fan_in, fan_out):
    return np.sqrt(6.0/(fan_in + fan_out))

def discriminator(
        num_classes,
        x_dims,
        num_hidden_units,
        one_hot_class_label,
        generator_x,
        discriminator_x):
    with tf.variable_scope("discriminator_vars"):
        bound1 = glorot_bound(num_classes + x_dims, num_hidden_units)
        hidden_weights1 = tf.Variable(
            tf.random_uniform(
                [num_classes + x_dims, num_hidden_units],
                minval = -bound1,
                maxval = +bound1
            ),
            name = "hidden_weights1"
        )
        hidden_biases1 = tf.Variable(
            0.1 * np.ones(num_hidden_units, dtype = np.float32),
            name = "hidden_biases1"
        )
        output_bound = glorot_bound(num_hidden_units, 1)
        output_weights = tf.Variable(
            tf.random_uniform(
                [num_hidden_units, 1],
                minval = -output_bound,
                maxval = +output_bound,
            ),
            name = "output_weights"
        )
        output_biases = tf.Variable(
            0.1,
            name = "output_biases"
        )

    def helper(x):
        with tf.name_scope("h1"):
            h1_input = tf.concat(
                [one_hot_class_label, x],
                axis = 1
            )
            h1 = tf.matmul(h1_input, hidden_weights1) + hidden_biases1
            h1 = leaky_relu(h1, name = "output")
        with tf.name_scope("output"):
            output_logits = tf.reshape(
                tf.matmul(h1, output_weights) + output_biases,
                (-1,),
                name = "logits"
            )
            output = tf.nn.sigmoid(output_logits, name = "output")

        return (output_logits, output)

    with tf.name_scope("discriminator"):
        discriminator_logits, discriminator_output = helper(
            discriminator_x
        )
    with tf.name_scope("generator_training_discriminator"):
        generator_logits, _ = helper(
            generator_x
        )

    return (
        discriminator_logits,
        discriminator_output,
        generator_logits,
    )

def discriminator_trainer(
        authenticity_label,
        discriminator_logits,
        discriminator_output):
    with tf.name_scope("discriminator_training"):
        d_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(
                tf.cast(tf.round(discriminator_output), tf.int64),
                authenticity_label
            ), tf.float32),
            name = "accuracy",
        )

        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tf.cast(authenticity_label, tf.float32),
                logits = discriminator_logits
            ),
            name = "loss",
        )

        discriminator_vars = [
            var for var in tf.trainable_variables()
            if re.match(r"discriminator_vars", var.name)
        ]
        d_train = tf.train.AdamOptimizer().minimize(
            d_loss,
            var_list = discriminator_vars
        )

    return (d_accuracy, d_loss, d_train)

def generator_trainer(generator_logits):
    with tf.name_scope("generator_training"):
        # NB: Rather than explicitly try to make the discriminator
        # maximize, we minimize the "wrong" loss, because the
        # gradients are stronger to learn from.
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tf.ones_like(generator_logits),
                logits = generator_logits
            ),
            name = "loss",
        )
        # We only want to update parameters of the generator, even
        # though the discriminator's estimates are part of the
        # training process.
        generator_vars = [
            var for var in tf.trainable_variables()
            if re.match(r"generator_vars", var.name)
        ]
        train_generator_op = tf.train.AdamOptimizer().minimize(
            generator_loss, var_list = generator_vars
        )

    return (generator_loss, train_generator_op)

def graph(
        num_classes,
        x_dims,
        z_dims,
        num_generator_hidden_units,
        num_discriminator_hidden_units):
    class_label = tf.placeholder(tf.int64, [None], name = "class_label")
    one_hot_class_label = tf.one_hot(class_label, num_classes)

    # Generator
    z, generator_x = generator(
        z_dims = z_dims,
        num_hidden_units = num_generator_hidden_units,
        x_dims = x_dims,
        one_hot_class_label = one_hot_class_label,
    )

    # Discriminator inputs
    discriminator_x = tf.placeholder(
        tf.float32, [None, x_dims], name = "discriminator_x"
    )
    authenticity_label = tf.placeholder(
        tf.int64, [None], name = "authenticity_label"
    )
    discriminator_logits, discriminator_output, generator_logits = discriminator(
        num_classes = num_classes,
        x_dims = x_dims,
        num_hidden_units = num_discriminator_hidden_units,
        one_hot_class_label = one_hot_class_label,
        discriminator_x = discriminator_x,
        generator_x = generator_x,
    )

    d_accuracy, d_loss, d_train = discriminator_trainer(
        authenticity_label = authenticity_label,
        discriminator_logits = discriminator_logits,
        discriminator_output = discriminator_output,
    )

    g_loss, g_train = generator_trainer(
        generator_logits
    )

    return Graph(
        class_label = class_label,
        # Generator input/output
        z = z,
        generator_x = generator_x,
        # Discriminator input/output
        discriminator_x = discriminator_x,
        authenticity_label = authenticity_label,
        discriminator_output = discriminator_output,
        # Discriminator training
        discriminator_accuracy = d_accuracy,
        discriminator_loss = d_loss,
        train_discriminator_op = d_train,
        # Generator training
        generator_loss = g_loss,
        train_generator_op = g_train,
    )
