import batch_normalization as bn
from collections import namedtuple
import config
import tensorflow as tf

def fc_layer(prev_layer, num_units, is_training):
    with tf.name_scope("fully_connected"):
        z = tf.layers.dense(
            inputs = prev_layer,
            units = num_units,
            activation = None
        )

        if config.USE_BATCH_NORMALIZATION:
            z = bn.batch_normalize(z, is_training)

        return tf.nn.relu(z)

def conv_layer(prev_layer, num_filters, perform_striding, is_training):
    if perform_striding:
        strides = (2, 2)
    else:
        strides = (1, 1)

    with tf.name_scope("convolution"):
        z = tf.layers.conv2d(
            inputs = prev_layer,
            filters = num_filters,
            kernel_size = config.KERNEL_SIZE,
            strides = strides,
            padding = "SAME",
        )

        if config.USE_BATCH_NORMALIZATION:
            z = bn.batch_normalize(z, is_training)

        return tf.nn.relu(z)

Network = namedtuple("Network", [
    "input_image",
    "one_hot_digit_label",
    "is_training",
    "loss",
    "train_op",
    "accuracy",
])

NUM_CLASSES = 10
def network():
    input_image = tf.placeholder(
        tf.float32,
        [None, 28, 28, 1],
        name = "input_image"
    )
    one_hot_digit_label = tf.placeholder(
        tf.float32,
        [None, NUM_CLASSES],
        name = "one_hot_digit_label"
    )
    is_training = tf.placeholder(tf.bool, name = "is_training")

    prev_layer = input_image
    for layer_idx in range(1, config.NUM_CONVOLUTION_LAYERS + 1):
        # This is silly, but the point is just to make a really deep
        # convolutional network.
        num_filters = 4 * (layer_idx + 1)
        prev_layer = conv_layer(
            prev_layer,
            num_filters = num_filters,
            perform_striding = layer_idx % 2 == 0,
            is_training = is_training
        )

    flattened_layer = tf.contrib.layers.flatten(
        inputs = prev_layer,
    )

    prev_layer = fc_layer(
        flattened_layer,
        config.NUM_HIDDEN_UNITS,
        is_training
    )

    prediction_logits = tf.layers.dense(
        prev_layer,
        units = NUM_CLASSES,
        activation = None,
        name = "prediction_logits",
    )

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels = one_hot_digit_label,
            logits = prediction_logits,
        ),
        name = "loss",
    )

    train_op = tf.train.AdamOptimizer().minimize(loss)

    with tf.name_scope("accuracy"):
        is_prediction_correct = tf.equal(
            tf.argmax(prediction_logits, axis = 1),
            tf.argmax(one_hot_digit_label, axis = 1),
            name = "is_prediction_correct"
        )
        accuracy = tf.reduce_mean(
            tf.cast(is_prediction_correct, tf.float32),
            name = "accuracy"
        )

    return Network(
        input_image = input_image,
        one_hot_digit_label = one_hot_digit_label,
        is_training = is_training,
        loss = loss,
        train_op = train_op,
        accuracy = accuracy,
    )
