import config
import tensorflow as tf

KSIZE = 5
INITIAL_NUM_FILTERS = 64
FILTERS_MULTIPLIER = 2

def conv2d(prev_layer, layer_idx, training, keep_prob):
    num_filters = (
        INITIAL_NUM_FILTERS * (FILTERS_MULTIPLIER ** (layer_idx - 1))
    )

    layer = tf.layers.conv2d(
        prev_layer,
        filters = num_filters,
        kernel_size = (KSIZE, KSIZE),
        strides = (1, 1),
        padding = "SAME",
        activation = None,
        use_bias = False
    )

    layer = tf.nn.relu(
        tf.layers.batch_normalization(
            layer,
            training = training
        ),
    )

    layer = tf.layers.max_pooling2d(
        layer, pool_size = (2, 2), strides = (2, 2)
    )

    layer = tf.nn.dropout(layer, keep_prob = keep_prob)

    return layer

def dense(prev_layer, num_units, training, keep_prob):
    layer = tf.layers.dense(
        prev_layer,
        num_units,
        activation = None,
        use_bias = False
    )

    layer = tf.nn.relu(
        tf.layers.batch_normalization(
            layer,
            training = training
        ),
    )

    layer = tf.nn.dropout(layer, keep_prob = keep_prob)

    return layer

def build_logits(x, keep_prob, num_classes, training):
    with tf.name_scope("layer1"):
        layer1 = conv2d(
            x, 1, training = training, keep_prob = keep_prob
        )
    # => 16 16 64

    with tf.name_scope("layer2"):
        layer2 = conv2d(
            layer1, 2, training = training, keep_prob = keep_prob
        )
    # => 8 8 128

    with tf.name_scope("layer3"):
        layer3 = conv2d(
            layer2, 3, training = training, keep_prob = keep_prob
        )
    # => 4 4 256

    with tf.name_scope("layer4"):
        # I tried concatenating the features extracted in layer 2 as
        # was done in the LeCun paper, but that didn't seem to help.
        layer3_flattened = tf.contrib.layers.flatten(layer3)
        # => 4096

        layer4 = dense(
            layer3_flattened,
            1024,
            training = training,
            keep_prob = keep_prob
        )
    # => 1024

    with tf.name_scope("layer5"):
        layer5 = dense(
            layer4,
            256,
            training = training,
            keep_prob = keep_prob
        )
    # => 256

    with tf.name_scope("layer6"):
        layer6 = tf.layers.dense(
            layer5,
            num_classes,
            activation = None
        )

    return layer6

def build_cost_and_accuracy(logits, one_hot_y):
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = one_hot_y,
                logits = logits
            )
        )

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(
                tf.argmax(logits, axis = 1),
                tf.argmax(one_hot_y, axis = 1),
            ),
            tf.float32
        ))

    return (cost, accuracy)

def build_optimizer(cost, learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        return tf.train.AdamOptimizer(
            learning_rate = learning_rate
        ).minimize(cost)

from collections import namedtuple
Network = namedtuple("Network", [
    "x",
    "y",
    "keep_prob",
    "learning_rate",
    "training",
    "logits",
    "cost",
    "accuracy",
    "train_op",

    "summary",
])

def build_network(image_shape, num_classes):
    with tf.name_scope("placeholders"):
        x = tf.placeholder(tf.float32, (None, *image_shape), name = "x")
        tf.summary.image("x", x)
        y = tf.placeholder(tf.int32, (None), name = "y")
        tf.summary.histogram("y", y)
        keep_prob = tf.placeholder(tf.float32, (), name = "keep_prob")
        learning_rate = tf.placeholder(
            tf.float32, (), name = "learning_rate"
        )
        training = tf.placeholder(tf.bool, (), name = "training")

    one_hot_y = tf.one_hot(y, num_classes)

    logits = build_logits(x, keep_prob, num_classes, training)
    tf.summary.histogram("logits", logits)
    cost, accuracy = build_cost_and_accuracy(logits, one_hot_y)
    train_op = build_optimizer(cost, learning_rate)

    return Network(
        x = x,
        y = y,
        keep_prob = keep_prob,
        learning_rate = learning_rate,
        training = training,
        logits = logits,
        cost = cost,
        accuracy = accuracy,
        train_op = train_op,
        summary = tf.summary.merge_all()
    )

def restore(session):
    network = build_network(
        config.PROCESSED_IMAGE_SHAPE,
        config.NUM_CLASSES
    )

    saver = tf.train.Saver()
    saver.restore(session, "models/model.ckpt")

    return network
