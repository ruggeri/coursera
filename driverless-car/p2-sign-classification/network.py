import config
import tensorflow as tf

KSIZE = 5

def build_logits(x, keep_prob, num_classes, training):
    with tf.name_scope("layer1"):
        layer1 = tf.layers.conv2d(
            x,
            filters = 64,
            kernel_size = (KSIZE, KSIZE),
            strides = (1, 1),
            padding = "SAME",
            activation = None,
            use_bias = False
        )
        layer1 = tf.nn.relu(
            tf.layers.batch_normalization(
                layer1,
                training = training
            ),
        )
        layer1 = tf.layers.max_pooling2d(
            layer1, pool_size = (2, 2), strides = (2, 2)
        )
        layer1 = tf.nn.dropout(layer1, keep_prob = keep_prob)
    # => 16 16 64

    with tf.name_scope("layer2"):
        layer2 = tf.layers.conv2d(
            layer1,
            filters = 128,
            kernel_size = (KSIZE, KSIZE),
            strides = (1, 1),
            padding = "SAME",
            activation = None,
            use_bias = False
        )
        layer2 = tf.nn.relu(
            tf.layers.batch_normalization(
                layer2,
                training = training
            ),
        )
        layer2 = tf.layers.max_pooling2d(
            layer2, pool_size = (2, 2), strides = (2, 2)
        )
        layer2 = tf.nn.dropout(layer2, keep_prob = keep_prob)
    # => 8 8 128

    with tf.name_scope("layer3"):
        layer3 = tf.layers.conv2d(
            layer2,
            filters = 256,
            kernel_size = (5, 5),
            strides = (1, 1),
            padding = "SAME",
            activation = None,
            use_bias = False
        )
        layer3 = tf.nn.relu(
            tf.layers.batch_normalization(
                layer3,
                training = training
            ),
        )
        layer3 = tf.layers.max_pooling2d(
            layer3, pool_size = (2, 2), strides = (2, 2)
        )
        layer3 = tf.nn.dropout(layer3, keep_prob = keep_prob)
    # => 4 4 256

    with tf.name_scope("layer4"):
        layer1_flattened = tf.contrib.layers.flatten(layer1)
        layer2_flattened = tf.contrib.layers.flatten(layer2)
        layer3_flattened = tf.contrib.layers.flatten(layer3)

        #layer4 = tf.concat(
        #    [layer1_flattened, layer2_flattened, layer3_flattened],
        #    axis = 0
        #)
        flattened = layer3_flattened
        # => 16384 + 8192 + 4096 = 28672

        layer4 = tf.layers.dense(
            flattened,
            1024,
            activation = None,
            use_bias = False
        )
        layer4 = tf.nn.relu(
            tf.layers.batch_normalization(
                layer4,
                training = training
            ),
        )
        layer4 = tf.nn.dropout(layer4, keep_prob = keep_prob)
    # => 1024

    with tf.name_scope("layer5"):
        layer5 = tf.layers.dense(
            layer4,
            256,
            activation = None,
            use_bias = False
        )
        layer5 = tf.nn.relu(
            tf.layers.batch_normalization(
                layer5,
                training = training
            )
        )
        layer5 = tf.nn.dropout(layer5, keep_prob = keep_prob)
    # => 256

    with tf.name_scope("layer6"):
        layer6 = tf.layers.dense(
            layer5,
            num_classes,
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
        cost = cost,
        accuracy = accuracy,
        train_op = train_op,
        summary = tf.summary.merge_all()
    )
