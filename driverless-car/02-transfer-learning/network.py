import alexnet
from collections import namedtuple
import config
import tensorflow as tf

Network = namedtuple("Network", [
    "x",
    "y",
    "bottleneck_in",
    "bottleneck_out",
    "accuracy",
    "loss",
    "train_op",
    "name",
])

def build_alexnet_network(num_classes):
    x = tf.placeholder(
        tf.float32, (None, *config.INPUT_IMAGE_SIZE), name = "x"
    )
    resized_x = tf.image.resize_images(
        x,
        config.ALEX_NET_IMG_DIM
    )
    bottleneck_in = tf.placeholder(
        tf.float32, (None, 4096), name = "bottleneck_in"
    )
    y = tf.placeholder(tf.int64, (None), name = "y")
    one_hot_y = tf.one_hot(y, num_classes)

    bottleneck_out = alexnet.AlexNet(resized_x, feature_extract=True)

    logits = tf.layers.dense(
        bottleneck_in, num_classes, activation = None
    )
    probs = tf.nn.softmax(logits)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(
            tf.argmax(logits, axis = 1),
            tf.argmax(one_hot_y, axis = 1)
        ), tf.float32
    ))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = one_hot_y,
        logits = logits
    ))
    train_op = tf.train.AdamOptimizer(
        learning_rate = config.LEARNING_RATE
    ).minimize(loss)

    return Network(
        x = x,
        y = y,
        accuracy = accuracy,
        bottleneck_out = bottleneck_out,
        bottleneck_in = bottleneck_in,
        loss = loss,
        train_op = train_op,
        name = "alexnet",
    )
