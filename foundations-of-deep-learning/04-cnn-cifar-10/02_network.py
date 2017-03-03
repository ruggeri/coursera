import numpy as np
import tensorflow as tf

WEIGHT_STDDEV = 0.1
def conv2d_maxpool(
        input_tensor,
        num_output_channels,
        conv_size,
        conv_stride,
        pool_size,
        pool_stride):
    # Build weights
    num_input_layers = int(input_tensor.get_shape()[3])
    conv_weights_size = (
        conv_size,
        conv_size,
        num_input_layers,
        num_output_channels
    )
    conv_weights = tf.truncated_normal(
        conv_weights_size, stddev=WEIGHT_STDDEV
    )
    conv_weights = tf.Variable(conv_weights, "conv_weights")

    # Apply convolution
    conv_layer = tf.nn.conv2d(
        input_tensor,
        conv_weights,
        [1, conv_stride, conv_stride, 1],
        "SAME"
    )
    bias = np.zeros(num_output_channels)
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    conv_layer = tf.nn.relu(conv_layer)

    # Do max pooling.
    pool_layer = tf.nn.max_pool(
        conv_layer,
        [1, pool_size, pool_size, 1],
        [1, pool_stride, pool_stride, 1],
        "SAME"
    )

    return pool_layer

# Flatten 3D images (includes channels dimension) to 1D.
def flatten(input_tensor):
    height = int(input_tensor.get_shape()[1])
    width = int(input_tensor.get_shape()[2])
    num_channels = int(input_tensor.get_shape()[3])
    num_features = (height * width * num_channels)
    return tf.reshape(input_tensor, [-1, num_features])

# Simple fully connected layer.
def fully_conn(input_tensor, num_outputs):
    num_input_units = int(input_tensor.get_shape()[1])
    weights = tf.truncated_normal(
        (num_input_units, num_outputs),
        stddev = WEIGHT_STDDEV
    )
    weights = tf.Variable(weights)
    bias = tf.Variable(tf.zeros(num_outputs))
    layer = tf.add(tf.matmul(input_tensor, weights), bias)
    layer = tf.nn.relu(layer)

    return layer

def output(input_tensor, num_outputs):
    num_input_units = int(input_tensor.get_shape()[1])
    weights = tf.truncated_normal(
        (num_input_units, num_outputs),
        stddev = WEIGHT_STDDEV
    )
    weights = tf.Variable(weights)
    bias = tf.Variable(tf.zeros(num_outputs))
    logits = tf.add(tf.matmul(input_tensor, weights), bias)

    # No need for RELU for logits, which are allowed to be
    # negative. At least I don't see any problem with them being
    # negative...

    return logits

def conv_net(x, keep_prob):
    # Two layers of convolution and max pooling. Each runs a 2x2
    # filter producing 16 channels, then does 2x2 max pooling.
    conv_layer1 = conv2d_maxpool(
        input_tensor = x,
        num_output_channels = 16,
        conv_size = 2,
        conv_stride = 1,
        pool_size = 2,
        pool_stride = 2
    )
    conv_layer2 = conv2d_maxpool(
        input_tensor = conv_layer1,
        num_output_channels = 16,
        conv_size = 2,
        conv_stride = 1,
        pool_size = 2,
        pool_stride = 2
    )

    # Simple flattening of 2D images to 1D. Apply dropout here.
    flattened_layer = flatten(conv_layer2)
    flattened_layer = tf.nn.dropout(flattened_layer, keep_prob)

    # Simple 512 unit layer connected to output layer.
    full_conn_layer1 = fully_conn(flattened_layer, 512)
    output_layer = output(full_conn_layer1, 10)

    return output_layer

def setup_training():
    # Inputs
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Model
    logits = conv_net(x, keep_prob)

    # Name logits tensor, so that is can be loaded from disk after
    # training.
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_pred, tf.float32), name='accuracy'
    )

if __name__ == "__main__":
    setup_training()
