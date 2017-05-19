import config
import tensorflow as tf

def leaky_relu(input_t):
    with tf.name_scope("leaky_relu"):
        return tf.maximum(input_t, config.LEAKAGE * input_t)

def batch_normalized_leaky_relu(ipt, is_training):
    with tf.name_scope("bn_leaky_relu"):
        return leaky_relu(
            tf.layers.batch_normalization(ipt, training = is_training)
        )

def batch_normalized_relu(ipt, is_training):
    with tf.name_scope("bn_relu"):
        return tf.nn.relu(
            tf.layers.batch_normalization(ipt, training = is_training)
        )

def get_activation_fn_by_name(activation_fn_name, is_training):
    if activation_fn_name == "leaky_relu":
        return leaky_relu
    elif activation_fn_name == "bn_leaky_relu":
        return lambda ipt: batch_normalized_leaky_relu(ipt, is_training)
    elif layer_info["activation"] == "bn_relu":
        return lambda ipt: batch_normalized_relu(ipt, is_training)
    elif layer_info["activation"] == "tanh":
        return tf.tanh
    else:
        raise Exception(
            f"Unknown activation function {activation_fn_name}"
        )

def build_layer(prev_layer, layer_info, is_training):
    if layer_info["type"] == "conv2d":
        activation_fn = get_activation_fn_by_name(
            activation_fn_name = activation_fn,
            is_training = is_training,
        )

        num_filters = layer_info.get(
            "num_filters",
            config.NUM_CONV_FILTERS
        )

        return tf.layers.conv2d(
            prev_layer,
            filters = num_filters,
            kernel_size = config.CONV_KSIZE,
            strides = 1,
            padding = "SAME",
            activation = activation_fn
        )
    elif layer_info["type"] == "maxpool":
        return tf.layers.max_pooling2d(
            prev_layer,
            pool_size = 2,
            strides = 2
        )
    elif layer_info["type"] == "resize":
        return tf.image.resize_nearest_neighbor(
            prev_layer,
            layer_info["size"],
        )
    else:
        raise Exception("unknown layer type")
