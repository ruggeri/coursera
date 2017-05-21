import activations
import config.network
import tensorflow as tf

def build(prev_layer, layer_info, is_training):
    activation_fn, is_batch_normalized = activations.build(
        activation_fn_name = layer_info["activation"],
        is_training = is_training,
    )

    ksize = layer_info["ksize"]
    num_filters = layer_info["num_filters"]
    strides = layer_info["strides"]

    with tf.name_scope("conv2d_transpose"):
        return tf.layers.conv2d_transpose(
            prev_layer,
            filters = num_filters,
            kernel_size = ksize,
            strides = strides,
            padding = "SAME",
            activation = activation_fn,
            kernel_initializer = config.network.KERNEL_INITIALIZER,
            use_bias = not is_batch_normalized
        )
