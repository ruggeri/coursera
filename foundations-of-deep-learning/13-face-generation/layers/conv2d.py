import activations
import config.network
import tensorflow as tf

def build(prev_layer, layer_info, is_training):
    activation_fn, is_batch_normalized = activations.build(
        activation_fn_name = layer_info["activation"],
        is_training = is_training,
    )

    num_filters = layer_info.get(
        "num_filters",
        config.network.NUM_CONV_FILTERS
    )

    with tf.name_scope("conv2d"):
        return tf.layers.conv2d(
            prev_layer,
            filters = num_filters,
            kernel_size = config.network.CONV_KSIZE,
            strides = 1,
            padding = "SAME",
            activation = activation_fn,
            use_bias = not is_batch_normalized
        )
