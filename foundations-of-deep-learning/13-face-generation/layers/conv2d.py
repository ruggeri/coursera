import tensorflow as tf

def build(prev_layer, layer_info, is_training):
    activation_fn = get_activation_fn_by_name(
        activation_fn_name = layer_info["activation"],
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
