import tensorflow as tf

def build_dense(prev_layer, layer_info, is_training):
    activation_fn = get_activation_fn_by_name(
        activation_fn_name = layer_info["activation"],
        is_training = is_training,
    )

    return tf.layers.dense(
        prev_layer,
        layer_info["num_units"],
        activation = activation_fn,
        name = "initial_layer"
    )
