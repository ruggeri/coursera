import activations
import tensorflow as tf

def build(prev_layer, layer_info, is_training):
    activation_fn = activations.build(
        activation_fn_name = layer_info["activation"],
        is_training = is_training,
    )

    with tf.name_scope("dense"):
        return tf.layers.dense(
            prev_layer,
            layer_info["num_units"],
            activation = activation_fn
        )
