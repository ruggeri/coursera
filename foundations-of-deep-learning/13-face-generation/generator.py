import config
import helper
import tensorflow as tf

def generator(fake_z, num_out_channels, is_training, reuse):
    with tf.variable_scope("generator", reuse = reuse):
        prev_layer = fake_z

        # Start an initial noise image.
        num_pixels = (
            config.INITIAL_SIZE[0]
            * config.INITIAL_SIZE[1]
            * num_out_channels
        )
        prev_layer = tf.layers.dense(
            z,
            num_pixels,
            activation = tf.tanh,
            name = "initial_layer"
        )

        for layer_info in config.GENERATOR_LAYERS:
            prev_layer = helper.build_layer(
                prev_layer,
                layer_info,
                is_training = is_training
            )

    return prev_layer
