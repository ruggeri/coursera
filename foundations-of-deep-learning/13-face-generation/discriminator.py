import config
import helper
import tensorflow as tf

def discriminator(images, reuse):
    with tf.variable_scope("discriminator", reuse = reuse):
        prev_layer = images
        for layer_info in config.DISCRIMINATOR_LAYERS:
            helper.build_layer(
                prev_layer,
                layer_info,
                # The discriminator is *only* used in training mode.
                is_training = True
            )

        _, dim1, dim2, dim3 = prev_layer.get_shape()
        prev_layer = tf.reshape(
            prev_layer,
            (-1, dim1.value * dim2.value * dim3.value)
        )

        # We'll just return the logits
        prev_layer = tf.layers.dense(prev_layer, 1, activation = None)

    return prev_layer
