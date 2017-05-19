import tensorflow as tf

def build(prev_layer, layer_info, is_training):
    with tf.name_scope("maxpool"):
        return tf.layers.max_pooling2d(
            prev_layer,
            pool_size = 2,
            strides = 2
        )
