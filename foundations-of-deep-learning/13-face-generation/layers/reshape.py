import tensorflow as tf

def build(prev_layer, layer_info, is_training):
    return tf.reshape(
        prev_layer,
        layer_info["dimensions"]
    )
