import tensorflow as tf

def build(prev_layer, layer_info, is_training):
    return tf.image.resize_nearest_neighbor(
        prev_layer,
        layer_info["size"],
    )
