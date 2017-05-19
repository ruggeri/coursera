import tensorflow as tf

def build(prev_layer, layer_info, is_training):
    dim1, dim2, dim3 = prev_layer.get_shape()[1:]
    return tf.reshape(
        prev_layer,
        (-1, dim1.value * dim2.value * dim3.value)
    )
