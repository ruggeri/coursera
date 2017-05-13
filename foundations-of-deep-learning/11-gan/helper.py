import config
import numpy as np
import tensorflow as tf

def leaky_relu(input, name = None):
    return tf.maximum(
        input,
        config.LEAKAGE * input,
        name = name
    )

def glorot_bound(fan_in, fan_out):
    return np.sqrt(6.0/(fan_in + fan_out))
