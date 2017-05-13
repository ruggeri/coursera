import config
import numpy as np
import tensorflow as tf

def leaky_relu(input, name = None):
    return tf.maximum(
        input,
        config.LEAKAGE * input,
        name = name
    )

def xavier_stddev(fan_in, fan_out):
    return np.sqrt(2.0 / (fan_in + fan_out))
