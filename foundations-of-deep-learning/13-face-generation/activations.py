import config
import tensorflow as tf

def leaky_relu(input_t):
    with tf.name_scope("leaky_relu"):
        return tf.maximum(input_t, config.LEAKAGE * input_t)

def batch_normalized_leaky_relu(ipt, is_training):
    with tf.name_scope("bn_leaky_relu"):
        return leaky_relu(
            tf.layers.batch_normalization(ipt, training = is_training)
        )

def batch_normalized_relu(ipt, is_training):
    with tf.name_scope("bn_relu"):
        return tf.nn.relu(
            tf.layers.batch_normalization(ipt, training = is_training)
        )

# Also tells whether activation is batch normalized.
def build(activation_fn_name, is_training):
    if activation_fn_name == "leaky_relu":
        return (leaky_relu, False)
    elif activation_fn_name == "bn_leaky_relu":
        return (
            lambda ipt: batch_normalized_leaky_relu(ipt, is_training),
            True
        )
    elif activation_fn_name == "bn_relu":
        return (
            lambda ipt: batch_normalized_relu(ipt, is_training),
            True
        )
    elif activation_fn_name == None:
        return (lambda ipt: ipt, False)
    elif activation_fn_name == "tanh":
        return (tf.tanh, False)
    else:
        raise Exception(
            f"Unknown activation function {activation_fn_name}"
        )
