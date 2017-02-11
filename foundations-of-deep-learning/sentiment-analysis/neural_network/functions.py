import numpy as np

def ce_derivative(output, target):
    return -target * (1/output) + (-(1-target) * -1/(1-output))

def cross_entropy(output, target):
    # This is the expected # of bits to encode the average token
    # if we draw tokens from an output stream where t% of tokens
    # are 1s and (1-t)% are 0s.
    return (-target*np.log(output)) + (-(1-target)*np.log(1-output))

def sigmoid(vec):
    np.exp(-vec, out=vec)
    vec += 1.0
    np.reciprocal(vec, out=vec)

def sigmoid_derivative(activation, out=None):
    if type(activation) == float:
        return activation * (1 - activation)

    out.fill(1)
    out -= activation
    out *= activation
