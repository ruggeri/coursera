from collections import namedtuple
import config
import math
import numpy as np
import tensorflow as tf

Graph = namedtuple("Graph", [
    "inputs",
    "labels",
    "keep_probability",
    "avg_cost",
    "accuracy",
    "optimization_op",
    "summaries",
])
def build_graph(num_code_units, num_labels):
    inputs = tf.placeholder(tf.float32, [None, num_code_units])
    labels = tf.placeholder(tf.int32, [None, num_labels])
    keep_probability = tf.placeholder(tf.float32)

    weights1 = tf.Variable(
        tf.truncated_normal(
            [num_code_units, config.NUM_HIDDEN_UNITS],
            stddev = 1 / math.sqrt(num_code_units)
        )
    )
    biases1 = tf.Variable(
        tf.zeros([config.NUM_HIDDEN_UNITS])
    )
    fc1 = tf.nn.relu(tf.matmul(inputs, weights1) + biases1)
    fc1 = tf.nn.dropout(fc1, keep_prob = keep_probability)

    weights2 = tf.Variable(
        tf.truncated_normal(
            [config.NUM_HIDDEN_UNITS, num_labels],
            stddev = 1 / math.sqrt(config.NUM_HIDDEN_UNITS)
        )
    )
    biases2 = tf.Variable(
        tf.zeros([num_labels])
    )
    logits = tf.matmul(fc1, weights2) + biases2

    avg_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels = labels, logits = logits
        )
    )
    optimization_op = tf.train.AdamOptimizer().minimize(avg_cost)

    estimated_probabilities = tf.nn.softmax(logits)
    predictions = tf.argmax(estimated_probabilities, 1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, tf.argmax(labels, 1)), tf.float32)
    )

    summaries = tf.summary.merge_all()

    return Graph(
        inputs = inputs,
        labels = labels,
        keep_probability = keep_probability,
        avg_cost = avg_cost,
        accuracy = accuracy,
        optimization_op = optimization_op,
        summaries = summaries,
    )
