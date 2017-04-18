from collections import namedtuple
import config
import tensorflow as tf

Graph = namedtuple("Graph", [
    "inputs",
    "labels",
    "keep_prob",
    "initial_cell_states",

    "avg_cost",
    "accuracy",
    "optimizer",
])

def build_graph(vocab_size):
    inputs = tf.placeholder(tf.int32, [None, config.SEQN_LEN])
    labels = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)

    embedding_matrix = tf.Variable(
        tf.truncated_normal([vocab_size, config.EMBEDDING_DIMS])
    )
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(config.LSTM_SIZE)
    drop_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_cell, output_keep_prob = keep_prob
    )
    multi_cell = tf.contrib.rnn.MultiRNNCell(
        [drop_cell] * config.LSTM_LAYERS
    )
    initial_cell_states = multi_cell.zero_state(
        config.BATCH_SIZE, tf.float32
    )

    outputs, _ = tf.nn.dynamic_rnn(
        multi_cell,
        embedded_inputs,
        initial_state = initial_cell_states
    )

    logits = tf.contrib.layers.fully_connected(
        outputs[:, -1], 1, activation_fn = None
    )
    avg_cost = tf.losses.sigmoid_cross_entropy(
        tf.reshape(labels, (-1, 1)), logits
    )
    estimated_probabilities = tf.sigmoid(logits)
    max_prob_estimates = tf.cast(
        tf.round(estimated_probabilities), tf.int32
    )
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(max_prob_estimates, labels), tf.float32)
    )

    optimizer = tf.train.AdamOptimizer(
        config.LEARNING_RATE
    ).minimize(avg_cost)

    return Graph(
        inputs = inputs,
        labels = labels,
        keep_prob = keep_prob,
        initial_cell_states = initial_cell_states,
        avg_cost = avg_cost,
        accuracy = accuracy,
        optimizer = optimizer
    )
