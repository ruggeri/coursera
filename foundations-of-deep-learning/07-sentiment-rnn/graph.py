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
    "summaries",
])

def build_graph(vocab_size):
    inputs = tf.placeholder(tf.int32, [None, config.SEQN_LEN])
    labels = tf.placeholder(tf.int32, [None])
    tf.summary.histogram("labels", labels)
    keep_prob = tf.placeholder(tf.float32)

    embedding_matrix = tf.Variable(
        tf.truncated_normal([vocab_size, config.EMBEDDING_DIMS])
    )
    tf.summary.histogram("embedding_matrix", embedding_matrix)
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
    tf.summary.histogram("embedded_inputs", embedded_inputs)

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
    tf.summary.histogram("outputs", outputs[:, -1])

    logits = tf.contrib.layers.fully_connected(
        outputs[:, -1], 1, activation_fn = None
    )
    tf.summary.histogram("logits", logits)
    avg_cost = tf.losses.sigmoid_cross_entropy(
        tf.reshape(labels, (-1, 1)),
        logits
    )
    tf.summary.scalar("avg_cost", avg_cost)
    estimated_probabilities = tf.sigmoid(logits)
    tf.summary.histogram(
        "estimated_probabilities",
        estimated_probabilities
    )

    max_prob_estimates = tf.cast(
        tf.round(estimated_probabilities), tf.int32
    )
    tf.summary.histogram("max_prob_estimates", max_prob_estimates)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(max_prob_estimates, tf.reshape(labels, (-1, 1))),
        tf.float32)
    )
    tf.summary.scalar("accuracy", accuracy)

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
        optimizer = optimizer,
        summaries = tf.summary.merge_all()
    )
