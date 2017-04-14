from collections import namedtuple
import config
import tensorflow as tf

Graph = namedtuple("Graph", [
    "inputs",
    "labels",
    "cost",
    "optimizer"
])

def build_graph(vocab_size, num_embedding_units, num_negative_samples):
    inputs = tf.placeholder(tf.int32, [None], name = "input_words")
    labels = tf.placeholder(tf.int32, [None], name = "labels")

    # Perform the embedding.
    embedding_matrix = tf.Variable(
        tf.random_uniform(
            [vocab_size, num_embedding_units],
            minval = -1,
            maxval = 1,
        ),
        name = "embedding_matrix"
    )
    embedded_inputs = tf.nn.embedding_lookup(
        embedding_matrix, inputs, name = "embedded_inputs"
    )

    # Prepare the softmax weights/biases.
    initial_weight_limit = 1 / config.NUM_EMBEDDING_UNITS
    softmax_w = tf.Variable(
        tf.random_uniform(
            [vocab_size, num_embedding_units],
            minval = -initial_weight_limit,
            maxval = initial_weight_limit
        ),
        name="softmax_weights"
    )
    softmax_b = tf.Variable(
        tf.zeros([vocab_size]), name = "softmax_biases"
    )

    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(
        softmax_w,
        softmax_b,
        tf.reshape(labels, (-1, 1)),
        embedded_inputs,
        num_negative_samples,
        vocab_size
    )

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    return Graph(
        inputs = inputs,
        labels = labels,
        cost = cost,
        optimizer = optimizer
    )
