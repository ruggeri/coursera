import config
import dataset
import graph
import random
import tensorflow as tf

def run_batch(session, d, g):
    examples = random.sample(d.idx_examples, config.BATCH_SIZE)
    _, loss = session.run(
        [g.training_op, g.training_loss],
        feed_dict = {
            g.input_sequence: [e[0] for e in examples],
            g.output_sequence: [e[1] for e in examples],
            g.learning_rate: config.LEARNING_RATE
        }
    )

    print(f"Loss: {loss}")

def run(session):
    d = dataset.dataset(
        max_length = config.EXAMPLE_MAX_LEN,
        num_examples = config.NUM_EXAMPLES
    )

    g = graph.graph(
        batch_size = config.BATCH_SIZE,
        sequence_length = config.EXAMPLE_MAX_LEN,
        vocab_size = len(d.word_to_idx),
        num_embedding_dimensions = config.NUM_EMBEDDING_DIMENSIONS,
        num_lstm_layers = config.NUM_LSTM_LAYERS,
        num_lstm_units = config.NUM_LSTM_UNITS,
        start_word_idx = d.word_to_idx[dataset.start_word],
        stop_word_idx = d.word_to_idx[dataset.stop_word],
    )

    session.run(tf.global_variables_initializer())

    file_writer = tf.summary.FileWriter('logs/', session.graph)

    for epoch_idx in range(1, config.NUM_EPOCHS):
        num_batches = len(d.idx_examples) // config.BATCH_SIZE
        for batch_idx in range(1, num_batches):
            run_batch(session, d, g)

with tf.Session() as session:
    run(session)
