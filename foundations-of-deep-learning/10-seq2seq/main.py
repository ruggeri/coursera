from collections import namedtuple
import config
import dataset
import graph
import random
import tensorflow as tf

RunInfo = namedtuple("RunInfo", [
    "dataset",
    "graph",
    "session",
])

BatchInfo = namedtuple("BatchInfo", [
    "epoch_idx",
    "batch_idx",
])

def run_batch(run_info, batch_info):
    d, g = run_info.dataset, run_info.graph

    examples = random.sample(d.training_idx_examples, config.BATCH_SIZE)
    _, loss = run_info.session.run(
        [g.training_op, g.training_loss],
        feed_dict = {
            g.input_sequence: [e[0] for e in examples],
            g.output_sequence: [e[1] for e in examples],
            g.learning_rate: config.LEARNING_RATE
        }
    )

    print(f"Epoch {batch_info.epoch_idx:02d} | "
          f"Batch {batch_info.batch_idx:02d} | "
          f"Loss: {loss:.3f}")

def run_validation(run_info, epoch_idx):
    print("Beginning validation")
    d, g = run_info.dataset, run_info.graph

    examples = random.sample(
        d.validation_idx_examples, config.BATCH_SIZE
    )
    accuracy = run_info.session.run(
        g.accuracy,
        feed_dict = {
            g.input_sequence: [e[0] for e in examples],
            g.output_sequence: [e[1] for e in examples],
        }
    )

    print(f"Epoch {epoch_idx:02d} | "
          f"Accuracy: {accuracy:.3f}")


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

    run_info = RunInfo(
        dataset = d,
        graph = g,
        session = session
    )
    for epoch_idx in range(1, config.NUM_EPOCHS):
        num_batches = len(d.training_idx_examples) // config.BATCH_SIZE
        for batch_idx in range(1, num_batches):
            batch_info = BatchInfo(
                epoch_idx = epoch_idx,
                batch_idx = batch_idx,
            )
            run_batch(run_info, batch_info)
        run_validation(run_info, epoch_idx)

with tf.Session() as session:
    run(session)
