import config
import dataset
import graph
import tensorflow as tf

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

file_writer = tf.summary.FileWriter('logs/', tf.get_default_graph())
