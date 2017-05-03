from collections import namedtuple
import tensorflow as tf
from tensorflow.python.layers.core import Dense

Graph = namedtuple("Graph", [
    "input_sequence",
    "output_sequence",
    "learning_rate",
    "training_loss",
    "training_op",
    "predictions",
])

def graph(batch_size,
          sequence_length,
          vocab_size,
          num_embedding_dimensions,
          num_lstm_layers,
          num_lstm_units,
          start_word_idx,
          stop_word_idx):
    input_sequence = tf.placeholder(
        tf.int32,
        [batch_size, sequence_length],
        name = "input_sequence"
    )
    output_sequence = tf.placeholder(
        tf.int32,
        [batch_size, sequence_length],
        name = "output_sequence"
    )
    learning_rate = tf.placeholder(
        tf.float32, name = "learning_rate"
    )

    embedded_encoder_input_sequence = tf.contrib.layers.embed_sequence(
        input_sequence,
        vocab_size = vocab_size,
        embed_dim = num_embedding_dimensions
    )

    encoder_cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.BasicLSTMCell(num_lstm_units)
         for _ in range(num_lstm_layers)]
    )
    _, final_encoder_state = tf.nn.dynamic_rnn(
        encoder_cells,
        embedded_encoder_input_sequence,
        dtype = tf.float32
    )

    with tf.variable_scope("decoder_lstm_cells"):
        training_decoder_cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(num_lstm_units)
             for _ in range(num_lstm_layers)]
        )
    with tf.variable_scope("decoder_lstm_cells"):
        inference_decoder_cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(num_lstm_units, reuse = True)
             for _ in range(num_lstm_layers)]
        )

    # During training, at each time step feed the previous correct
    # word into the decoder, even if this was not the one that was
    # selected at the previous time step. This leads to better
    # training.
    decoder_training_inputs = tf.strided_slice(
        output_sequence,
        [0, 0],
        [batch_size, sequence_length - 1],
        [1, 1]
    )
    decoder_training_inputs = tf.concat(
        [tf.fill([batch_size, 1], start_word_idx),
         decoder_training_inputs],
        axis = 1
    )

    decoder_embedding_matrix = tf.Variable(
        tf.truncated_normal([vocab_size, num_embedding_dimensions]),
        name ="decoder_embedding_matrix"
    )
    embedded_decoder_training_inputs = tf.nn.embedding_lookup(
        decoder_embedding_matrix,
        decoder_training_inputs
    )

    # This densely connects the encoder LSTM units to make predictions
    # on the words.
    output_layer = Dense(
        vocab_size,
        activation = tf.nn.relu,
        name = "decoder_predictions",
    )

    # Both the training and inference decoders will use the same LSTM
    # cell, but will have a different Helper. The helper decides what
    # to feed to the next time step's LSTM
    training_decoder = tf.contrib.seq2seq.BasicDecoder(
        training_decoder_cells,
        tf.contrib.seq2seq.TrainingHelper(
            embedded_decoder_training_inputs,
            tf.fill([batch_size], sequence_length)
        ),
        final_encoder_state,
        output_layer = output_layer
    )
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
        inference_decoder_cells,
        tf.contrib.seq2seq.GreedyEmbeddingHelper(
            decoder_embedding_matrix,
            tf.fill([batch_size], start_word_idx),
            stop_word_idx
        ),
        final_encoder_state,
        output_layer = output_layer
    )

    training_output, _ = tf.contrib.seq2seq.dynamic_decode(
        training_decoder
    )
    inference_output, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder
    )

    # TODO: (1) should not count loss on padding words?, (2) may want
    # to learn how to do a sampled loss?
    training_loss = tf.contrib.seq2seq.sequence_loss(
        training_output.rnn_output,
        output_sequence,
        tf.fill([batch_size, sequence_length], 1.0)
    )

    training_op = tf.train.AdamOptimizer(
        learning_rate = learning_rate,
    ).minimize(training_loss)

    return Graph(
        input_sequence = input_sequence,
        output_sequence = output_sequence,
        learning_rate = learning_rate,
        training_loss = training_loss,
        training_op = training_op,
        predictions = inference_output.rnn_output
    )
