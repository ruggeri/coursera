import config
import graph as graph_fns
import numpy as np
import pdb
import tensorflow as tf

def sample_one_char(session, graph, states, prev_char):
    one_hot_prev_char = (
        config.file_reader.one_hot_encode_letter(prev_char)
    )
    one_hot_prev_char = one_hot_prev_char.reshape([1, 1, -1])

    results = session.run({
        "final_states": graph.final_states,
        "all_predictions": graph.all_predictions
    }, feed_dict={
        graph.inputs: one_hot_prev_char,
        tuple(graph.initial_states): tuple(states)
    })

    probs = results["all_predictions"][0].reshape([-1])
    probs[probs.argsort()[:-config.TOP_N]] = 0.0
    probs /= sum(probs)
    letter_int = np.random.choice(
        config.file_reader.vocab_size(), p=probs
    )
    letter = config.file_reader.int_to_char(letter_int)

    return (letter, results["final_states"])

def sample_chars(session, graph, prefix, num_chars_to_generate):
    states = graph_fns.make_initial_sampling_states(
        config.NUM_LAYERS, config.NUM_LSTM_UNITS
    )

    print("Beginning processing of prefix!")
    for char_idx in range(len(prefix) - 1):
        _, states = sample_one_char(
            session, graph, states, prefix[char_idx]
        )

    print("Beginning production of new characters!")
    result = prefix[-1]
    for char_idx in range(0, num_chars_to_generate):
        prev_char = result[-1]
        char, states = sample_one_char(
            session, graph, states, prev_char
        )
        result += char

    return result

def run(session):
    graph = graph_fns.build_sampling_graph(
        config.file_reader.vocab_size(),
        config.NUM_LAYERS,
        config.NUM_LSTM_UNITS
    )

    saver = tf.train.Saver()
    saver.restore(session, "./two-layer-rnn-model-anna-simplified-19-0439.ckpt")

    result = sample_chars(
        session,
        graph,
        config.file_reader.text()[:config.BURN_IN_LETTERS],
        config.CHARS_TO_GENERATE
    )

    print(result)

with tf.Session() as session:
    run(session)
