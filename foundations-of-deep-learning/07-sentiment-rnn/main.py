from collections import Counter, namedtuple
import numpy as np
import string
import tensorflow as tf

BATCH_SIZE = 500
BATCHES_PER_VALIDATION = 5
CHECKPOINT_NAME = "checkpoints/sentiment.ckpt"
EMBEDDING_DIMS = 300
LEARNING_RATE = 0.001
LSTM_LAYERS = 1
LSTM_SIZE = 256
NUM_EPOCHS = 1
SEQN_LEN = 200
SPLIT_FRAC = (0.8, 0.1, 0.1)

class DataSet:
    def __init__(self):
        self.reviews_words_ = None
        self.reviews_ints_ = None
        self.features_ = None
        self.labels_ = None

        self.word_counts_ = None
        self.words_to_ints_ = None
        self.ints_to_words_ = None

        self.datasets_ = None

    def reviews_words(self):
        if self.reviews_words_ is not None:
            return self.reviews_words_

        with open('./reviews.txt', 'r') as f:
            all_reviews_string = f.read()
        reviews_strings = all_reviews_string.split("\n")

        self.reviews_words_ = list(map(
            self.clean_and_split_review_string,
            reviews_strings
        ))

        return self.reviews_words_

    def clean_and_split_review_string(self, review_string):
        review_string = "".join([
            c for c in review_string if c not in string.punctuation
        ])
        review_words = review_string.split()
        return review_words

    def review_words_to_review_ints(self, review_words):
        words_to_ints = self.words_to_ints()
        review_ints = list(map(
            lambda word: words_to_ints[word], review_words
        ))
        return np.array(review_ints)

    def reviews_ints(self):
        if self.reviews_ints_ is not None:
            return self.reviews_ints_

        words_to_ints = self.words_to_ints()
        self.reviews_ints_ = list(map(
            self.review_words_to_review_ints,
            self.reviews_words()
        ))
        self.reviews_ints_ = np.array(self.reviews_ints_)

        return self.reviews_ints_

    def word_counts(self):
        if self.word_counts_ is not None:
            return self.word_counts_

        self.word_counts_ = Counter()
        for review_words in self.reviews_words():
            for review_word in review_words:
                self.word_counts_[review_word] += 1

        return self.word_counts_

    def word_count(self, word):
        return self.word_counts()[word]

    def build_words_to_ints_maps(self):
        if self.words_to_ints_ and self.ints_to_words_:
            return (self.words_to_ints_, self.ints_to_words_)

        vocabulary = sorted(
            self.word_counts().keys(),
            key = lambda word: self.word_count(word),
            reverse = True
        )

        # Note we will reserve word_int == 0 for a not-present word.
        self.words_to_ints_ = {
            word: word_int
            for word_int, word in enumerate(vocabulary, 1)
        }
        self.ints_to_words_ = {
            word_int: word
            for word, word_int in self.words_to_ints_.items()
        }

        return (self.words_to_ints_, self.ints_to_words_)

    def words_to_ints(self):
        if self.words_to_ints_ is None:
            self.build_words_to_ints_maps()
        return self.words_to_ints_

    def ints_to_words(self):
        if self.ints_to_words_ is None:
            self.build_words_to_ints_maps()
        return self.ints_to_words_

    def vocab_size(self):
        return len(self.words_to_ints())

    def labels(self):
        if self.labels_ is not None:
            return self.labels_
        with open('./labels.txt', 'r') as f:
            labels_text = f.read().split("\n")
        self.labels_ = [
            1 if label == "positive" else 0 for label in labels_text
        ]
        return self.labels_

    def label_counts(self):
        return Counter(self.labels())

    def features(self):
        if self.features_ is not None:
            return self.features_
        self.features_ = np.array(
            list(map(self.featurize_review, self.reviews_ints())),
            dtype=np.int32
        )
        return self.features_

    def featurize_review(self, review):
        if len(review) >= SEQN_LEN:
            return review[:SEQN_LEN]
        num_zeros = SEQN_LEN - len(review)
        featurized_review = np.zeros(SEQN_LEN)
        featurized_review[num_zeros:] = review
        return featurized_review

    def dataset(self, name):
        if self.datasets_ is not None:
            return self.datasets_[name]

        features = self.features()
        labels = self.labels()

        splits = [int(features.shape[0] * SPLIT_FRAC[0])]
        splits.append(
            splits[-1] + int(features.shape[0] * SPLIT_FRAC[1])
        )
        splits.append(
            splits[-1] + int(features.shape[0] * SPLIT_FRAC[2])
        )

        train_x = features[:splits[0], :]
        train_y = labels[:splits[0]]

        validation_x = features[splits[0]:splits[1], :]
        validation_y = labels[splits[0]:splits[1]]

        test_x = features[splits[1]:splits[2], :]
        test_y = labels[splits[1]:splits[2]]

        self.datasets_ = {
            "train": (train_x, train_y),
            "validation": (validation_x, validation_y),
            "test": (test_x, test_y)
        }

        return self.datasets_[name]

    def get_batches(self, x, y):
        num_batches = self.num_batches(x, y)

        x = x[:(num_batches * BATCH_SIZE)]
        y = y[:(num_batches * BATCH_SIZE)]

        for batch_start_idx in range(0, len(x), BATCH_SIZE):
            batch_end_idx = batch_start_idx + BATCH_SIZE
            yield (
                x[batch_start_idx:batch_end_idx],
                y[batch_start_idx:batch_end_idx]
            )

    def num_batches(self, x, y):
        return len(x) // BATCH_SIZE

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
    inputs = tf.placeholder(tf.int32, [None, SEQN_LEN])
    labels = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)

    embedding_matrix = tf.Variable(
        tf.truncated_normal([vocab_size, EMBEDDING_DIMS])
    )
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
    drop_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_cell, output_keep_prob = keep_prob
    )
    multi_cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * LSTM_LAYERS)
    initial_cell_states = multi_cell.zero_state(BATCH_SIZE, tf.float32)

    outputs, _ = tf.nn.dynamic_rnn(
        multi_cell,
        embedded_inputs,
        initial_state = initial_cell_states
    )

    predictions = tf.contrib.layers.fully_connected(
        outputs[:, -1], 1, activation_fn = tf.sigmoid
    )
    avg_cost = tf.losses.mean_squared_error(
        tf.reshape(labels, (-1, 1)), predictions
    )
    prediction_is_correct = tf.equal(
        tf.cast(tf.round(predictions), tf.int32), labels
    )
    accuracy = tf.reduce_mean(
        tf.cast(prediction_is_correct, tf.float32)
    )

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(avg_cost)

    return Graph(
        inputs = inputs,
        labels = labels,
        keep_prob = keep_prob,
        initial_cell_states = initial_cell_states,
        avg_cost = avg_cost,
        accuracy = accuracy,
        optimizer = optimizer
    )

BatchInfo = namedtuple("BatchInfo", [
    "epoch_idx",
    "batch_idx",
    "inputs",
    "labels",
    "num_batches",
])
RunInfo = namedtuple("RunInfo", [
    "session",
    "graph",
    "data_set",
    "saver",
])

def run_batch(run_info, batch_info):
    ri = run_info
    initial_cell_states = run_info.session.run(
        ri.graph.initial_cell_states
    )

    avg_cost, accuracy, _ = ri.session.run([
        ri.graph.avg_cost, ri.graph.accuracy, ri.graph.optimizer
    ], feed_dict = {
        ri.graph.inputs: batch_info.inputs,
        ri.graph.labels: batch_info.labels,
        ri.graph.keep_prob: 0.5,
        ri.graph.initial_cell_states: initial_cell_states
    })

    print(f"Epoch: {batch_info.epoch_idx}/{NUM_EPOCHS}",
          f"Batch: {batch_info.batch_idx}/{batch_info.num_batches}",
          f"Train loss: {avg_cost:.3f}",
          f"accuracy: {accuracy:.3f}")

def run_validation(run_info, batch_info):
    ri = run_info

    initial_cell_states = ri.session.run(ri.graph.initial_cell_states)

    validation_accuracy = 0.0
    (validation_x, validation_y) = ri.data_set.dataset("validation")
    num_batches = ri.data_set.num_batches(validation_x, validation_y)
    batches = run_info.data_set.get_batches(validation_x, validation_y)
    for inputs, labels in batches:
        feed = {inputs_: x,
                labels_: y,
                keep_prob: 1,
                initial_state: val_state}
        batch_accuracy = sess.run(ri.accuracy, feed_dict = {
            ri.graph.inputs: inputs,
            ri.graph.labels: labels,
            ri.graph.keep_prob: 1.0,
        })
        validation_accuracy += batch_acc

    validation_accuracy /= num_batches
    print(f"Val acc: {validation_accuracy:.3f}")

def run_epoch(run_info, epoch_idx):
    (train_x, train_y) = run_info.data_set.dataset("train")
    batches = run_info.data_set.get_batches(train_x, train_y)
    num_batches = run_info.data_set.num_batches(train_x, train_y)
    for batch_idx, (inputs, labels) in enumerate(batches, 1):
        batch_info = BatchInfo(
            epoch_idx = epoch_idx,
            batch_idx = batch_idx,
            inputs = inputs,
            labels = labels,
            num_batches = num_batches
        )

        run_batch(run_info, batch_info)

        if batch_idx % BATCHES_PER_VALIDATION == 0:
            run_validation()

def run(session):
    data_set = DataSet()
    run_info = RunInfo(
        session = session,
        graph = build_graph(data_set.vocab_size()),
        data_set = data_set,
        saver = tf.train.Saver()
    )

    print(f"Label counts: {run_info.data_set.label_counts()}")

    session.run(tf.global_variables_initializer())
    for epoch_idx in range(NUM_EPOCHS):
        run_epoch(run_info, epoch_idx)
        run_info.saver.save(session, CHECKPOINT_NAME)

with tf.Session() as session:
    run(session)
