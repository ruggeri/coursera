from collections import Counter
import config
import numpy as np
import string

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

        with open('./datasets/reviews.txt', 'r') as f:
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
        with open('./datasets/labels.txt', 'r') as f:
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
        if len(review) >= config.SEQN_LEN:
            return review[:config.SEQN_LEN]
        num_zeros = config.SEQN_LEN - len(review)
        featurized_review = np.zeros(config.SEQN_LEN)
        featurized_review[num_zeros:] = review
        return featurized_review

    def dataset(self, name):
        if self.datasets_ is not None:
            return self.datasets_[name]

        features = self.features()
        labels = self.labels()

        splits = [int(features.shape[0] * config.SPLIT_FRAC[0])]
        splits.append(
            splits[-1] + int(features.shape[0] * config.SPLIT_FRAC[1])
        )
        splits.append(
            splits[-1] + int(features.shape[0] * config.SPLIT_FRAC[2])
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

        x = x[:(num_batches * config.BATCH_SIZE)]
        y = y[:(num_batches * config.BATCH_SIZE)]

        for batch_start_idx in range(0, len(x), config.BATCH_SIZE):
            batch_end_idx = batch_start_idx + config.BATCH_SIZE
            yield (
                x[batch_start_idx:batch_end_idx],
                y[batch_start_idx:batch_end_idx]
            )

    def num_batches(self, x, y):
        return len(x) // config.BATCH_SIZE
