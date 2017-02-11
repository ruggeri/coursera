from collections import Counter
from itertools import chain
import numpy as np

class Vocabulary:
    THRESHOLD = 800

    def __init__(self, reviews):
        self.word_counts = Vocabulary.count_words(reviews)
        self.most_common_words = \
          Vocabulary.most_common_words(self.word_counts)
        self.words_to_index = \
          Vocabulary.word_index(self.most_common_words)
        self.num_words = len(self.words_to_index)

    @staticmethod
    def count_words(reviews):
        word_counts = Counter()

        words = chain.from_iterable(rev.split() for rev in reviews)
        for word in words:
            word_counts[word] += 1

        return word_counts

    @staticmethod
    def most_common_words(word_counts):
        return list(w for (w, c) in word_counts.items() \
                                 if c > Vocabulary.THRESHOLD)

    @staticmethod
    def word_index(most_common_words):
        words_to_index = {}
        for idx, word in enumerate(most_common_words):
            words_to_index[word] = idx

        return words_to_index

    def featurize(self, reviews):
        inputs = []
        for review in reviews:
            input_v = np.zeros((self.num_words, 1))
            self.fill_counts(review, input_v)
            inputs.append(input_v)

        return inputs

    def fill_counts(self, review, counts):
        counts *= 0
        for word in review.split():
            if not self.is_in_vocabulary(word):
                continue
            index = self.words_to_index[word]
            # Note that we are just doing this binary style.
            counts[index] = 1.0

    def is_in_vocabulary(self, word):
        return word in self.words_to_index
