from collections import Counter
from math import log
import numpy as np

class Vocabulary:
    THRESHOLD = 0.05
    PSEUDOCOUNTS = 2

    def __init__(self, reviews, targets):
        self.total_counts = Counter()
        self.positive_counts = Counter()
        self.negative_counts = Counter()

        self.num_reviews = len(targets)
        self.num_positive_reviews = sum(targets)
        self.num_negative_reviews = \
          self.num_reviews - self.num_positive_reviews

        self.count_words(reviews, targets)

        selected_words = self.select_words()
        self.words_to_index = Vocabulary.word_index(selected_words)
        self.num_words = len(self.words_to_index)

    def count_words(self, reviews, targets):
        for (review, target) in zip(reviews, targets):
            self.count_words_for_review(review, target)

    def count_words_for_review(self, review, target):
        for word in set(review.split()):
            self.total_counts[word] += 1
            if target == 1:
                self.positive_counts[word] += 1
            else:
                self.negative_counts[word] += 1

    def log_odds_ratio(self, word):
        pos_count = self.positive_counts[word] + self.PSEUDOCOUNTS
        neg_count = self.negative_counts[word] + self.PSEUDOCOUNTS
        odds_ratio = pos_count / neg_count
        # Clearly if we pick a lot of positive reviews, this will bias
        # the odds ratio high for every feature. But I don't know if
        # this is the correct way to correct the effect. It doesn't
        # really matter, since normalizer is the same everywhere.
        normalizer = self.num_positive_reviews / self.num_negative_reviews
        log_odds_ratio = log(odds_ratio / normalizer)

        return log_odds_ratio

    def log_odds_ratio_pairs(self):
        lor_pairs = []
        for word in self.total_counts:
            lor = self.log_odds_ratio(word)
            lor_pairs.append((word, lor))

        # Sort in descending order of predictive power.
        lor_pairs.sort(key = lambda p: -abs(p[1]))

        return lor_pairs

    def select_words(self):
        lor_pairs = self.log_odds_ratio_pairs()

        # Only pick the best words.
        threshold = int(self.THRESHOLD * len(lor_pairs))
        selected_words = []
        for (word, _) in lor_pairs[:threshold]:
            selected_words.append(word)

        return selected_words

    @staticmethod
    def word_index(words):
        words_to_index = {}
        for idx, word in enumerate(words):
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
