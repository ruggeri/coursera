from collections import Counter
import numpy as np

class Vocabulary:
    THRESHOLD = 0.01

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

    def select_words(self):
        # TODO: I'm supposed to be using Index99 here.
        threshold = (self.THRESHOLD * self.num_reviews)

        selected_words = []
        for (word, count) in self.total_counts.items():
            if count > threshold:
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
