from itertools import *
import math
import numpy as np
import random
from collections import Counter

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

    def convert_reviews(self, reviews):
        inputs = []
        for review in reviews:
            input_v = np.zeros((self.num_words, 1))
            self.fill_counts(review, input_v)
            inputs.append(input_v)

        return inputs

    def is_in_vocabulary(self, word):
        return word in self.words_to_index

    def fill_counts(self, review, counts):
        counts *= 0
        for word in review.split():
            if not self.is_in_vocabulary(word):
                continue
            index = self.words_to_index[word]
            counts[index] += 1.0

class NeuralNetwork:
    DEFAULT_NUM_HIDDEN_UNITS = 40
    DEFAULT_BATCH_SIZE = 400
    DEFAULT_LEARNING_RATE = 0.1

    def __init__(self,
                 num_input_units,
                 num_hidden_units = DEFAULT_NUM_HIDDEN_UNITS,
                 learning_rate = DEFAULT_LEARNING_RATE,
                 batch_size = DEFAULT_BATCH_SIZE):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Arrays for forward propagation.
        self.input_layer = np.zeros((num_input_units, 1))
        self.input_to_hidden_weights = np.random.normal(
            0.0,
            num_hidden_units ** -0.5,
            size=(num_hidden_units, num_input_units)
        )
        self.hidden_bias = np.random.normal(
            0.0,
            num_hidden_units ** -0.5,
            size=(num_hidden_units, 1)
        )
        self.hidden_layer = np.zeros((num_hidden_units, 1))
        self.hidden_to_output_weights = \
          np.random.normal(size=(1, num_hidden_units))
        self.output_bias = np.random.normal()

        # Arrays for back propagation.
        self.output_input_derivative = 0.0
        self.output_weights_derivative = \
          np.zeros(self.hidden_to_output_weights.shape)
        self.hidden_input_derivative = np.zeros(self.hidden_layer.shape)
        self.hidden_weights_derivative = \
          np.zeros(self.input_to_hidden_weights.shape)

        # Arrays for collecting GD step
        self.output_bias_step = 0.0
        self.output_weights_step = \
          np.zeros(self.hidden_to_output_weights.shape)
        self.hidden_bias_step = np.zeros(self.hidden_layer.shape)
        self.hidden_weights_step = \
          np.zeros(self.input_to_hidden_weights.shape)

    def forward_propagate(self, input_vector):
        self.input_layer[:] = input_vector

        # Compute hidden layer
        self.input_to_hidden_weights.dot(self.input_layer,
                                         out=self.hidden_layer)
        self.hidden_layer += self.hidden_bias
        NeuralNetwork.sigmoid(self.hidden_layer)

        # Compute output layer
        output = self.hidden_to_output_weights.dot(self.hidden_layer)
        output += self.output_bias
        NeuralNetwork.sigmoid(output)

        return float(output)

    def backward_propagate(self, output, target):
        ce_derivative = NeuralNetwork.ce_derivative(output, target)
        self.output_input_derivative = \
          ce_derivative * NeuralNetwork.sigmoid_derivative(output)

        # Propagate to hidden to output weights
        self.output_weights_derivative.fill(1)
        self.output_weights_derivative *= self.hidden_layer.T
        self.output_weights_derivative *= self.output_input_derivative

        # Propagate to hidden inputs
        NeuralNetwork.sigmoid_derivative(
            self.hidden_layer, out=self.hidden_input_derivative
        )
        self.hidden_input_derivative *= self.hidden_to_output_weights.T
        self.hidden_input_derivative *= self.output_input_derivative

        # Propagate to input to hidden weights.
        self.hidden_weights_derivative.fill(1)
        self.hidden_weights_derivative *= self.input_layer.T
        self.hidden_weights_derivative *= self.hidden_input_derivative

    def train(self, inputs, targets, num_epochs):
        num_examples = len(inputs)
        num_batches = math.ceil(num_examples / self.batch_size)

        batch_epoch_product = \
          product(range(num_batches), range(num_epochs))
        for (batch_idx, epoch_idx) in batch_epoch_product:
            print(f"Training {(epoch_idx, batch_idx)} of {(num_epochs, num_batches)}")
            start_pos = batch_idx * self.batch_size
            end_pos = min((batch_idx + 1) * self.batch_size, num_examples)
            inputs_batch = islice(inputs, start_pos, end_pos)
            targets_batch = islice(targets, start_pos, end_pos)
            self.train_batch(inputs_batch, targets_batch, end_pos - start_pos)

    def train_batch(self, inputs_batch, targets_batch, num_examples):
        self.output_weights_step *= 0
        self.hidden_weights_step *= 0

        input_target_pairs = zip(inputs_batch, targets_batch)
        for (input_v, target) in input_target_pairs:
            output = self.forward_propagate(input_v)
            self.backward_propagate(output, target)

            # Subtract so that we *decrease* the CE.
            self.output_bias_step -= self.output_input_derivative
            self.output_weights_step -= self.output_weights_derivative
            self.hidden_bias_step -= self.hidden_input_derivative
            self.hidden_weights_step -= self.hidden_weights_derivative

        self.output_bias_step *= self.learning_rate / num_examples
        self.output_weights_step *= self.learning_rate / num_examples
        self.hidden_bias_step *= self.learning_rate / num_examples
        self.hidden_weights_step *= self.learning_rate / num_examples

        self.output_bias += self.output_bias_step
        self.hidden_to_output_weights += self.output_weights_step
        self.hidden_bias += self.hidden_bias_step
        self.input_to_hidden_weights += self.hidden_weights_step

    def after_trained_batch(self, inputs, targets):
        num_examples = len(inputs)
        samples = random.sample(range(num_examples), 100)
        test_inputs = [inputs[i] for i in samples]
        test_targets = [targets[i] for i in samples]
        self.evaluate_loss(test_inputs, test_targets)

    def evaluate_loss(self, inputs, labels):
        loss = 0.0

        num_pos = 0
        num_neg = 0
        misclassifications = 0
        for (review, label) in zip(inputs, labels):
            output = self.forward_propagate(review)

            if (output < 0.5):
                num_neg += 1
                if (label == 1):
                    misclassifications += 1
            elif (output > 0.5):
                num_pos += 1
                if (label == 0):
                    misclassifications += 1
            loss += NeuralNetwork.cross_entropy(output, label)

        error_rate = misclassifications / (num_pos + num_neg)
        print((num_pos, num_neg, misclassifications, error_rate))
        return loss

    @staticmethod
    def sigmoid(vec):
        np.exp(-vec, out=vec)
        vec += 1.0
        np.reciprocal(vec, out=vec)

    @staticmethod
    def cross_entropy(output, target):
        # This is the expected # of bits to encode the average token
        # if we draw tokens from an output stream where t% of tokens
        # are 1s and (1-t)% are 0s.
        return (-target*np.log(output)) + (-(1-target)*np.log(1-output))

    @staticmethod
    def ce_derivative(output, target):
        #return 2 * (output - target)
        return -target * (1/output) + (-(1-target) * -1/(1-output))

    @staticmethod
    def sigmoid_derivative(activation, out=None):
        if type(activation) == float:
            return activation * (1 - activation)

        out.fill(1)
        out -= activation
        out *= activation

def read_input_files():
    with open("reviews.txt", "r") as f:
        reviews = list(review.strip() for review in f.readlines())
    with open("labels.txt", "r") as f:
        labels = []
        for label in f.readlines():
            label = label.strip().upper()
            if label == "POSITIVE":
                labels.append(1)
            elif label == "NEGATIVE":
                labels.append(0)
            else:
                raise "WTF?"

    return (reviews, labels)

def main():
    print("Reading Input")
    (reviews, labels) = read_input_files()

    print("Building Vocabulary")
    vocabulary = Vocabulary(reviews)
    print(f"Vocab size: {vocabulary.num_words}")
    print("Converting inputs")
    inputs = vocabulary.convert_reviews(reviews)

    train_inputs = inputs[:-1000]
    train_labels = labels[:-1000]
    test_inputs = inputs[-1000:]
    test_labels = labels[-1000:]

    print("Beginning Training")
    nn = NeuralNetwork(vocabulary.num_words)
    for epoch in range(1000):
        nn.train(train_inputs, train_labels, 1)
        print(nn.evaluate_loss(test_inputs, test_labels))

if __name__ == "__main__":
    main()
