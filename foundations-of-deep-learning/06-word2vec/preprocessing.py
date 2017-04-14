from collections import Counter
import config
import math
import random

def replace_punctuation(text):
    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words

def create_lookup_tables(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_word = {ii: word for ii, word in enumerate(sorted_vocab)}
    word_to_int = {word: ii for ii, word in int_to_word.items()}

    return word_to_int, int_to_word

def int_encode_words(words, word_to_int):
    return [word_to_int[word] for word in words]

def subsample(words):
    total_num_words = len(words)
    counts = Counter(words)
    new_words = []
    for word in words:
        word_count = counts[word]
        word_freq = word_count / total_num_words
        reject_prob = (
            1 - math.sqrt(config.SUBSAMPLE_THRESHOLD / word_freq)
        )
        if random.random() >= reject_prob:
            new_words.append(word)
    return new_words

def word_context(words, idx, window_size=5):
    context_size = random.randrange(1, window_size)
    window_start = max(idx - context_size, 0)
    window_end = min(idx + context_size, len(words)) + 1
    context = words[window_start:idx]
    context.extend(words[(idx + 1):window_end])

    return context

def make_training_batches(words, batch_size, window_size):
    n_batches = len(words) // batch_size

    # get only full batches
    words = words[:n_batches*batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:(idx + batch_size)]
        for ii in range(len(batch)):
            word = batch[ii]
            context_words = word_context(batch, ii, window_size)
            y.extend(context_words)
            x.extend([word] * len(context_words))
        yield x, y

class Batcher:
    def __init__(self):
        self.training_text_ = None
        self.training_words_ = None
        self.training_int_words_ = None
        self.words_to_ints_ = None
        self.ints_to_words_ = None
        self.int_words_ = None
        self.vocab_size_ = None

    def training_text(self):
        if not self.training_text_:
            with open('data/text8') as f:
                self.training_text_ = f.read()

        return self.training_text_

    def training_words(self):
        if not self.training_words_:
            self.training_words_ = replace_punctuation(
                self.training_text()
            )
            self.training_words_ = subsample(
                self.training_words_
            )
        return self.training_words_

    def make_words_and_ints_maps(self):
        if not self.words_to_ints_ or not self.ints_to_words_:
            self.words_to_ints_, self.ints_to_words_ = (
                create_lookup_tables(self.training_words())
            )
        return (self.words_to_ints_, self.ints_to_words_)

    def words_to_ints(self):
        if not self.words_to_ints_:
            self.make_words_and_ints_maps()
        return self.words_to_ints_

    def ints_to_words(self):
        if not self.ints_to_words_:
            self.make_words_and_ints_maps()
        return self.ints_to_words_

    def training_int_words(self):
        if not self.training_int_words_:
            self.training_int_words_ = int_encode_words(
                self.training_words(), self.words_to_ints()
            )
        return self.training_int_words_

    def word_to_int(self, word):
        return self.words_to_ints()[word]

    def int_to_word(self, int_word):
        return self.ints_to_words()[int_word]

    def batches(self, batch_size, window_size):
        return make_training_batches(
            self.training_int_words(), batch_size, window_size
        )

    def num_batches(self, batch_size):
        return len(self.training_int_words()) // batch_size

    def vocab_size(self):
        if not self.vocab_size_:
            self.vocab_size_ = len(self.words_to_ints())
        return self.vocab_size_
