from collections import Counter
import config
import math
import random
import re

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

def subsample(int_words):
    total_num_words = len(int_words)
    counts = Counter(int_words)
    new_int_words = []
    for int_word in int_words:
        word_count = counts[int_word]
        word_freq = word_count / total_num_words
        reject_prob = (
            1 - math.sqrt(config.SUBSAMPLE_THRESHOLD / word_freq)
        )
        if random.random() >= reject_prob:
            new_int_words.append(int_word)
    return new_int_words

def word_context(words, idx, window_size=5):
    context_size = random.randrange(1, window_size)
    window_start = max(idx - context_size, 0)
    window_end = min(idx + context_size, len(words)) + 1
    context = words[window_start:idx]
    context.extend(words[(idx + 1):window_end])

    return context

def make_training_batches(words, batch_size, window_size=5):
    n_batches = len(words) // batch_size

    # get only full batches
    words = words[:n_batches*batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            word = batch[ii]
            context_words = word_context(batch, ii, window_size)
            y.extend(word)
            x.extend([context_words] * len(batch_y))
        yield x, y
