import preprocessing
import config

with open('data/text8') as f:
    text = f.read()
words = preprocessing.replace_punctuation(text)
word_to_int, int_to_word = preprocessing.create_lookup_tables(words)
int_words = preprocessing.int_encode_words(words, word_to_int)
int_words = preprocessing.subsample(int_words)
batches = preprocessing.make_training_batches(
    int_words, config.BATCH_SIZE, config.WINDOW_SIZE
)
