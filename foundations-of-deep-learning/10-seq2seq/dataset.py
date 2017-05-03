from collections import namedtuple
import random
import string

def random_sentence(max_length):
    string_length = random.randrange(1, max_length + 1)
    num_letters = len(string.ascii_lowercase)

    result = [
        string.ascii_lowercase[random.randrange(num_letters)]
        for _ in range(string_length)
    ]

    return result

def new_example(max_length):
    x = random_sentence(max_length)
    y = sorted(x)

    return (x, y)

def examples(max_length, num_examples):
    return [
        new_example(max_length) for _ in range(num_examples)
    ]

pad_word = "<pad>"
start_word = "<s>"
stop_word = "</s>"

special_words = [
    pad_word,
    "<unk>",
    start_word,
    stop_word,
]

def vocab_maps(dataset):
    words = set()
    for (x, y) in dataset:
        words.update(x)

    words.update(special_words)

    # For deterministic ordering.
    words = sorted(words)

    word_to_idx = { word: idx for (idx, word) in enumerate(words) }
    idx_to_word = { idx: word for (idx, word) in enumerate(words) }

    return (word_to_idx, idx_to_word)

def sentence_to_idxs(sentence, word_to_idx):
    return [word_to_idx[word] for word in sentence]

def examples_to_idx_examples(examples, word_to_idx):
    return [
        (sentence_to_idxs(x, word_to_idx),
         sentence_to_idxs(y, word_to_idx))
        for (x, y) in examples
    ]

def pad_examples(examples):
    max_len = max(map(lambda p: len(p[0]), examples))
    return [
        (x + ([pad_word] * (max_len - len(x))),
         y + ([pad_word] * (max_len - len(y))))
        for (x, y) in examples
    ]

Dataset = namedtuple("Dataset", [
    "examples",
    "word_to_idx",
    "idx_to_word",
    "training_idx_examples",
    "validation_idx_examples",
])

TRAINING_FRACTION = 0.9
def dataset(max_length, num_examples):
    examples_ = examples(max_length, num_examples)
    word_to_idx, idx_to_word = vocab_maps(examples_)
    idx_examples = examples_to_idx_examples(
        pad_examples(examples_), word_to_idx
    )

    num_training_examples = int(len(examples_) * TRAINING_FRACTION)

    return Dataset(
        examples = examples_,
        word_to_idx = word_to_idx,
        idx_to_word = idx_to_word,
        training_idx_examples = idx_examples[:num_training_examples],
        validation_idx_examples = idx_examples[num_training_examples:],
    )
