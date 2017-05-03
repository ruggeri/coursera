import random
import string

def random_sentence(max_length):
    string_length = random.randrange(1, max_length + 1)
    num_letters = len(string.ascii_lowercase)

    result = "".join(
        [string.ascii_lowercase[random.randrange(num_letters)]
         for _ in range(string_length)]
    )

    return result

def new_example(max_length):
    x = random_sentence(max_length)
    y = "".join(sorted(x))

    return (x, y)

def dataset(max_length, num_examples):
    return [
        new_example(max_length) for _ in range(num_examples)
    ]
