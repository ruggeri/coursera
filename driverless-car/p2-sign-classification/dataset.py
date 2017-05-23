from collections import namedtuple
import config
import numpy as np
import pickle

def normalize_x(x):
    # LeCun says he did better with just grayscale!
    x = np.sum(x, axis = 3) / 3
    # I want to keep x a 4d tensor, even if only one image map.
    x = x.reshape((*x.shape, 1))

    # The idea here is to normalize each image so that it has mean
    # zero and unit variance.
    x = x - np.mean(x, axis = (1, 2), keepdims = True)
    x = x / np.std(x, axis = (1, 2), keepdims = True)

    return x

def shuffle(x, y):
    idxs = np.random.permutation(x.shape[0])
    x = x[idxs, :, :, :]
    y = y[idxs]

    return (x, y)

def build_batches(x, y, batch_size):
    num_examples = x.shape[0]
    num_batches = np.ceil(num_examples / batch_size)

    def batches():
        for start_i in range(0, num_examples, batch_size):
            end_i = min(start_i + batch_size, num_examples)
            batch_x = x[start_i:end_i, :, :, :]
            batch_y = y[start_i:end_i]

            yield (batch_x, batch_y)

    return (num_batches, batches())

Dataset = namedtuple("Dataset", [
    "X_train",
    "y_train",
    "X_valid",
    "y_valid",
    "X_test",
    "y_test",

    "num_classes",
    "image_shape",
])

def build(X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_train = normalize_x(X_train)
    X_valid = normalize_x(X_valid)
    X_test = normalize_x(X_test)

    X_train, y_train = shuffle(X_train, y_train)

    num_classes = len(set(y_train))

    return Dataset(
        X_train = X_train,
        y_train = y_train,
        X_valid = X_valid,
        y_valid = y_valid,
        X_test = X_test,
        y_test = y_test,

        num_classes = num_classes,
        image_shape = X_train.shape[1:],
    )

def load():
    training_file = f"{config.DATA_DIR}/train_augmented.p"
    validation_file= f"{config.DATA_DIR}/valid.p"
    testing_file = f"{config.DATA_DIR}/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return build(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test
    )
