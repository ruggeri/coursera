from collections import namedtuple
import pickle
from sklearn.model_selection import train_test_split

Dataset = namedtuple("Dataset", [
    "train_x",
    "train_y",
    "valid_x",
    "valid_y",
    "num_classes",
    "name",
])

def load_cifar10_dataset():
    from keras.datasets import cifar10
    (train_x, train_y), (valid_x, valid_y) = cifar10.load_data()

    # train_y.shape is 2d, (50000, 1). While Keras is smart enough to
    # handle this it's a good idea to flatten the array.
    train_y = train_y.reshape(-1)
    valid_y = valid_y.reshape(-1)

    return Dataset(
        train_x = train_x,
        train_y = train_y,
        valid_x = valid_x,
        valid_y = valid_y,
        num_classes = 10,
        name = "cifar10",
    )

def load_traffic_signs_dataset():
    with open("../data/train.p", "rb") as f:
        dataset = pickle.load(f)
        x, y, num_classes = dataset["features"], dataset["labels"], 43

    # Note that I believe stratify means to partition 30% for test
    # within each class.
    train_x, valid_x, train_y, valid_y = train_test_split(
        x, y, train_size = 0.70, stratify = y
    )

    return Dataset(
        train_x = train_x,
        train_y = train_y,
        valid_x = valid_x,
        valid_y = valid_y,
        num_classes = num_classes,
        name = "traffic_signs",
    )
