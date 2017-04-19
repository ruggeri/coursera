import config
import numpy as np
import pickle
import random

#one_hot_labels = np.zeros((len(labels), len(labels_map)))
#for label_idx, label in enumerate(labels):
#    one_hot_labels[label_idx, label] = 1.0


class Dataset:
    def __init__(self):
        self.codes_ = None
        self.labels_ = None
        self.labels_map_ = None
        self.did_perform_train_test_split_ = False

    def codes(self):
        if self.codes_ is None:
            self.codes_ = np.load(config.CODES_FILENAME)
        return self.codes_

    def labels(self):
        if self.labels_ is None:
            labels_ints = np.load(config.LABELS_FILENAME)
            self.labels_ = np.zeros(
                (len(labels_ints), self.num_labels())
            )
            for idx, label_int in enumerate(labels_ints):
                self.labels_[idx, label_int] = 1.0

        return self.labels_

    def labels_map(self):
        if self.labels_map_ is None:
            with open(config.LABELS_MAP_FILENAME, "rb") as f:
                self.labels_map_ = pickle.load(f)
        return self.labels_map_

    def num_code_units(self):
        return self.codes().shape[1]

    def num_labels(self):
        return len(self.labels_map())

    def perform_train_test_split(self):
        idxs = list(range(self.codes().shape[0]))
        random.shuffle(idxs)

        num_train_examples = int(config.TRAINING_FRACTION * len(idxs))
        self.training_codes_ = self.codes()[
            idxs[:num_train_examples]
        ]
        self.training_labels_ = self.labels()[
            idxs[:num_train_examples]
        ]
        self.validation_codes_ = self.codes()[
            idxs[num_train_examples:]
        ]
        self.validation_labels_ = self.labels()[
            idxs[num_train_examples:]
        ]

        self.did_perform_train_test_split_ = True

    def dataset(self, name):
        if not self.did_perform_train_test_split_:
            self.perform_train_test_split()

        if name == "training":
            return (self.training_codes_, self.training_labels_)
        elif name == "validation":
            return (self.validation_codes_, self.validation_labels_)
        else:
            raise Exception(f"unknown training set: {name}")

    def get_batches(self, name):
        codes, labels = self.dataset(name)
        num_batches = codes.shape[0] // config.TRAINING_BATCH_SIZE

        def batch_generator():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * config.TRAINING_BATCH_SIZE
                end_idx = min(
                    codes.shape[0],
                    start_idx + config.TRAINING_BATCH_SIZE
                )

                yield (
                    codes[start_idx:end_idx],
                    labels[start_idx:end_idx]
                )

        return (num_batches, batch_generator())
