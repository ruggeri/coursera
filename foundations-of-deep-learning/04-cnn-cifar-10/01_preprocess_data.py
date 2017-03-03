import numpy as np
import pickle

# Normalize all the feature data to be in the range [0, 1].
def normalize(x):
    result = (x - x.min()) / (x.max() - x.min())
    return result

def one_hot_encode(y):
    result = np.zeros((len(y), 10))
    for idx, label in enumerate(y):
        result[idx][label] = 1.0
    return result

# Preprocess some data and dump it out into a pickle file.
def preprocess_and_save_batch(features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    filename = f"pickle-files/{filename}"
    pickle.dump((features, labels), open(filename, 'wb'))

def unpack_batch_data(batch_data):
    # Looks like the data starts out flattened. Also it looks like
    # it's three image maps, rather than one 32x32 image with 3
    # channels per pixel.
    num_examples = len(batch_data['data'])
    features = (batch_data['data'].reshape(num_examples, 3, 32, 32))
    features = features.transpose(0, 2, 3, 1)
    labels = batch_data['labels']

    return features, labels

CIFAR_10_DATASET_FOLDER_PATH = 'cifar-10-batches-py'
# Read in a batch worth of data from the CIFAR dataset.
def load_cifar10_batch(batch_id):
    batch_fname = (
        CIFAR_10_DATASET_FOLDER_PATH + f"/data_batch_{batch_id+1}"
    )
    with open(batch_fname, mode='rb') as file:
        batch_data = pickle.load(file, encoding='latin1')

    return unpack_batch_data(batch_data)

NUM_BATCHES = 5
# Go through each batch, normalizing and encoding, pickling, and
# writing it out again.
def preprocess_and_save_data():
    valid_features = []
    valid_labels = []

    for batch_i in range(0, NUM_BATCHES):
        features, labels = load_cifar10_batch(batch_i)
        validation_count = int(len(features) * 0.1)

        # Prprocess and save a batch of training data
        preprocess_and_save_batch(
            features[:-validation_count],
            labels[:-validation_count],
            'preprocess_batch_' + str(batch_i) + '.p'
        )

        # Use a portion of each training batch for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    preprocess_and_save_batch(
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation.p'
    )

    test_batch_fname = CIFAR_10_DATASET_FOLDER_PATH + '/test_batch'
    with open(test_batch_fname, mode='rb') as file:
        test_batch_data = pickle.load(file, encoding='latin1')

    # load the training data
    test_features, test_labels = unpack_batch_data(test_batch_data)

    # Preprocess and save all training data
    preprocess_and_save_batch(
        np.array(test_features),
        np.array(test_labels),
        'preprocess_training.p'
    )

preprocess_and_save_data()
