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
def preprocess_and_save_segment(features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    filename = f"pickle-files/{filename}"
    pickle.dump((features, labels), open(filename, 'wb'))

def unpack_segment_data(segment_data):
    # Looks like the data starts out flattened. Also it looks like
    # it's three image maps, rather than one 32x32 image with 3
    # channels per pixel.
    num_examples = len(segment_data['data'])
    features = (segment_data['data'].reshape(num_examples, 3, 32, 32))
    features = features.transpose(0, 2, 3, 1)
    labels = segment_data['labels']

    return features, labels

# They call the individual files "batches" but I call them
# "segments". I'll set my own batch size later when training.
CIFAR_10_DATASET_FOLDER_PATH = 'cifar-10-batches-py'
# Read in a segment worth of data from the CIFAR dataset.
def load_cifar10_segment(segment_id):
    segment_fname = (
        f"{CIFAR_10_DATASET_FOLDER_PATH}/data_batch_{segment_id+1}"
    )
    with open(segment_fname, mode='rb') as file:
        segment_data = pickle.load(file, encoding='latin1')

    return unpack_segment_data(segment_data)

NUM_SEGMENTS = 5
# Go through each segment, normalizing and encoding, pickling, and
# writing it out again.
def preprocess_and_save_data():
    valid_features = []
    valid_labels = []

    for segment_i in range(0, NUM_SEGMENTS):
        features, labels = load_cifar10_segment(segment_i)
        validation_count = int(len(features) * 0.1)

        # Prprocess and save a segment of training data
        preprocess_and_save_segment(
            features[:-validation_count],
            labels[:-validation_count],
            f"preprocessed_segment_{segment_i}.p"
        )

        # Use a portion of each training segment for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    preprocess_and_save_segment(
        np.array(valid_features),
        np.array(valid_labels),
        "preprocessed_validation_data.p"
    )

    test_segment_fname = f"{CIFAR_10_DATASET_FOLDER_PATH}/test_batch"
    with open(test_segment_fname, mode='rb') as file:
        test_segment_data = pickle.load(file, encoding='latin1')

    # load the training data
    test_features, test_labels = unpack_segment_data(test_segment_data)

    # Preprocess and save all training data
    preprocess_and_save_segment(
        np.array(test_features),
        np.array(test_labels),
        "preprocessed_training_data.p"
    )

preprocess_and_save_data()
