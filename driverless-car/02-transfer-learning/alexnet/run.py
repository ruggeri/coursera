import alexnet
from collections import namedtuple
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf

# This is my work.

ALEX_NET_IMG_DIM = (227, 227)
BATCH_SIZE = 64
LEARNING_RATE = 0.01

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

def load_dataset():
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

Network = namedtuple("Network", [
    "x",
    "y",
    "bottleneck",
    "accuracy",
    "loss",
    "train_op",
    "name",
])

def build_network(num_classes):
    x = tf.placeholder(tf.float32, (None, 32, 32, 3), name = "x")
    resized_x = tf.image.resize_images(
        x,
        ALEX_NET_IMG_DIM
    )
    y = tf.placeholder(tf.int64, (None), name = "y")
    one_hot_y = tf.one_hot(y, num_classes)

    fc7 = alexnet.AlexNet(resized_x, feature_extract=True)
    fc7 = tf.stop_gradient(fc7)

    logits = tf.layers.dense(fc7, num_classes, activation = None)
    probs = tf.nn.softmax(logits)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(
            tf.argmax(logits, axis = 1),
            tf.argmax(one_hot_y, axis = 1)
        ), tf.float32
    ))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = one_hot_y,
        logits = logits
    ))
    train_op = tf.train.AdamOptimizer(
        learning_rate = LEARNING_RATE
    ).minimize(loss)

    return Network(
        x = x,
        y = y,
        accuracy = accuracy,
        bottleneck = fc7,
        loss = loss,
        train_op = train_op,
        name = "alexnet",
    )

def run_training_batch(session, network, batch_x, batch_y):
    _, loss_val, accuracy_val = session.run(
        [network.train_op, network.loss, network.accuracy],
        feed_dict = {
            network.x: batch_x,
            network.y: batch_y
        }
    )

    return loss_val, accuracy_val

def run_validation_batch(session, network, batch_x, batch_y):
    loss_val, accuracy_val = session.run(
        [network.loss, network.accuracy],
        feed_dict = {
            network.x: batch_x,
            network.y: batch_y
        }
    )

    return loss_val, accuracy_val

def save_bottleneck_features(session, network, dataset):
    def transform(name, x, y):
        num_batches, batches = make_batches(x, y)

        transformed_results = []
        for batch_idx, (batch_x, _) in enumerate(batches):
            transformed_results.append(
                session.run(network.bottleneck, feed_dict = {
                    network.x: batch_x
                })
            )

            print(f"{name}: {batch_idx}/{num_batches}")

        return np.concatenate(transformed_results, axis = 0)

    transform_train_x = transform(
        "train_x", dataset.train_x, dataset.train_y
    )
    transform_valid_x = transform(
        "valid_x", dataset.valid_x, dataset.valid_y
    )

    fname = f"../data/bottleneck_{network.name}_{dataset.name}.p"
    with open(fname, "wb") as f:
        pickle.dump({
            "train_x": transform_train_x,
            "train_y": dataset.train_y,
            "valid_x": transform_valid_x,
            "valid_x": dataset.valid_y,
        }, f)

def make_batches(x, y):
    num_batches = x.shape[0] // BATCH_SIZE

    def helper():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, x.shape[0])
            yield (x[start_idx:end_idx], y[start_idx:end_idx])

    return num_batches, helper()

def run_training_epoch(session, network, epoch_idx, dataset):
    num_batches, batches = make_batches(
        dataset.train_x, dataset.train_y
    )

    for batch_idx, (batch_x, batch_y) in enumerate(batches):
        loss_val, accuracy_val = run_training_batch(
            session,
            network,
            batch_x,
            batch_y
        )

        print(f"E {epoch_idx} | B {batch_idx}/{num_batches} | "
              f"Loss: {loss_val:.3f} | Accuracy: {accuracy_val:.3f}")

def run_validation(session, network, dataset):
    num_batches, batches = make_batches(
        dataset.valid_x, dataset.valid_y
    )

    loss_val, accuracy_val = 0.0, 0.0
    for batch_idx, (batch_x, batch_y) in enumerate(batches):
        batch_loss_val, batch_accuracy_val = run_validation_batch(
            session,
            network,
            batch_x,
            batch_y
        )

        loss_val += batch_loss_val
        accuracy_val += batch_accuracy_val

    loss_val /= num_batches
    accuracy_val /= num_batches

    print(f"Valid loss: {loss_val:.3f} | "
          f"Valid accuracy: {accuracy_val:3f}")

def train_without_bottleneck(session, dataset, network):
    for epoch_idx in range(5):
        run_training_epoch(session, network, epoch_idx, dataset)
        run_validation(session, network, dataset)

with tf.Session() as session:
    dataset = load_cifar10_dataset()
    network = build_network(dataset.num_classes)

    session.run(tf.global_variables_initializer())
    save_bottleneck_features(session, network, dataset)
