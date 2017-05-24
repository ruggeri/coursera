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
    "valid_y"
])

def traffic_dataset():
    with open("../data/train.p", "rb") as f:
        dataset = pickle.load(f)
        x, y = dataset["features"], dataset["labels"]

    # Note that I believe stratify means to partition 30% for test
    # within each class.
    train_x, valid_x, train_y, valid_y = train_test_split(
        x, y, train_size = 0.70, stratify = y
    )

    return Dataset(
        train_x = train_x,
        train_y = train_y,
        valid_x = valid_x,
        valid_y = valid_y
    )

Network = namedtuple("Network", [
    "x",
    "y",
    "accuracy",
    "loss",
    "train_op"
])

def build_network():
    x = tf.placeholder(tf.float32, (None, 32, 32, 3), name = "x")
    resized_x = tf.image.resize_images(
        x,
        ALEX_NET_IMG_DIM
    )
    y = tf.placeholder(tf.int64, (None), name = "y")
    one_hot_y = tf.one_hot(y, 43)

    fc7 = alexnet.AlexNet(resized_x, feature_extract=True)
    fc7 = tf.stop_gradient(fc7)

    logits = tf.layers.dense(fc7, 43, activation = None)
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
        loss = loss,
        train_op = train_op
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

with tf.Session() as session:
    dataset = traffic_dataset()
    network = build_network()

    session.run(tf.global_variables_initializer())
    for epoch_idx in range(5):
        run_training_epoch(session, network, epoch_idx, dataset)
        run_validation(session, network, dataset)
