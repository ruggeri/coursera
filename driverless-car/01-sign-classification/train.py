from collections import namedtuple
import config
import dataset as dataset_mod
import network as network_mod
import numpy as np
import tensorflow as tf
import time

TrainingSession = namedtuple("TrainingSession", [
    "session",
    "network",
    "dataset",
    "file_writer",
    "saver",
    "learning_rate",
])

def train_batch(ts, x, y):
    _, cost, accuracy, summary = ts.session.run([
        ts.network.train_op,
        ts.network.cost,
        ts.network.accuracy,
        ts.network.summary,
    ], feed_dict = {
        ts.network.x: x,
        ts.network.y: y,
        ts.network.keep_prob: config.KEEP_PROB,
        ts.network.learning_rate: ts.learning_rate,
        ts.network.training: True
    })

    ts.file_writer.add_summary(summary)

    return (cost, accuracy)

def run_validation(ts):
    # I use as large a batch size as possible to make the most of the
    # hardware and evaluate quickly.
    num_batches, batches = dataset_mod.build_batches(
        ts.dataset.X_valid,
        ts.dataset.y_valid,
        batch_size = 1024
    )

    cost, accuracy = 0.0, 0.0
    for batch_x, batch_y in batches:
        batch_cost, batch_accuracy = ts.session.run([
            ts.network.cost,
            ts.network.accuracy
        ], feed_dict = {
            ts.network.x: batch_x,
            ts.network.y: batch_y,
            ts.network.keep_prob: 1.0,
            ts.network.training: False
        })

        cost += batch_cost
        accuracy += batch_accuracy
    cost /= num_batches
    accuracy /= num_batches

    return cost, accuracy

def train_epoch(ts, epoch_idx):
    # Must subtract one from epoch_idx, which is one indexed
    batch_size = config.BATCH_SIZES[epoch_idx - 1]
    num_batches, batches = dataset_mod.build_batches(
        ts.dataset.X_train,
        ts.dataset.y_train,
        batch_size = batch_size
    )

    prev_train_cost = np.inf
    train_cost, train_accuracy = 0.0, 0.0
    prev_time = time.time()
    for batch_idx, (batch_x, batch_y) in enumerate(batches, 1):
        train_batch_cost, train_batch_accuracy = train_batch(
            ts, batch_x, batch_y
        )

        train_cost += train_batch_cost
        train_accuracy += train_batch_accuracy

        # -1 because epoch_idx is one indexed
        batches_per_logging = config.BATCHES_PER_LOGGING[epoch_idx - 1]
        if batch_idx % batches_per_logging  == 0:
            train_cost /= batches_per_logging
            train_accuracy /= batches_per_logging
            valid_cost, valid_accuracy = run_validation(ts)

            print(f"E {epoch_idx} | B {batch_idx}/{num_batches} | "
                  f"Train Cost {train_cost:0.3f} | "
                  f"Train Acc  {100*train_accuracy:3.1f}% | "
                  f"Valid Cost {valid_cost:0.3f} | "
                  f"Valid Acc {100*valid_accuracy:3.1f}% | "
                  f"Learning Rate {ts.learning_rate:1.2E}")

            current_time = time.time()
            examples_per_second = (
                (batch_size * batches_per_logging)
                / (current_time - prev_time)
            )
            print(f"Ex/sec: {examples_per_second:0.1f}")
            prev_time = current_time

            if train_cost > prev_train_cost:
                ts = decay_learning_rate(ts)
            prev_train_cost = train_cost
            train_cost, train_accuracy = 0.0, 0.0

    print("Saving model!")
    ts.saver.save(ts.session, "models/model.ckpt")
    print("Model saved!")

    return ts

def decay_learning_rate(ts):
    return TrainingSession(
        session = ts.session,
        network = ts.network,
        dataset = ts.dataset,
        file_writer = ts.file_writer,
        saver = ts.saver,
        learning_rate = ts.learning_rate * config.LEARNING_RATE_DECAY
    )

def train(session, dataset):
    network = network_mod.build_network(
        dataset.image_shape,
        dataset.num_classes
    )

    session.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter("logs/", graph = session.graph)
    saver = tf.train.Saver()

    ts = TrainingSession(
        session = session,
        network = network,
        dataset = dataset,
        file_writer = file_writer,
        saver = saver,
        learning_rate = config.INITIAL_LEARNING_RATE,
    )

    for epoch_idx in range(1, config.NUM_EPOCHS + 1):
        ts = train_epoch(ts, epoch_idx)

def main():
    with tf.Session() as session:
        dataset = dataset_mod.load()
        train(session, dataset)

if __name__ == "__main__":
    main()
