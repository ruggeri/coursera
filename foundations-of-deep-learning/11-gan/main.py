import batch as batch_module
from collections import namedtuple
import config
import graph as graph_module
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

RunInfo = namedtuple("RunInfo", [
    "session",
    "graph",
    "dataset",
    "writer",
])

def run_discriminator_batch(run_info, batch):
    session, graph = run_info.session, run_info.graph
    discriminator = graph.discriminator

    _, loss, accuracy = session.run([
        discriminator.train_op,
        discriminator.loss,
        discriminator.accuracy,
    ], feed_dict = {
        discriminator.all_class_label: np.concatenate([
            batch.fake_class_label,
            batch.real_class_label
        ], axis = 0),
        discriminator.all_x: np.concatenate([
            batch.fake_x,
            batch.real_x,
        ], axis = 0),
        discriminator.all_authenticity_label: np.concatenate([
            np.zeros(config.BATCH_SIZE, dtype = np.int64),
            np.ones(config.BATCH_SIZE, dtype = np.int64)
        ], axis = 0),
    })

    return (loss, accuracy)

def run_generator_batch(run_info, batch):
    session, graph = run_info.session, run_info.graph

    _, loss, summary = session.run([
        graph.generator.train_op,
        graph.generator.loss,
        graph.generator.summary
    ], feed_dict = {
        graph.generator.class_label: batch.fake_class_label,
        graph.generator.z: batch.fake_z,
    })
    run_info.writer.add_summary(summary)

    return loss

def should_log(run_info, batch_idx):
    num_batches = run_info.dataset.num_examples // config.BATCH_SIZE
    return (
        (batch_idx % int(config.LOG_FREQUENCY * num_batches)) == 0
    )

def run_batch(run_info, epoch_idx, batch_idx):
    batch = batch_module.next_batch(
        run_info,
        batch_size = config.BATCH_SIZE
    )

    d_loss, d_accuracy = run_discriminator_batch(run_info, batch)
    g_loss = run_generator_batch(run_info, batch)

    if should_log(run_info, batch_idx):
        print(f"Epoch {epoch_idx:03d} | Batch {batch_idx:03d} | "
              f"Gen Loss {g_loss:.2f} | "
              f"Dis Loss {d_loss:.2f} | "
              f"Dis Acc {(100 * d_accuracy):.1f}%")

def run_epoch(run_info, epoch_idx):
    num_batches = run_info.dataset.num_examples // config.BATCH_SIZE
    for batch_idx in range(1, num_batches + 1):
        run_batch(
            run_info,
            epoch_idx = epoch_idx,
            batch_idx = batch_idx,
        )

def show_samples(run_info, epoch_idx):
    if not (epoch_idx % config.EPOCHS_PER_SAMPLING == 0):
        return

    batch = batch_module.next_batch(run_info, config.NUM_SAMPLES_TO_SAVE)
    for idx, fake_x in enumerate(batch.fake_x):
        plt.imshow(fake_x.reshape((28, 28)), cmap = "Greys_r")
        plt.title(f"class_label: {batch.fake_class_label[idx]}")
        plt.savefig(f"samples/sample_e{epoch_idx:03d}_{idx:03d}.png")
        plt.close()

def run(session,
        graph,
        num_epochs = config.NUM_EPOCHS,
        epoch_callback = None):
    ri = run_info(session, graph)
    session.run(tf.global_variables_initializer())
    for epoch_idx in range(1, num_epochs + 1):
        run_epoch(
            run_info = ri,
            epoch_idx = epoch_idx,
        )
        if epoch_callback:
            epoch_callback(ri, epoch_idx)

def run_info(session, graph):
    from tensorflow.examples.tutorials.mnist import input_data
    dataset = input_data.read_data_sets('mnist_data').train
    writer = tf.summary.FileWriter("logs/", graph = session.graph)

    return RunInfo(
        session = session,
        graph = graph,
        dataset = dataset,
        writer = writer
    )

def graph():
    g = graph_module.graph(graph_module.NetworkConfiguration(
        num_classes = config.NUM_CLASSES,
        num_discriminator_hidden_units = config.NUM_DISCRIMINATOR_HIDDEN_UNITS,
        num_generator_hidden_units = config.NUM_GENERATOR_HIDDEN_UNITS,
        num_x_dims = config.IMAGE_DIMS,
        num_z_dims = config.Z_DIMS,
    ))

    return g

if __name__ == "__main__":
    with tf.Session() as session:
        run(session, graph(), epoch_callback = show_samples)
