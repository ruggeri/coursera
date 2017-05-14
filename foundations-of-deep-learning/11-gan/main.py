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
        discriminator.fake_class_label: batch.fake_class_label,
        discriminator.fake_x: batch.fake_x,
        discriminator.real_class_label: batch.real_class_label,
        discriminator.real_x: batch.real_x,
    })

    return (loss, accuracy)

def run_generator_batch(run_info, batch):
    session, graph = run_info.session, run_info.graph

    _, loss = session.run(
        [graph.generator.train_op, graph.generator.loss],
        feed_dict = {
            graph.generator.class_label: batch.fake_class_label,
            graph.generator.z: batch.fake_z,
        }
    )

    return loss

def run_batch(run_info, epoch_idx, batch_idx):
    batch = batch_module.next_batch(
        run_info,
        batch_size = config.BATCH_SIZE
    )

    run_discriminator_batch(
        run_info = run_info,
        batch = batch,
    )
    run_generator_batch(
        run_info,
        batch = batch,
    )

    num_batches = run_info.dataset.num_examples // config.BATCH_SIZE
    should_log = ((
        batch_idx % int(config.LOG_FREQUENCY * num_batches)) == 0
    )
    if should_log:
        g_loss, d_loss, d_accuracy = evaluate(
            run_info, config.BATCH_SIZE
        )
        print(f"Epoch {epoch_idx:03d} | Batch {batch_idx:03d} | "
              f"Gen Loss {g_loss:.2f} | "
              f"Dis Loss {d_loss:.2f} | "
              f"Dis Acc {(100 * d_accuracy):.1f}%")

def evaluate(run_info, batch_size):
    session, graph = run_info.session, run_info.graph
    generator, discriminator = graph.generator, graph.discriminator

    batch = batch_module.next_batch(
        run_info, batch_size
    )

    g_loss, g_summary = session.run(
        [generator.loss, generator.summary],
        feed_dict = {
            generator.class_label: batch.fake_class_label,
            generator.z: batch.fake_z,
        }
    )
    run_info.writer.add_summary(g_summary)

    d_loss, d_accuracy = session.run([
        discriminator.loss,
        discriminator.accuracy,
    ], feed_dict = {
        discriminator.fake_class_label: batch.fake_class_label,
        discriminator.fake_x: batch.fake_x,
        discriminator.real_class_label: batch.real_class_label,
        discriminator.real_x: batch.real_x,
    })

    return (g_loss, d_loss, d_accuracy)

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

    batch = batch_module.next_batch(run_info, 8)
    for fake_x in batch.fake_x:
        plt.figure()
        plt.imshow(fake_x.reshape((28, 28)), cmap = "Greys_r")
    plt.show()

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
