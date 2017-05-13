import batch as batch_module
import config
import graph as graph_module
import numpy as np
import tensorflow as tf

def run_discriminator_batch(session, graph, batch):
    d = graph.discriminator
    _, loss, accuracy = session.run([
        d.train_op,
        d.loss,
        d.accuracy,
    ], feed_dict = {
        d.class_label: batch.combined_class_label,
        d.x: batch.combined_x,
        d.authenticity_label: batch.combined_authenticity_label
    })

    return (loss, accuracy)

def run_generator_batch(session, graph, batch):
    _, loss = session.run(
        [graph.generator.train_op, graph.generator.loss],
        feed_dict = {
            graph.generator.class_label: batch.fake_class_label,
            graph.generator.z: batch.fake_z,
        }
    )

    return loss

def run_batch(session, graph, epoch_idx, batch_idx, dataset):
    batch = batch_module.next_batch(
        session = session,
        graph = graph,
        dataset = dataset,
        batch_size = config.BATCH_SIZE
    )

    run_discriminator_batch(
        session = session,
        graph = graph,
        batch = batch,
    )
    run_generator_batch(
        session = session,
        graph = graph,
        batch = batch,
    )

    num_batches = dataset.num_examples // config.BATCH_SIZE
    should_log = ((
        batch_idx % int(config.LOG_FREQUENCY * num_batches)) == 0
    )
    if should_log:
        g_loss, d_loss, d_accuracy = evaluate(
            session, graph, dataset, config.BATCH_SIZE
        )
        print(f"Epoch {epoch_idx:03d} | Batch {batch_idx:03d} | "
              f"Gen Loss {g_loss:.2f} | "
              f"Dis Loss {d_loss:.2f} | "
              f"Dis Acc {(100 * d_accuracy):.1f}%")

def evaluate(session, graph, dataset, batch_size):
    batch = batch_module.next_batch(
        session, graph, dataset, batch_size
    )

    g_loss = session.run(
        graph.generator.loss,
        feed_dict = {
            graph.generator.class_label: batch.fake_class_label,
            graph.generator.z: batch.fake_z,
        }
    )
    d_loss, d_accuracy = session.run([
        graph.discriminator.loss,
        graph.discriminator.accuracy,
    ], feed_dict = {
        graph.discriminator.class_label: batch.combined_class_label,
        graph.discriminator.x: batch.combined_x,
        graph.discriminator.authenticity_label: batch.combined_authenticity_label,
    })

    return (g_loss, d_loss, d_accuracy)

def run_epoch(session, graph, epoch_idx, dataset):
    num_batches = dataset.num_examples // config.BATCH_SIZE
    for batch_idx in range(1, num_batches + 1):
        run_batch(
            session = session,
            graph = graph,
            epoch_idx = epoch_idx,
            batch_idx = batch_idx,
            dataset = dataset
        )

def run(session,
        graph,
        num_epochs = config.NUM_EPOCHS,
        epoch_callback = None):
    from tensorflow.examples.tutorials.mnist import input_data
    dataset = input_data.read_data_sets('mnist_data').train
    writer = tf.summary.FileWriter("logs/", graph = session.graph)

    session.run(tf.global_variables_initializer())
    for epoch_idx in range(1, num_epochs + 1):
        run_epoch(
            session = session,
            graph = graph,
            epoch_idx = epoch_idx,
            dataset = dataset,
        )
        if epoch_callback:
            epoch_callback(session, graph)

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
        run(session, graph())
