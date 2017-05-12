import config
import graph
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def generate_samples(session, graph, class_label):
    num_samples = class_label.shape[0]
    z = np.random.uniform(size = [num_samples, config.Z_DIMS])

    return session.run(graph.generated_x, feed_dict = {
        graph.class_label: class_label,
        graph.z: z,
    })

def run_discriminator_batch(session, graph, x, class_label):
    num_samples = x.shape[0]

    # For each true example, generate a fake one.
    generated_x = generate_samples(
        session = session,
        graph = graph,
        class_label = class_label
    )

    # Concatenate everything
    x = np.concatenate([x, generated_x], axis = 0)
    class_label = np.concatenate([class_label, class_label], axis = 0)
    authenticity_label = np.concatenate(
        [np.ones(num_samples), np.zeros(num_samples)],
        axis = 0
    )

    _, loss, accuracy = session.run([
        graph.train_discriminator_op,
        graph.discriminator_loss,
        graph.discriminator_accuracy,
    ], feed_dict = {
        graph.class_label: class_label,
        graph.discriminator_x: x,
        graph.authenticity_label: authenticity_label
    })

    return (loss, accuracy)

def run_generator_batch(session, graph, batch_size, num_classes):
    class_label = np.random.choice(
        num_classes, size = batch_size, replace = True
    )
    z = np.random.uniform(size = [batch_size, config.Z_DIMS])

    _, loss = session.run(
        [graph.train_generator_op, graph.generator_loss],
        feed_dict = {
            graph.class_label: class_label,
            graph.z: z,
        }
    )

    return loss

def run_batch(session, graph, epoch_idx, batch_idx, dataset):
    generator_loss = run_generator_batch(
        session = session,
        graph = graph,
        batch_size = config.BATCH_SIZE,
        num_classes = config.NUM_CLASSES,
    )

    x, class_label = dataset.next_batch(config.BATCH_SIZE)
    x = x.reshape([-1, config.IMAGE_DIMS])

    discriminator_loss, discriminator_accuracy = (
        run_discriminator_batch(
            session = session,
            graph = graph,
            x = x,
            class_label = class_label
        )
    )

    print(f"Epoch {epoch_idx:02d} | Batch {batch_idx:02d} | "
          f"Gen Loss {generator_loss:.2f} | "
          f"Dis Loss {discriminator_loss:.2f} | "
          f"Dis Acc {(100 * discriminator_accuracy):.1f}%")

def run_epoch(session, graph, epoch_idx, dataset):
    for batch_idx in range(1, config.NUM_BATCHES + 1):
        run_batch(
            session = session,
            graph = graph,
            epoch_idx = epoch_idx,
            batch_idx = batch_idx,
            dataset = dataset
        )

def run(session):
    g = graph.graph(
        num_classes = config.NUM_CLASSES,
        x_dims = config.IMAGE_DIMS,
        z_dims = config.Z_DIMS,
        num_generator_hidden_units = config.NUM_GENERATOR_HIDDEN_UNITS,
        num_discriminator_hidden_units = config.NUM_DISCRIMINATOR_HIDDEN_UNITS,
    )
    dataset = input_data.read_data_sets('mnist_data').train

    session.run(tf.global_variables_initializer())
    for epoch_idx in range(1, config.NUM_EPOCHS + 1):
        run_epoch(
            session = session,
            graph = g,
            epoch_idx = epoch_idx,
            dataset = dataset
        )

if __name__ == "__main__":
    with tf.Session() as session:
        run(session)
