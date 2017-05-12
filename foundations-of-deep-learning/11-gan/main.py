import config
import graph as graph_module
import numpy as np
import tensorflow as tf

def generate_z(num_samples):
    return np.random.uniform(-1, 1, size = [num_samples, config.Z_DIMS])

def generate_samples(session, graph, class_label):
    num_samples = class_label.shape[0]
    z = generate_z(num_samples)

    return session.run(graph.generator_x, feed_dict = {
        graph.class_label: class_label,
        graph.z: z,
    })

def run_discriminator_batch(session, graph, dataset, batch_size):
    x, class_label = dataset.next_batch(batch_size)
    # TODO: Trying unconditional to get that working first...
    class_label = np.ones_like(class_label)
    x = x.reshape([-1, config.IMAGE_DIMS])
    # Renormalize x to (-1, +1)
    x = (2 * x) - 1

    # For each true example, generate a fake one.
    generator_x = generate_samples(
        session = session,
        graph = graph,
        class_label = class_label
    )

    # Concatenate real and fake results
    x = np.concatenate([x, generator_x], axis = 0)
    class_label = np.concatenate([class_label, class_label], axis = 0)
    authenticity_label = np.concatenate(
        [np.ones(batch_size) * (1 - config.LABEL_SMOOTHING),
         np.zeros(batch_size)],
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
    z = generate_z(batch_size)

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

    discriminator_loss, discriminator_accuracy = (
        run_discriminator_batch(
            session = session,
            graph = graph,
            dataset = dataset,
            batch_size = config.BATCH_SIZE
        )
    )

    num_batches = dataset.num_examples // config.BATCH_SIZE
    should_log = ((
        batch_idx % int(config.LOG_FREQUENCY * num_batches)) == 0
    )
    if should_log:
        print(f"Epoch {epoch_idx:03d} | Batch {batch_idx:03d} | "
              f"Gen Loss {generator_loss:.2f} | "
              f"Dis Loss {discriminator_loss:.2f} | "
              f"Dis Acc {(100 * discriminator_accuracy):.1f}%")

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

def run(session, graph, num_epochs = config.NUM_EPOCHS, epoch_callback = None):
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
    g = graph_module.graph(
        num_classes = config.NUM_CLASSES,
        x_dims = config.IMAGE_DIMS,
        z_dims = config.Z_DIMS,
        num_generator_hidden_units = config.NUM_GENERATOR_HIDDEN_UNITS,
        num_discriminator_hidden_units = config.NUM_DISCRIMINATOR_HIDDEN_UNITS,
    )

    return g

if __name__ == "__main__":
    with tf.Session() as session:
        run(session, graph())
