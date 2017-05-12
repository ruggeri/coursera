import config
import graph
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def generate_samples(session, graph, one_hot_class_label, z_dims):
    num_samples = one_hot_class_label.shape[0]
    z = np.random.uniform(size = [num_samples, z_dims])

    return session.run(graph.generated_x, feed_dict = {
        graph.one_hot_class_label: one_hot_class_label,
        graph.z: z,
    })

def run_discriminator_batch(session, graph, x, one_hot_class_label):
    num_samples = x.shape[0]

    # For each true example, generate a fake one.
    generated_x = generate_samples(session, graph, one_hot_class_label)
    x = np.concatenate([x, generated_x], axis = 0)
    authenticity_label = np.concatenate(
        [tf.ones(num_samples), tf.zeros(num_samples)],
        axis = 0
    )

    _, loss, percentage = session.run([
        graph.train_discriminator_op,
        graph.discriminator_loss,
        graph.discriminator_percentage,
    ], feed_dict = {
        discriminator_x: x,
        authenticity_label: authenticity_label
    })

    return (loss, percentage)

def run_generator_batch(session, graph, batch_size, num_classes):
    one_hot_class_label = np.random.choice(
        num_classes, size = batch_size, replace = True
    )
    z = np.random.uniform(size = [batch_size, config.Z_DIMS])

    _, loss = session.run(
        [graph.train_generator_op, graph.generator_loss],
        feed_dict = {
            graph.one_hot_class_label: one_hot_class_label,
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
    x = x.reshape([-1, IMAGE_DIMENSION])
    one_hot_label = np.zeros(
        [config.BATCH_SIZE, config.NUM_CLASSES], dtype = np.float32
    )
    one_hot_label[:, class_label] = 1.0

    discriminator_loss, discriminator_accuracy = (
        run_discriminator_batch(
            session = session,
            graph = graph,
            x = x,
            one_hot_class_label = one_hot_class_label)
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
    g = graph.graph()
    dataset = input_data.read_data_sets('mnist_data')
    for epoch_idx in range(1, config.NUM_EPOCHS + 1):
        run_epoch(
            session = session,
            graph = graph,
            epoch_idx = epoch_idx,
            dataset = dataset
        )
