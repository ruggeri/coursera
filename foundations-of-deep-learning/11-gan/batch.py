from collections import namedtuple
import config
import numpy as np
import tensorflow as tf

Batch = namedtuple("Batch", [
    # Fake
    "fake_class_label",
    "fake_z",
    "fake_x",
    # Real
    "real_class_label",
    "real_x",
    # Combined
    "combined_x",
    "combined_class_label",
    "combined_authenticity_label",
])

def generate_z(num_samples):
    return np.random.uniform(-1, 1, size = [num_samples, config.Z_DIMS])

def generate_samples(session, graph, class_label, z = None):
    num_samples = class_label.shape[0]
    if z is None:
        z = generate_z(num_samples)

    return session.run(graph.generator_x, feed_dict = {
        graph.class_label: class_label,
        graph.z: z,
    })

def generate_fake_data(session, graph, num_classes, num_samples):
    fake_class_label = np.random.choice(
        num_classes + 1, size = num_samples, replace = True
    )
    # TODO: Trying unconditional to get that working first...
    fake_class_label = np.ones_like(fake_class_label)
    z = generate_z(num_samples)
    x = generate_samples(session, graph, fake_class_label, z)

    return (fake_class_label, z, x)

def generate_real_data(dataset, num_samples):
    real_x, real_class_label = dataset.next_batch(num_samples)
    # TODO: Trying unconditional to get that working first...
    real_class_label = np.ones_like(real_class_label)
    real_x = real_x.reshape([-1, config.IMAGE_DIMS])
    # Renormalize x to (-1, +1)
    real_x = (2 * real_x) - 1

    return (real_class_label, real_x)

def next_batch(session, graph, dataset, batch_size):
    fake_class_label, fake_z, fake_x = generate_fake_data(
        session,
        graph,
        num_classes = config.NUM_CLASSES,
        num_samples = batch_size,
    )
    real_class_label, real_x = generate_real_data(
        dataset = dataset ,
        num_samples = batch_size,
    )

    # Concatenate real and fake results
    combined_x = np.concatenate([real_x, fake_x], axis = 0)
    combined_class_label = np.concatenate(
        [real_class_label, fake_class_label], axis = 0
    )
    combined_authenticity_label = np.concatenate(
        [np.ones(batch_size) * (1 - config.LABEL_SMOOTHING),
         np.zeros(batch_size)],
        axis = 0
    )

    return Batch(
        fake_class_label = fake_class_label,
        fake_z = fake_z,
        fake_x = fake_x,

        real_class_label = real_class_label,
        real_x = real_x,

        combined_x = combined_x,
        combined_class_label = combined_class_label,
        combined_authenticity_label = combined_authenticity_label
    )
