import graph
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def generate_samples(graph, one_hot_class_label, z_dims):
    num_samples = one_hot_class_label.shape[0]
    z = np.random.uniform(size = [num_samples, z_dims])

    return session.run(graph.generated_x, feed_dict = {
        graph.one_hot_class_label: one_hot_class_label,
        graph.z: z,
    })

def run_discriminator_batch(graph, x, one_hot_class_label):
    num_samples = x.shape[0]

    # For each true example, generate a fake one.
    generated_x = generate_samples(graph, one_hot_class_label)
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

def run_generator_batch(graph, batch_size, num_classes):
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

def run():
    mnist_data = input_data.read_data_sets('mnist_data')
