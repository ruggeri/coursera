import config
import dataset
import matplotlib.pyplot as plt
import network as network_mod
import numpy as np
import tensorflow as tf
import time

# TODO: will have to scale real images to -1, +1.

def train_discriminator(session, network, real_x):
    fake_z = np.random.uniform(
        low = -1.0,
        high = +1.0,
        size = (config.BATCH_SIZE, config.Z_DIMS),
    )

    _, d_loss, d_accuracy = session.run([
        network.trainer.d_train_op,
        network.trainer.d_loss,
        network.trainer.accuracy
    ], feed_dict = {
        network.real_x: real_x,
        network.fake_z: fake_z,
    })

    return d_loss, d_accuracy

def train_generator(session, network, real_x):
    fake_z = np.random.uniform(
        low = -1.0,
        high = +1.0,
        size = (config.BATCH_SIZE, config.Z_DIMS),
    )

    _, g_loss, g_accuracy = session.run([
        network.trainer.g_train_op,
        network.trainer.g_loss,
        network.trainer.accuracy
    ], feed_dict = {
        network.real_x: real_x,
        network.fake_z: fake_z,
    })

    return g_loss, g_accuracy

def train_batch(session, network, real_x):
    d_loss, d_accuracy = train_discriminator(session, network, real_x)

    total_g_loss, total_g_accuracy = 0, 0
    for _ in range(config.GENERATOR_ROUND_MULTIPLIER):
        result = train_generator(session, network, real_x)
        total_g_loss += result[0]
        total_g_accuracy += result[1]
    g_loss = total_g_loss / config.GENERATOR_ROUND_MULTIPLIER
    g_accuracy = total_g_accuracy / config.GENERATOR_ROUND_MULTIPLIER

    return {
        "d_loss": d_loss,
        "d_accuracy": d_accuracy,

        "g_loss": g_loss,
        "g_accuracy": g_accuracy
    }

def log_batch_result(epoch_idx, batch_idx, result, prev_time):
    current_time = time.time()
    examples_per_sec = (
        (config.BATCHES_PER_LOG * config.BATCH_SIZE)
        / (current_time - prev_time)
    )

    print(f"E {epoch_idx} | B {batch_idx} | "
          f"D Loss {result['d_loss']:0.3f} | "
          f"D Accuracy {100 * result['d_accuracy']:03.1f}% | "
          f"G Loss {result['g_loss']:0.3f} | "
          f"G Accuracy {100 * result['g_accuracy']:03.1f}%")
    print(f"Ex/sec: {examples_per_sec:3.1f}")

def sample_generator_output(epoch_idx, batch_idx, session, network):
    fake_z = np.random.uniform(
        low = -1.0,
        high = +1.0,
        size = (config.NUM_SAMPLES_PER_SAMPLING, config.Z_DIMS)
    )

    samples = session.run(
        network.fake_x,
        feed_dict = {
            network.fake_z: fake_z
        }
    )

    for sample_idx, sample in enumerate(samples):
        fname = (
            f"sample_e{epoch_idx:02d}_b{batch_idx:04d}_{sample_idx:02d}"
        )

        # Convert to 8bit format.
        sample = ((sample + 1) / 2) * 255
        sample = sample.astype(np.uint8)

        if config.COLOR_DEPTH == 1:
            # imshow only likes to show 3d images if the third
            # dimension is 3 or 4. If it's 1, it just wants a 2d
            # image.
            sample = sample.reshape(config.IMAGE_DIMS[0:2])

        plt.imshow(sample, cmap = config.CMAP)
        plt.title(fname)
        plt.savefig(f"samples/{fname}.png")
        plt.close()

def train_epoch(session, network, epoch_idx, get_batches):
    prev_time = time.time()
    batches = get_batches(config.BATCH_SIZE)
    for batch_idx, real_x in enumerate(batches, 1):
        result = train_batch(session, network, real_x)

        if batch_idx % config.BATCHES_PER_LOG == 0:
            log_batch_result(
                epoch_idx,
                batch_idx,
                result,
                prev_time
            )
            prev_time = time.time()
        if batch_idx % config.BATCHES_PER_SAMPLING == 0:
            # Note: this will cause examples/sec to dip whenever we
            # sample because we'll spend extra time on that. I could
            # fix this, but meh.
            sample_generator_output(
                epoch_idx,
                batch_idx,
                session,
                network
            )

def train(session):
    network = network_mod.network()
    session.run(tf.global_variables_initializer())
    fw = tf.summary.FileWriter("logs/", graph = session.graph)

    get_batches = dataset.get_get_batches()
    for epoch_idx in range(1, config.NUM_EPOCHS + 1):
        train_epoch(session, network, epoch_idx, get_batches)

if __name__ == "__main__":
    with tf.Session() as session:
        # Turn interactive mode off because later we will be saving
        # images to the FS but don't want them to be shown using QT.
        plt.ioff()
        train(session)
