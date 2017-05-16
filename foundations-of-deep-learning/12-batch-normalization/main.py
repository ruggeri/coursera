import config
import network
import tensorflow as tf

def run_batch(session, network, image_data, label_data, is_training):
    if is_training:
        operations = [network.train_op, network.loss, network.accuracy]
    else:
        operations = [network.loss, network.accuracy]

    results = session.run(
        operations,
        feed_dict = {
            network.input_image: image_data,
            network.one_hot_digit_label: label_data,
            network.is_training: is_training
        }
    )

    return results[-2:]

def log_result(namespace, batch_idx, loss, accuracy):
    print(f"{namespace} | "
          f"Batch {batch_idx} | "
          f"Loss {loss:0.3f} | "
          f"Accuracy {100 * accuracy:0.1f}")

def run_training_batch(session, mnist, network):
    image_data, label_data = mnist.train.next_batch(config.BATCH_SIZE)
    loss, accuracy = run_batch(
        session = session,
        network = network,
        image_data = image_data,
        label_data = label_data,
        is_training = True
    )

    return loss, accuracy

def run_validation_batch(session, mnist, network):
    image_data, label_data = (
        mnist.validation.images, mnist.validation.labels
    )
    loss, accuracy = run_batch(
        session = session,
        network = network,
        image_data = image_data,
        label_data = label_data,
        is_training = False
    )

    return loss, accuracy

def run(session):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(
        "mnist_data/",
        one_hot = True,
        reshape = False
    )

    n = network.network()
    session.run(tf.global_variables_initializer())
    for batch_idx in range(1, config.NUM_BATCHES + 1):
        train_loss, train_accuracy = run_training_batch(
            session = session,
            mnist = mnist,
            network = n,
        )

        if batch_idx % 25 == 0:
            log_result(
                "training", batch_idx, train_loss, train_accuracy
            )
        if batch_idx % 100 == 0:
            valid_loss, valid_accuracy = run_validation_batch(
                session = session,
                mnist = mnist,
                network = n,
            )
            log_result(
                "validation", batch_idx, valid_loss, valid_accuracy
            )

with tf.Session() as session:
    run(session)
