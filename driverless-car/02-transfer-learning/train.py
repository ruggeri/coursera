import batches as batches_mod
import config
import tensorflow as tf

def run_training_batch(session, network, batch_x, batch_y):
    _, loss_val, accuracy_val = session.run(
        [network.train_op, network.loss, network.accuracy],
        feed_dict = {
            network.bottleneck_in: batch_x,
            network.y: batch_y
        }
    )

    return loss_val, accuracy_val

def run_validation_batch(session, network, batch_x, batch_y):
    loss_val, accuracy_val = session.run(
        [network.loss, network.accuracy],
        feed_dict = {
            network.bottleneck_in: batch_x,
            network.y: batch_y
        }
    )

    return loss_val, accuracy_val

def run_training_epoch(session, network, epoch_idx, dataset):
    num_batches, batches = batches_mod.make_batches(
        dataset.train_x, dataset.train_y
    )

    for batch_idx, (batch_x, batch_y) in enumerate(batches):
        loss_val, accuracy_val = run_training_batch(
            session,
            network,
            batch_x,
            batch_y
        )

        print(f"E {epoch_idx} | B {batch_idx}/{num_batches} | "
              f"Loss: {loss_val:.3f} | Accuracy: {accuracy_val:.3f}")

def run_validation(session, network, dataset):
    num_batches, batches = batches_mod.make_batches(
        dataset.valid_x, dataset.valid_y
    )

    loss_val, accuracy_val = 0.0, 0.0
    for batch_idx, (batch_x, batch_y) in enumerate(batches):
        batch_loss_val, batch_accuracy_val = run_validation_batch(
            session,
            network,
            batch_x,
            batch_y
        )

        loss_val += batch_loss_val
        accuracy_val += batch_accuracy_val

    loss_val /= num_batches
    accuracy_val /= num_batches

    print(f"Valid loss: {loss_val:.3f} | "
          f"Valid accuracy: {accuracy_val:3f}")

def train(dataset, network):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch_idx in range(config.NUM_EPOCHS):
            run_training_epoch(session, network, epoch_idx, dataset)
            run_validation(session, network, dataset)
