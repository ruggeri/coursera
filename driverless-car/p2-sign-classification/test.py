import dataset as dataset_mod
import network as network_mod
import tensorflow as tf

def run_test(session, network, dataset):
    # I use as large a batch size as possible to make the most of the
    # hardware and evaluate quickly.
    num_batches, batches = dataset_mod.build_batches(
        dataset.X_test,
        dataset.y_test,
        batch_size = 1024
    )

    cost, accuracy = 0.0, 0.0
    for batch_x, batch_y in batches:
        batch_cost, batch_accuracy = session.run([
            network.cost,
            network.accuracy
        ], feed_dict = {
            network.x: batch_x,
            network.y: batch_y,
            network.keep_prob: 1.0,
            network.training: False
        })

        cost += batch_cost
        accuracy += batch_accuracy
    cost /= num_batches
    accuracy /= num_batches

    return cost, accuracy

def main():
    with tf.Session() as session:
        network = network_mod.restore(session)
        dataset = dataset_mod.load()
        cost, accuracy = run_test(session, network, dataset)

        print(f"Test Cost: {cost:0.3f} | Test Acc: {100*accuracy:3.1f}")

if __name__ == "__main__":
    main()
