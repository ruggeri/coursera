import batches as batches_mod
import dataset as dataset_mod
import numpy as np
import pickle
import tensorflow as tf

def save_bottleneck_features(session, network, dataset):
    def transform(name, x, y):
        num_batches, batches = batches_mod.make_batches(x, y)

        transformed_results = []
        for batch_idx, (batch_x, _) in enumerate(batches):
            transformed_results.append(
                session.run(network.bottleneck_out, feed_dict = {
                    network.x: batch_x
                })
            )

            print(f"{name}: {batch_idx}/{num_batches}")
            if batch_idx == 10: break

        return np.concatenate(transformed_results, axis = 0)

    transform_train_x = transform(
        "train_x", dataset.train_x, dataset.train_y
    )
    transform_valid_x = transform(
        "valid_x", dataset.valid_x, dataset.valid_y
    )

    fname = f"../data/bottleneck_{network.name}_{dataset.name}.p"
    with open(fname, "wb") as f:
        pickle.dump(dataset_mod.Dataset(
            train_x = transform_train_x,
            train_y = dataset.train_y,
            valid_x = transform_valid_x,
            valid_y = dataset.valid_y,
            num_classes = dataset.num_classes,
            name = dataset.name
        ), f)

def transform(dataset, network):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        save_bottleneck_features(session, network, dataset)
