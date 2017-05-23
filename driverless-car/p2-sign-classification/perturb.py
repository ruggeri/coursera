import config
import dataset
import numpy as np
import pickle
import tensorflow as tf

BATCH_SIZE = 1024
IMAGE_DIM = 32

def perturb_x(session, x, y):
    tf_x = tf.placeholder(
        tf.float32, (None, *x.shape[1:]), name = "x"
    )

    tf_angles = tf.placeholder(
        tf.float32, (None,), name = "angles"
    )
    tf_rotated_x = tf.contrib.image.rotate(
        tf_x,
        tf_angles,
    )

    tf_new_dimension = tf.placeholder(
        tf.int32, (), name = "new_dimension"
    )
    tf_zoomed_x = tf.image.resize_image_with_crop_or_pad(
        tf.image.resize_bilinear(
            tf_x,
            size = (tf_new_dimension, tf_new_dimension)
        ),
        target_height = IMAGE_DIM,
        target_width = IMAGE_DIM
    )

    rotated_xs = []
    zoomed_xs = []

    _, batches = dataset.build_batches(x, y, batch_size = BATCH_SIZE)
    for (batch_x, _) in batches:
        angles = np.random.uniform(
            -config.MAX_ROTATION,
            config.MAX_ROTATION,
            size = len(batch_x)
        )
        rotated_xs.append(
            session.run(tf_rotated_x, feed_dict = {
                tf_x: batch_x,
                tf_angles: angles,
            })
        )

        new_dimension = int(
            IMAGE_DIM * np.random.uniform(
                1.0 - config.MAX_ZOOM,
                1.0 + config.MAX_ZOOM
            )
        )
        zoomed_xs.append(
            session.run(tf_zoomed_x, feed_dict = {
                tf_x: batch_x,
                tf_new_dimension: new_dimension
            })
        )

    return (
        np.concatenate(rotated_xs, axis = 0),
        np.concatenate(zoomed_xs, axis = 0),
    )

with tf.Session() as session:
    training_file = f"{config.DATA_DIR}/train.p"
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    X_train, y_train = train["features"], train["labels"]
    # Need to shuffle because some perturbations will be constant per
    # batch.
    X_train, y_train = dataset.shuffle(X_train, y_train)

    rotated_x, zoomed_x = perturb_x(session, X_train, y_train)
    print(rotated_x.shape)
    print(zoomed_x.shape)

    X_train_augmented = np.concatenate(
        [X_train, rotated_x, zoomed_x],
        axis = 0
    )
    y_train_augmented = np.concatenate(
        [y_train, y_train, y_train],
        axis = 0
    )

    augmented_training_file = f"/{config.DATA_DIR}/train_augmented.p"
    with open(augmented_training_file, "wb") as f:
        pickle.dump({
            "features": X_train_augmented,
            "labels": y_train_augmented,
        }, f)
