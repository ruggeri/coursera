import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# This is my work.

LEARNING_RATE = 0.01

# TODO: Load traffic signs data.
with open("train.p", "rb") as f:
    dataset = pickle.load(f)

# TODO: Split data into training and validation sets.
NUM_TRAINING_EXAMPLES = int(dataset["features"].shape[0] * 0.70)
train_x, train_y = (
    dataset["features"][:NUM_TRAINING_EXAMPLES],
    dataset["labels"][:NUM_TRAINING_EXAMPLES],
)
shuffle_idxs = np.random.permutation(train_x.shape[0])
train_x = train_x[shuffle_idxs]
train_x = train_x - np.mean(train_x)
train_y = train_y[shuffle_idxs]

valid_x, valid_y = (
    dataset["features"][NUM_TRAINING_EXAMPLES:],
    dataset["labels"][NUM_TRAINING_EXAMPLES:],
)
valid_x = valid_x - np.mean(valid_x)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3), name = "x")
resized_x = tf.image.resize_images(
    x,
    (227, 227)
)
y = tf.placeholder(tf.int64, (None), name = "y")
one_hot_y = tf.one_hot(y, 43)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized_x, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
logits = tf.layers.dense(
    fc7,
    43,
    activation = None
)
probs = tf.nn.softmax(logits)
accuracy = tf.reduce_mean(tf.cast(
    tf.equal(
        tf.argmax(logits, axis = 1),
        tf.argmax(one_hot_y, axis = 1)
    ), tf.float32
))

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels = one_hot_y,
    logits = logits
))
train_op = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss)

# TODO: Train and evaluate the feature extraction model.
def run_training_batch(session, batch_x, batch_y):
    _, loss_val, accuracy_val = session.run(
        [train_op, loss, accuracy], feed_dict = {
        x: batch_x,
        y: batch_y
    })

    return loss_val, accuracy_val

def run_validation_batch(session, batch_x, batch_y):
    loss_val, accuracy_val = session.run([loss, accuracy], feed_dict = {
        x: batch_x,
        y: batch_y
    })

    return loss_val, accuracy_val

BATCH_SIZE = 64
def run_training_epoch(session, epoch_idx):
    num_batches = train_x.shape[0] // BATCH_SIZE
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, train_x.shape[0])
        loss_val, accuracy_val = run_training_batch(
            session,
            train_x[start_idx:end_idx],
            train_y[start_idx:end_idx]
        )
        print(f"E {epoch_idx} | B {batch_idx}/{num_batches} | "
              f"Loss: {loss_val:.3f} | Accuracy: {accuracy_val:.3f}")

VALID_BATCH_SIZE = 256
# Something is broken with this...
def run_validation(session):
    num_batches = valid_x.shape[0] // VALID_BATCH_SIZE
    loss_val, accuracy_val = 0.0, 0.0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * VALID_BATCH_SIZE
        end_idx = min(start_idx + VALID_BATCH_SIZE, valid_x.shape[0])
        batch_loss_val, batch_accuracy_val = run_validation_batch(
            session,
            valid_x[start_idx:end_idx],
            valid_y[start_idx:end_idx]
        )

        loss_val += batch_loss_val
        accuracy_val += batch_accuracy_val

    loss_val /= num_batches
    accuracy_val /= num_batches

    print(f"Valid loss: {loss_val:.3f} | "
          f"Valid accuracy: {accuracy_val:3f}")

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch_idx in range(5):
        run_training_epoch(session, epoch_idx)
        run_validation(session)
