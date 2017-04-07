# This shows how to setup a CNN with dropout in TensorFlow! I don't
# know that this CNN does any better than my two-layer basic FFNN, but
# it's just a demo of how to setup a CNN.
#
# With a batch size of 10, this does ~450ex/sec on my machine.

from collections import namedtuple
import math
import random
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
BENCHMARK_MODE = True
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
BATCH_SIZE = 10
KEEP_PROB = 0.90  # Keep_Prob, probability to keep units

# Constants
NUM_CLASSES = 10  # MNIST total classes (0-9 digits)
IMAGE_DIM = 28 # MNIST images are 28x28
MODEL_FNAME = "./model.ckpt"

def build_weights_and_biases():
    # Initial weights and biases.
    weights = {
        # 5x5 filter from 1 channel to 20 channels.
        'conv_weights1': tf.Variable(
            tf.random_normal([5, 5, 1, 20], stddev=1/math.sqrt(5*5))
        ),
        # 5x5 filter from 20 channels to 40 channels.
        'conv_weights2': tf.Variable(
            tf.random_normal(
                [5, 5, 20, 40], stddev=1/math.sqrt(20 * 25)
            )
        ),
        # Starts out as 28x28 input, but I do maxpooling with a stride
        # of 2 after both convolutional layers.
        'dense_weights1': tf.Variable(
            tf.random_normal([7*7*40, 100], stddev=1/math.sqrt(7*7*40))
        ),
        'out': tf.Variable(
            tf.random_normal(
                [100, NUM_CLASSES], stddev=1/math.sqrt(10*10)
            )
        )
    }

    biases = {
        'conv_biases1': tf.Variable(tf.ones([20])),
        'conv_biases2': tf.Variable(tf.ones([40])),
        'dense_biases1': tf.Variable(tf.ones([100])),
        'out': tf.Variable(tf.ones([NUM_CLASSES]))
    }

    return (weights, biases)

def build_placeholders():
    x = tf.placeholder(tf.float32, [None, IMAGE_DIM, IMAGE_DIM, 1])
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)

    return (x, y, keep_prob)

# Setup a basic convolutional layer.
def build_conv2d(x, W, b, strides = 1):
    layer = tf.nn.conv2d(
        x, W, strides=[1, strides, strides, 1], padding='SAME'
    )
    layer = tf.nn.bias_add(layer, b)
    layer = tf.nn.relu(layer)
    return layer

# Setup a basic maxpooling layer.
def build_maxpool2d(x, dim = 2):
    return tf.nn.max_pool(
        x,
        ksize = [1, dim, dim, 1],
        strides = [1, dim, dim, 1],
        padding = 'SAME'
    )

# This builds the CNN
Model = namedtuple("Model", "x, y, keep_prob, logits")
def build_model():
    x, y, keep_prob = build_placeholders()
    weights, biases = build_weights_and_biases()

    # Layer 1 - 28*28*1 to 14*14*20
    conv1 = build_conv2d(
        x, weights['conv_weights1'], biases['conv_biases1']
    )
    conv1 = build_maxpool2d(conv1, dim = 2)

    # Layer 2 - 14*14*20 to 7*7*40
    conv2 = build_conv2d(
        conv1, weights['conv_weights2'], biases['conv_biases2']
    )
    conv2 = build_maxpool2d(conv2, dim = 2)
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # Fully connected layer - 7*7*40 to 1024
    # First we flatten the 2D image to a 1D representation.
    num_prev_units = weights["dense_weights1"].get_shape().as_list()[0]
    fc1 = tf.reshape(conv2, [-1, num_prev_units])
    # Now we perform the layer.
    fc1 = tf.add(tf.matmul(
        fc1, weights['dense_weights1']), biases['dense_biases1']
    )
    fc1 = tf.nn.relu(fc1)
    # We use dropout here to regularize.
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output Layer.
    logits = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return Model(
        x = x,
        y = y,
        keep_prob = keep_prob,
        logits = logits
    )

Trainer = namedtuple("Trainer", "cost, optimizer, accuracy")
def build_trainer(model):
    # Define loss and optimizer
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits = model.logits, labels = model.y
        )
    )
    optimizer = tf.train.AdamOptimizer(
        learning_rate = LEARNING_RATE
    ).minimize(cost)

    # Calculate accuracy
    correct_pred = tf.equal(
        tf.argmax(model.logits, 1), tf.argmax(model.y, 1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return Trainer(
        cost = cost,
        optimizer = optimizer,
        accuracy = accuracy
    )

VALID_ACC_TEST_FREQ = 0.1
def should_perform_acc_test(batch_info):
    batches_per_acc_test = max(
        int(VALID_ACC_TEST_FREQ * batch_info.num_batches),
        1
    )

    return (batch_info.batch_idx + 1) % batches_per_acc_test == 0

EPOCH_TIME = None
def train_batch(session, model, trainer, batch_info):
    bi = batch_info

    batch_x, batch_y = bi.dataset.train.next_batch(BATCH_SIZE)

    session.run(trainer.optimizer, feed_dict = {
        model.x: batch_x, model.y: batch_y, model.keep_prob: KEEP_PROB
    })

    if not should_perform_acc_test(bi) or BENCHMARK_MODE:
        elapsed_time = time.time() - EPOCH_TIME
        rate = round(BATCH_SIZE * (bi.batch_idx+1) / elapsed_time, 2)
        print("\r\033[K", end="")
        print(f"Rate: {rate:5.2f}, ", end = "")
        print(
            f"Epoch {bi.epoch_idx + 1}, Batch {bi.batch_idx + 1}",
            end = "",
            flush = True
        )
        return

    # Calculate loss and accuracy
    loss = session.run(trainer.cost, feed_dict = {
        model.x: bi.dataset.validation.images,
        model.y: bi.dataset.validation.labels,
        model.keep_prob: 1.0
    })
    # Notice use of 1.0 as keep probability when evaluating.
    accuracy = session.run(trainer.accuracy, feed_dict = {
        model.x: bi.dataset.validation.images,
        model.y: bi.dataset.validation.labels,
        model.keep_prob: 1.0
    })
    print("\r\033[K", end = "")
    print(
        f"Epoch {bi.epoch_idx + 1}, Batch {bi.batch_idx + 1} - "
        f"Validation Loss: {loss:>10.4f} "
        f"Validation Accuracy: {accuracy:.6f}"
    )

BatchInfo = namedtuple(
    "BatchInfo",
    "dataset, epoch_idx, batch_idx, batch_size, num_batches"
)

def train(dataset, model, trainer):
    global EPOCH_TIME

    num_batches = int(dataset.train.num_examples / BATCH_SIZE)
    for epoch_idx in range(NUM_EPOCHS):
        if EPOCH_TIME is None:
            EPOCH_TIME = time.time()

        for batch_idx in range(num_batches):
            batch_info = BatchInfo(
                dataset = dataset,
                epoch_idx = epoch_idx,
                batch_idx = batch_idx,
                batch_size = BATCH_SIZE,
                num_batches = num_batches,
            )

            train_batch(
                session = session,
                model = model,
                trainer = trainer,
                batch_info = batch_info,
            )

    # Calculate Final Test Accuracy
    test_acc = session.run(trainer.accuracy, feed_dict = {
        model.x: dataset.test.images,
        model.y: dataset.test.labels,
        model.keep_prob: 1.0
    })

    print()
    print(f"Testing Accuracy: {test_acc}")

def run(session):
    # Load data and build model.
    dataset = input_data.read_data_sets(
        ".", one_hot = True, reshape = False
    )
    model = build_model()
    trainer = build_trainer(model)

    # This quickly loads the trained file if it already exists.
    ipt = input("Load old model? ")
    saver = tf.train.Saver()
    if ipt != "no":
        saver.restore(session, MODEL_FNAME)
    else:
        # Initializing the variables
        session.run(tf.global_variables_initializer())

    train(dataset, model, trainer)
    saver.save(session, MODEL_FNAME)

with tf.Session() as session:
    run(session)
