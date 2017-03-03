# This shows how to setup a CNN with dropout in TensorFlow! I don't
# know that this CNN does any better than my two-layer basic FFNN, but
# it's just a demo of how to setup a CNN.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

# Parameters
learning_rate = 0.00001
epochs = 1
batch_size = 128

# Network Parameters
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# Initial weights and biases.
weights = {
    # 5x5 filter from 1 channel to 32 channels.
    'conv_weights1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 filter from 32 channels to 64 channels.
    'conv_weights2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # Starts out as 28x28 input, but I do maxpooling with a stride of
    # 2 after both convolutional layers.
    'dense_weights1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'conv_biases1': tf.Variable(tf.random_normal([32])),
    'conv_biases2': tf.Variable(tf.random_normal([64])),
    'dense_biases1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Setup a basic convolutional layer.
def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(
        x, W, strides=[1, strides, strides, 1], padding='SAME'
    )
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Setup a basic maxpooling layer.
def maxpool2d(x, k = 2):
    return tf.nn.max_pool(
        x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME'
    )

# This builds the CNN
def conv_net(x, weights, biases, dropout):
    # Layer 1 - 28*28*1 to 14*14*32
    conv1 = conv2d(x, weights['conv_weights1'], biases['conv_biases1'])
    conv1 = maxpool2d(conv1, k=2)

    # Layer 2 - 14*14*32 to 7*7*64
    conv2 = conv2d(
        conv1, weights['conv_weights2'], biases['conv_biases2']
    )
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer - 7*7*64 to 1024
    # First we flatten the 2D image to a 1D representation.
    num_prev_units = weights["dense_weights1"].get_shape().as_list()[0]
    fc1 = tf.reshape(conv2, [-1, num_prev_units])
    # Now we perform the layer.
    fc1 = tf.add(tf.matmul(
        fc1, weights['dense_weights1']), biases['dense_biases1']
    )
    fc1 = tf.nn.relu(fc1)
    # We use dropout here to regularize.
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output Layer.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Placeholders
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Model
logits = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate
).minimize(cost)

# Calculate accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

VALID_TEST_SIZE = 1024
def run_batch():
    # Train on batch.
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict={
        x: batch_x, y: batch_y, keep_prob: dropout
    })

    # Calculate batch loss and accuracy
    loss = sess.run(cost, feed_dict={
        x: batch_x, y: batch_y, keep_prob: 1.0
    })
    # Notice use of 1.0 as keep probability when evaluating.
    valid_acc = sess.run(accuracy, feed_dict={
        x: mnist.validation.images[:VALID_TEST_SIZE],
        y: mnist.validation.labels[:VALID_TEST_SIZE],
        keep_prob: 1.0
    })

    print("".join([
        f"Epoch {epoch+1}, Batch {batch+1} - ",
        f"Validation Loss: {loss:>10.4f} "
        f"Validation Accuracy: {valid_acc:.6f}"
    ]))

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        num_batches = int(mnist.train.num_examples / batch_size)
        for batch in range(num_batches):
            run_batch()

    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_valid_size],
        y: mnist.test.labels[:test_valid_size],
        keep_prob: 1.0
    })
    print(f"Testing Accuracy: {test_acc}")
