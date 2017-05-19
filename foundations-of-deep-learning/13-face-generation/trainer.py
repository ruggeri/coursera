from collections import namedtuple
import config
import tensorflow as tf

Trainer = namedtuple("Trainer", [
    "d_train_op",
    "g_train_op",
    "d_loss",
    "g_loss",
    "accuracy",
])

def losses(real_logits, fake_logits):
    with tf.name_scope("d_loss"):
        with tf.name_scope("d_real_loss"):
            d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = (
                    (1 - config.LABEL_SMOOTHING) * tf.ones_like(real_logits)
                ),
                logits = real_logits
            )
        with tf.name_scope("d_fake_loss"):
            d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tf.zeros_like(fake_logits),
                logits = fake_logits
            )
        d_loss = tf.reduce_mean(
            tf.concat(
                [d_real_loss, d_fake_loss],
                axis = 0
            )
        )

    with tf.name_scope("g_loss"):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.ones_like(fake_logits),
            logits = fake_logits
        ))

    return d_loss, g_loss

def accuracy(real_logits, fake_logits):
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.concat([
            tf.cast(tf.equal(
                tf.round(tf.sigmoid(real_logits)),
                tf.ones_like(real_logits)), tf.float32
            ),
            tf.cast(tf.equal(
                tf.round(tf.sigmoid(fake_logits)),
                tf.zeros_like(fake_logits)), tf.float32
            )
        ], axis = 0))

    return accuracy

def build(real_logits, fake_logits):
    d_loss, g_loss = losses(
        real_logits = real_logits,
        fake_logits = fake_logits
    )

    discriminator_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope = "discriminator"
    )
    generator_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope = "generator"
    )

    optimizer = tf.train.AdamOptimizer(
        learning_rate = config.LEARNING_RATE,
        beta1 = config.BETA1
    )

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        d_train_op = optimizer.minimize(
            d_loss,
            var_list = discriminator_vars
        )

    with tf.control_dependencies(update_ops):
        g_train_op = optimizer.minimize(
            g_loss,
            var_list = generator_vars
        )

    return Trainer(
        d_train_op = d_train_op,
        g_train_op = g_train_op,
        d_loss = d_loss,
        g_loss = g_loss,
        accuracy = accuracy(
            real_logits = real_logits,
            fake_logits = fake_logits,
        ),
    )
