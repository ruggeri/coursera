import config
import discriminator
import tensorflow as tf

GENERATOR_MODE = "GENERATOR_MODE"
DISCRIMINATOR_REAL_MODE = "DISCRIMINATOR_REAL_MODE"
DISCRIMINATOR_FAKE_MODE = "DISCRIMINATOR_FAKE_MODE"

def loss(prediction_logits, mode):
    with tf.name_scope("loss"):
        if mode == GENERATOR_MODE:
            # NB: Rather than explicitly try to make the discriminator
            # maximize, we minimize the "wrong" loss, because the
            # gradients are stronger to learn from.
            labels = tf.ones_like(
                prediction_logits, dtype = tf.float32, name = "labels"
            )
        elif mode == DISCRIMINATOR_REAL_MODE:
            labels = config.LABEL_SMOOTHING * tf.ones_like(
                prediction_logits, dtype = tf.float32, name = "labels"
            )
        elif mode == DISCRIMINATOR_FAKE_MODE:
            labels = tf.zeros_like(
                prediction_logits, dtype = tf.float32, name = "labels"
            )
        else:
            raise Exception("unknown label mode")

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = labels,
                logits = prediction_logits
            ),
            name = "loss",
        )

    return loss

def accuracy(prediction, mode):
    with tf.name_scope("accuracy"):
        if (mode == GENERATOR_MODE) or (mode == DISCRIMINATOR_FAKE_MODE):
            labels = tf.zeros_like(
                prediction, dtype = tf.int64, name = "labels"
            )
        elif mode == DISCRIMINATOR_REAL_MODE:
            labels = tf.ones_like(
                prediction, dtype = tf.int64, name = "labels"
            )
        else:
            raise Exception("unknown label mode")

        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(
                tf.cast(tf.round(prediction), tf.int64),
                labels
            ), tf.float32),
            name = "accuracy",
        )

    return accuracy

def build_for_generator(
        one_hot_class_label,
        generated_x,
        discriminator_parameters,
        variable_scope):
    with tf.name_scope("trainer"):
        with tf.name_scope("discriminator"):
            prediction_logits, _ = discriminator.apply_parameters(
                one_hot_class_label = one_hot_class_label,
                x = generated_x,
                discriminator_parameters = discriminator_parameters,
            )

        loss_ = loss(prediction_logits, GENERATOR_MODE)

        train_op = tf.train.AdamOptimizer().minimize(
            loss_,
            var_list = variable_scope.trainable_variables()
        )

    return (loss_, train_op)

def build_for_discriminator(
        fake_prediction_logits,
        fake_prediction,
        real_prediction_logits,
        real_prediction,
        variable_scope):
    with tf.name_scope("trainer"):
        with tf.name_scope("fake"):
            fake_loss = loss(
                fake_prediction_logits,
                DISCRIMINATOR_FAKE_MODE,
            )
            fake_accuracy = accuracy(
                fake_prediction,
                DISCRIMINATOR_FAKE_MODE
            )
        with tf.name_scope("real"):
            real_loss = loss(
                real_prediction_logits,
                DISCRIMINATOR_REAL_MODE,
            )
            real_accuracy = accuracy(
                real_prediction,
                DISCRIMINATOR_REAL_MODE
            )

        with tf.name_scope("combined"):
            loss_ = tf.truediv(
                (fake_loss + real_loss),
                2.0,
                name = "loss"
            )
            accuracy_ = tf.truediv(
                (fake_accuracy + real_accuracy),
                2.0,
                name = "accuracy"
            )

        train_op = tf.train.AdamOptimizer().minimize(
            loss_,
            var_list = variable_scope.trainable_variables()
        )

    return (accuracy_, loss_, train_op)
