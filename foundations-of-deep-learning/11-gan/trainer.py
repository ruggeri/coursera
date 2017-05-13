import tensorflow as tf

def build(
        prediction_logits,
        prediction,
        authenticity_label,
        variable_scope):
    with tf.name_scope("trainer"):
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(
                    tf.cast(tf.round(prediction), tf.int64),
                    tf.cast(authenticity_label, tf.int64),
                ), tf.float32),
                name = "accuracy",
            )

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = tf.cast(authenticity_label, tf.float32),
                    logits = prediction_logits
                ),
                name = "loss",
            )

        train_op = tf.train.AdamOptimizer().minimize(
            loss,
            var_list = variable_scope.global_variables()
        )

    return (accuracy, loss, train_op)
