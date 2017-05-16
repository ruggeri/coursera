import config
import tensorflow as tf

def training_mean_and_variance(z, population_mean, population_variance):
    axes = list(range(0, len(z.get_shape()) - 1))

    with tf.name_scope("training"):
        batch_mean, batch_variance = tf.nn.moments(z, axes = axes)

        with tf.name_scope("mean"):
            new_population_mean = tf.identity(
                (config.DECAY * population_mean)
                + ((1 - config.DECAY) * batch_mean),
                name = "new_population_mean",
            )
            update_mean = tf.assign(
                population_mean,
                new_population_mean,
                name = "update_mean"
            )

        with tf.name_scope("variance"):
            new_population_variance = tf.identity(
                (config.DECAY * population_variance)
                + ((1 - config.DECAY) * batch_variance),
                name = "new_population_variance"
            )
            update_variance = tf.assign(
                population_variance,
                new_population_variance,
                name = "update_variance"
            )

        with tf.control_dependencies([update_mean, update_variance]):
            return [
                tf.identity(batch_mean, name = "batch_mean"),
                tf.identity(batch_variance, name = "batch_variance")
            ]

def inference_mean_and_variance(
        z,
        population_mean,
        population_variance):
    return population_mean, population_variance

def batch_normalize(z, is_training):
    with tf.name_scope("batch_normalization"):
        num_dims = z.get_shape()[-1]

        with tf.name_scope("statistics"):
            population_mean = tf.Variable(
                tf.zeros(num_dims),
                trainable = False,
                name = "population_mean"
            )
            population_variance = tf.Variable(
                tf.ones(num_dims),
                trainable = False,
                name = "population_variance"
            )

        with tf.name_scope("parameters"):
            beta = tf.Variable(tf.zeros(num_dims), name = "beta")
            gamma = tf.Variable(tf.ones(num_dims), name = "gamma")

        mean, variance = tf.cond(
            is_training,
            lambda: training_mean_and_variance(
                z, population_mean, population_variance
            ),
            lambda: inference_mean_and_variance(
                z, population_mean, population_variance
            )
        )

        with tf.name_scope("output"):
            normalized_z = tf.identity(
                (z - mean) / tf.sqrt(variance + config.EPSILON),
                name = "normalized_z"
            )
            output = tf.identity(
                gamma * normalized_z + beta, name = "output"
            )

    return output
