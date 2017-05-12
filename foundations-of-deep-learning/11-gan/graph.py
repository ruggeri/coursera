from collections import namedtuple
import tensorflow as tf

# TODO: Ian recommends Leaky ReLU. He also recommends using a tanh for
# the generator, but I will ignore that advice for the moment.

# TODO: Label smoothing.

Graph = namedtuple("Graph", [
    "one_hot_class_label",
    # Generator input/output
    "z",
    "generated_x",
    # Discriminator input/output
    "discriminator_x",
    "authenticity_label",
    "discriminator_percentage",
    # Discriminator training
    "discriminator_loss",
    "train_discriminator_op",
    # Generator training
    "generator_loss",
    "train_generator_op",
])

def generator(z_dims, num_hidden_units, x_dims, one_hot_class_label):
    with tf.variable_scope("generator"):
        z = tf.placeholder(tf.float32, [None, z_dims])
        h1 = tf.layers.dense(
            inputs = tf.concat([one_hot_class_label, z], axis = 1),
            units = num_hidden_units,
            activation = tf.nn.relu,
        )
        generated_x = tf.layers.dense(
            inputs = h1,
            units = x_dims,
            activation = tf.nn.sigmoid,
        )

    return (z, generated_x)

def discriminator(num_hidden_units, one_hot_class_label, x, reuse):
    with tf.variable_scope("discriminator", reuse = reuse):
        h1 = tf.layers.dense(
            inputs = tf.concat([one_hot_class_label, x], axis = 1),
            units = num_hidden_units,
            activation = tf.nn.relu,
        )
        estimated_authenticity_logits = tf.layers.dense(
            inputs = h1,
            units = 1,
            activation = None,
        )
        estimated_authenticity_percentage = tf.nn.sigmoid(choice_logits)

    return (
        estimated_authenticity_logits,
        estimated_authenticity_percentage
    )

def build(
        num_classes,
        x_dims,
        z_dims,
        num_generator_hidden_units,
        num_discriminator_hidden_units):
    one_hot_class_label = tf.placeholder(
        tf.float32, [None, num_classes]
    )

    # Generator
    z, generated_x = generator(
        z_dims = z_dims,
        num_hidden_units = num_generator_hidden_units,
        x_dims = x_dims,
        one_hot_class_label = one_hot_class_label,
    )

    # Discriminator
    discriminator_x = tf.placeholder(tf.float32, [None, x_dims])
    authenticity_label = tf.placeholder(tf.float32, [None])
    discriminator_logits, discriminator_percentage = discriminator(
        num_hidden_units = num_discriminator_hidden_units,
        one_hot_class_label = one_hot_class_label,
        x = discriminator_x,
        reuse = False
    )

    # Discriminator training
    optimizer = tf.train.AdamOptimizer()
    discriminator_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = authenticity_label,
        logits = discriminator_logits
    )
    train_discriminator_op = optimizer.minimize(discriminator_loss)

    # Generator training
    generator_logits, _ = discriminator(
        num_hidden_units = num_discriminator_hidden_units,
        one_hot_class_label = one_hot_class_label,
        x = generated_x,
        reuse = True
    )
    # NB: Rather than explicitly try to make the discriminator
    # maximize, we minimize the "wrong" loss, because the gradients
    # are stronger to learn from.
    generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = tf.ones_like(one_hot_class_label, dtype = tf.float32),
        logits = generator_logits
    )
    # We only want to update parameters of the generator, even though
    # the discriminator's estimates are part of the training process.
    generator_vars = [
        var for var in tf.traininable_variables()
        if re.match(r"generator", var.name)
    ]
    train_generator_op = optimizer.minimize(
        generator_loss, var_list = generator_vars
    )

    return Graph(
        one_hot_class_label = one_hot_class_label,
        # Generator input/output
        z = z,
        generated_x = generated_x,
        # Discriminator input/output
        discriminator_x = discriminator_x,
        authenticity_label = authenticity_label,
        discriminator_percentage = discriminate_percentage,
        # Discriminator training
        discriminator_loss = discriminator_loss,
        train_discriminator_op = train_discriminator_op,
        # Generator training
        generator_loss = generator_loss,
        train_generator_op = train_generator_op
    )
