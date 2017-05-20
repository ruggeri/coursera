from collections import namedtuple
import config
import layers
import tensorflow as tf
import trainer as trainer_mod

Network = namedtuple("Network", [
    "fake_z",
    "inference_fake_x",
    "real_x",
    "trainer",
])

def discriminator(images, reuse):
    with tf.variable_scope("discriminator", reuse = reuse):
        return layers.build_layers(
            images,
            config.DISCRIMINATOR_LAYERS,
            # The discriminator is *only* used in training mode.
            is_training = True
        )

def generator(fake_z, is_training, reuse):
    with tf.variable_scope("generator", reuse = reuse):
        return layers.build_layers(
            fake_z,
            config.GENERATOR_LAYERS,
            is_training = is_training
        )

def network():
    with tf.name_scope("placeholders"):
        real_x = tf.placeholder(
            tf.float32,
            (None, *config.IMAGE_DIMS),
            name = "real_x"
        )

        fake_z = tf.placeholder(
            tf.float32,
            (None, config.Z_DIMS),
            name = "fake_z"
        )

    training_fake_x = generator(
        fake_z = fake_z,
        is_training = True,
        reuse = False
    )
    inference_fake_x = generator(
        fake_z = fake_z,
        is_training = False,
        reuse = True,
    )

    discriminator_fake_logits = discriminator(
        images = training_fake_x,
        reuse = False,
    )
    discriminator_real_logits = discriminator(
        images = real_x,
        reuse = True,
    )

    trainer = trainer_mod.build(
        real_logits = discriminator_real_logits,
        fake_logits = discriminator_fake_logits
    )

    return Network(
        fake_z = fake_z,
        inference_fake_x = inference_fake_x,
        real_x = real_x,
        trainer = trainer
    )
