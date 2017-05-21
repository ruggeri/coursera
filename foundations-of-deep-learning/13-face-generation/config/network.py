import config
import tensorflow as tf

# The DCGAN people recommend this initialization.
KERNEL_INITIALIZER = tf.truncated_normal_initializer(
    mean = 0.0,
    stddev = 0.02
)
LEAKAGE = 0.20
Z_DIMS = 100

if config.NETWORK_NAME == "A":
    import config.networkA
    DISCRIMINATOR_LAYERS = config.networkA.DISCRIMINATOR_LAYERS
    GENERATOR_LAYERS = config.networkA.GENERATOR_LAYERS
elif config.NETWORK_NAME == "B":
    import config.networkB
    DISCRIMINATOR_LAYERS = config.networkB.DISCRIMINATOR_LAYERS
    GENERATOR_LAYERS = config.networkB.GENERATOR_LAYERS
else:
    raise Exception("Not yet implemented!")
