import tensorflow as tf

MODE = "A"

# The DCGAN people recommend this initialization.
KERNEL_INITIALIZER = tf.truncated_normal_initializer(
    mean = 0.0,
    stddev = 0.02
)
LEAKAGE = 0.20
Z_DIMS = 100

if MODE == "A":
    import config.networkA
    DISCRIMINATOR_LAYERS = config.networkA.DISCRIMINATOR_LAYERS
    GENERATOR_LAYERS = config.networkA.GENERATOR_LAYERS
elif MODE == "B":
    raise Exception("Not yet implemented!")
