BATCH_SIZE = 32
BETA1 = 0.5
CONV_KSIZE = 7
LABEL_SMOOTHING = 0.10
LEAKAGE = 0.20
LEARNING_RATE = 0.0002
NUM_CONV_FILTERS = 128
NUM_OUTPUT_FILTERS = 1
Z_DIM = 100

# Discriminator Configuration
DISCRIMINATOR_LAYERS = [
    # DCGAN paper says no BN on first discriminator layer. Says use
    # leaky ReLU throughout discriminator.
    { "type": "conv2d", "activation": "leaky_relu" },
    { "type": "maxpool" },
    { "type": "conv2d", "activation": "bn_leaky_relu" },
    { "type": "maxpool" },
    { "type": "conv2d", "activation": "bn_leaky_relu" },
    { "type": "maxpool" },
]

# Generator Configuration
INITIAL_SIZE = (3, 3)
GENERATOR_LAYERS = [
    { "type": "resize", "size": (7, 7), },
    { "type": "conv2d", "activation": "bn_relu" },
    { "type": "resize", "size": (14, 14), },
    { "type": "conv2d", "activation": "bn_relu" },
    { "type": "resize", "size": (28, 28), },
    { "type": "conv2d", "activation": "bn_relu" },
    # I added a final extra tanh convolution.
    { "type": "conv2d",
      "activation": "tanh",
      "num_filters": NUM_OUTPUT_FILTERS }
]
