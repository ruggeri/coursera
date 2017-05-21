import config

DEFAULT_CONV_KSIZE = 7

# Discriminator Configuration
DISCRIMINATOR_LAYERS = [
    # == CONV #1 ==
    # DCGAN paper says no BN on first discriminator layer. Says
    # use leaky ReLU throughout discriminator.
    { "type": "conv2d",
      "activation": "leaky_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "strides": 2,
      "num_filters": 64, },
    # == CONV #2 ==
    { "type": "conv2d",
      "activation": "bn_leaky_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "strides": 2,
      "num_filters": 256, },
    # == CONV #3 ==
    { "type": "conv2d",
      "activation": "bn_leaky_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "strides": 2,
      "num_filters": 512, },
    # == FLATTEN ==
    { "type": "flatten" },
    # == DENSE OUTPUT ==
    # Final sigmoid layer, but we hold off on activation fn so we
    # get logits.
    { "type": "dense", "num_units": 1, "activation": None }
]

# Generator Configuration
INITIAL_SIZE = (7, 7, 512)
INITIAL_PIXELS = INITIAL_SIZE[0] * INITIAL_SIZE[1] * INITIAL_SIZE[2]
GENERATOR_LAYERS = [
    # == DENSE INITIAL ==
    { "type": "dense",
      "num_units": INITIAL_PIXELS,
      "activation": "bn_relu" },
    # == RESHAPE ==
    { "type": "reshape",
      "dimensions": INITIAL_SIZE },
    # == CONV #1 ==
    { "type": "conv2d_transpose",
      "activation": "bn_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "strides": 2,
      "num_filters": 256 },
    # == CONV #2 ==
    { "type": "conv2d_transpose",
      "activation": "bn_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "strides": 2,
      "num_filters": 128, },
    # == CONV #3 ==
    { "type": "conv2d_transpose",
      "activation": "bn_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "strides": 1,
      "num_filters": 64, },
    # == CONV #4 ==
    # I added a final extra tanh convolution.
    { "type": "conv2d",
      "activation": "tanh",
      "ksize": DEFAULT_CONV_KSIZE,
      "strides": 1,
      "num_filters": config.IMAGE_DIMS[2] }
]
