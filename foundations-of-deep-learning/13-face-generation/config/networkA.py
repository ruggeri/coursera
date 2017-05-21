import config

DEFAULT_CONV_KSIZE = 7
DEFAULT_NUM_CONV_FILTERS = 128
DEFAULT_CONV_STRIDE = 1

# Discriminator Configuration
DISCRIMINATOR_LAYERS = [
    # == CONV #1 ==
    # DCGAN paper says no BN on first discriminator layer. Says
    # use leaky ReLU throughout discriminator.
    { "type": "conv2d",
      "activation": "leaky_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "stride": DEFAULT_CONV_STRIDE,
      "num_filters": DEFAULT_NUM_CONV_FILTERS, },
    # == MAXPOOL #1 ==
    { "type": "maxpool" },
    # == CONV #2 ==
    { "type": "conv2d",
      "activation": "bn_leaky_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "stride": DEFAULT_CONV_STRIDE,
      "num_filters": DEFAULT_NUM_CONV_FILTERS, },
    # == MAXPOOL #2 ==
    { "type": "maxpool" },
    # == CONV #3 ==
    { "type": "conv2d",
      "activation": "bn_leaky_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "stride": DEFAULT_CONV_STRIDE,
      "num_filters": DEFAULT_NUM_CONV_FILTERS, },
    # == MAXPOOL #3 ==
    { "type": "maxpool" },
    # == FLATTEN ==
    { "type": "flatten" },
    # == DENSE OUTPUT ==
    # Final sigmoid layer, but we hold off on activation fn so we
    # get logits.
    { "type": "dense", "num_units": 1, "activation": None }
]

# Generator Configuration
INITIAL_SIZE = (3, 3, config.IMAGE_DIMS[2])
INITIAL_PIXELS = INITIAL_SIZE[0] * INITIAL_SIZE[1] * INITIAL_SIZE[2]
GENERATOR_LAYERS = [
    # == DENSE INITIAL ==
    { "type": "dense",
      "num_units": INITIAL_PIXELS,
      "activation": "tanh" },
    # == RESHAPE ==
    { "type": "reshape",
      "dimensions": INITIAL_SIZE },
    # == RESIZE 7x7 ==
    { "type": "resize", "size": (7, 7), },
    # == CONV #1 ==
    { "type": "conv2d",
      "activation": "bn_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "stride": DEFAULT_CONV_STRIDE,
      "num_filters": DEFAULT_NUM_CONV_FILTERS, },
    # == RESIZE 14x14 ==
    { "type": "resize", "size": (14, 14), },
    # == CONV #2 ==
    { "type": "conv2d",
      "activation": "bn_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "stride": DEFAULT_CONV_STRIDE,
      "num_filters": DEFAULT_NUM_CONV_FILTERS, },
    # == RESIZE 28x28 ==
    { "type": "resize",
      "size": (config.IMAGE_DIMS[0], config.IMAGE_DIMS[1]), },
    # == CONV #3 ==
    { "type": "conv2d",
      "activation": "bn_relu",
      "ksize": DEFAULT_CONV_KSIZE,
      "stride": DEFAULT_CONV_STRIDE,
      "num_filters": DEFAULT_NUM_CONV_FILTERS, },
    # == CONV #4 ==
    # I added a final extra tanh convolution.
    { "type": "conv2d",
      "activation": "tanh",
      "ksize": DEFAULT_CONV_KSIZE,
      "stride": DEFAULT_CONV_STRIDE,
      "num_filters": config.IMAGE_DIMS[2] }
]
