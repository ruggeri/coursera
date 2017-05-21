import config

MODE = "A"
LEAKAGE = 0.20
Z_DIMS = 100

if MODE == "A":
    DEFAULT_CONV_KSIZE = 7
    DEFAULT_NUM_CONV_FILTERS = 128

    # Discriminator Configuration
    DISCRIMINATOR_LAYERS = [
        # DCGAN paper says no BN on first discriminator layer. Says
        # use leaky ReLU throughout discriminator.
        { "type": "conv2d",
          "activation": "leaky_relu",
          "ksize": DEFAULT_CONV_KSIZE,
          "num_filters": DEFAULT_NUM_CONV_FILTERS, },
        { "type": "maxpool" },
        { "type": "conv2d",
          "activation": "bn_leaky_relu",
          "ksize": DEFAULT_CONV_KSIZE,
          "num_filters": DEFAULT_NUM_CONV_FILTERS, },
        { "type": "maxpool" },
        { "type": "conv2d",
          "activation": "bn_leaky_relu",
          "ksize": DEFAULT_CONV_KSIZE,
          "num_filters": DEFAULT_NUM_CONV_FILTERS, },
        { "type": "maxpool" },
        { "type": "flatten" },
        # Final sigmoid layer, but we hold off on activation fn so we
        # get logits.
        { "type": "dense", "num_units": 1, "activation": None }
    ]

    # Generator Configuration
    INITIAL_SIZE = (3, 3, config.IMAGE_DIMS[2])
    INITIAL_PIXELS = INITIAL_SIZE[0] * INITIAL_SIZE[1] * INITIAL_SIZE[2]
    GENERATOR_LAYERS = [
        { "type": "dense",
          "num_units": INITIAL_PIXELS,
          "activation": "tanh" },
        { "type": "reshape",
          "dimensions": INITIAL_SIZE },
        { "type": "resize", "size": (7, 7), },
        { "type": "conv2d",
          "activation": "bn_relu",
          "ksize": DEFAULT_CONV_KSIZE,
          "num_filters": DEFAULT_NUM_CONV_FILTERS, },
        { "type": "resize", "size": (14, 14), },
        { "type": "conv2d",
          "activation": "bn_relu",
          "ksize": DEFAULT_CONV_KSIZE,
          "num_filters": DEFAULT_NUM_CONV_FILTERS, },
        { "type": "resize",
          "size": (config.IMAGE_DIMS[0], config.IMAGE_DIMS[1]), },
        { "type": "conv2d",
          "activation": "bn_relu",
          "ksize": DEFAULT_CONV_KSIZE,
          "num_filters": DEFAULT_NUM_CONV_FILTERS, },
        # I added a final extra tanh convolution.
        { "type": "conv2d",
          "activation": "tanh",
          "ksize": DEFAULT_CONV_KSIZE,
          "num_filters": config.IMAGE_DIMS[2] }
    ]
elif MODE == "B":
    pass
