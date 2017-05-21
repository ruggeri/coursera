BATCH_SIZE = 32
BATCHES_PER_LOG = 10
BATCHES_PER_SAMPLING = 100
# Recommended by DCGAN paper: beta1 = 0.5
BETA1 = 0.5
# Performance is much better if the generator can train multiple
# rounds per discriminator training.
GENERATOR_ROUND_MULTIPLIER = 5
LABEL_SMOOTHING = 0.10
# Recommended by DCGAN paper: lr = 0.0002
LEARNING_RATE = 0.0002
NUM_EPOCHS = 50
