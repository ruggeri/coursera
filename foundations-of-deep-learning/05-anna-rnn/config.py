import numpy as np
import random
import tensorflow as tf

# Needed to make sure vocab is not random in test mode
np.random.seed(1)

# These settings work well for the two 6-letter words test.
#BATCH_SIZE = 40
#BURN_IN_LETTERS = 1000
#CHARS_TO_GENERATE = 100
#CLIP_GRADIENT = 5
#KEEP_PROB = 0.50
#LEARNING_RATE = .10
#NUM_EPOCHS = 20
#NUM_LAYERS = 1
#NUM_LSTM_UNITS = 8
#SAVE_FREQUENCY = 0.10
#STEP_SIZE = 2
#TEST_MODE = True
#TOP_N = 2
#VALIDATION_FREQUENCY = 0.10

# These settings worked pretty well to train an ~60% accuracy model of
# Anna Karenina.
#BATCH_SIZE = 40
#BURN_IN_LETTERS = 128
#CHARS_TO_GENERATE = 2048
#CLIP_GRADIENT = 5
#KEEP_PROB = 0.50
#LEARNING_RATE = .001
#NUM_EPOCHS = 20
#NUM_LAYERS = 2
#NUM_LSTM_UNITS = 256
#SAVER_BASENAME = "two-layer-rnn-model-anna-simplified"
#SAVE_FREQUENCY = 0.10
#STEP_SIZE = 100
#TEST_MODE = False
#TOP_N = 5
#VALIDATION_FREQUENCY = 0.10

# These settings max out at ~60% accuracy, too.
BATCH_SIZE = 10
BURN_IN_LETTERS = 128
CHARS_TO_GENERATE = 2048
CLIP_GRADIENT = 5
KEEP_PROB = 0.50
LEARNING_RATE = .01
NUM_EPOCHS = 20
NUM_LAYERS = 3
NUM_LSTM_UNITS = 256
RESTORE_FILENAME = "./two-layer-rnn-model-anna-simplified-19-0439.ckpt"
SAVER_BASENAME = "three-layer-rnn-model-anna-simplified"
SAVE_FREQUENCY = 0.10
STEP_SIZE = 100
TEST_MODE = False
TOP_N = 5
VALIDATION_FREQUENCY = 0.10

import file_reader as fr
file_reader = fr.FileReader('./anna-simplified.txt')
