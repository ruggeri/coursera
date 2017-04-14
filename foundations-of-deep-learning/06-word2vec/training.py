import config
import preprocessing

def run():
    batcher = preprocessing.Batcher()
    saver = tf.train.Saver()

    for epoch_idx in range(1, config.NUM_EPOCHS + 1):
        batches = batcher.batches(config.BATCH_SIZE, config.WINDOW_SIZE)

