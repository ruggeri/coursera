import numpy as np

def make_batches(one_hot_txt, batch_size, num_training_steps):
    num_chars_in_text = one_hot_txt.shape[0]
    num_batches = (num_chars_in_text - 1) // (num_training_steps * batch_size)
    usable_chars = num_batches * (num_training_steps * batch_size)

    one_hot_txt_x = one_hot_txt[:usable_chars]
    one_hot_txt_y = one_hot_txt[1:usable_chars+1]

    # create matrix where dim 0 in the stream num; dim 1 is a character pos
    # dim 2 in the one hot encoding
    streams_x = np.stack(np.split(one_hot_txt_x, batch_size))
    batches_x = np.split(streams_x, num_batches, axis=1)
    streams_y = np.stack(np.split(one_hot_txt_y, batch_size))
    batches_y = np.split(streams_y, num_batches, axis=1)
    return list(zip(batches_x, batches_y))

def partition_batches(batches):
    num_training_batches = int(len(batches) * 0.9)
    train_batches = batches[:num_training_batches]
    val_batches = batches[num_training_batches:]
    return (train_batches, val_batches)
