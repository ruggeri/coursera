import config

def make_batches(x, y, batch_size = config.BATCH_SIZE):
    num_batches = x.shape[0] // batch_size

    def helper():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, x.shape[0])
            yield (x[start_idx:end_idx], y[start_idx:end_idx])

    return num_batches, helper()
