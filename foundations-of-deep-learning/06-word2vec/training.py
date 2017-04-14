from collections import namedtuple
import config
import graph as graph_fns
import preprocessing
import tensorflow as tf

RunInfo = namedtuple("RunInfo", [
    "session",
    "graph",
    "saver",
    "batcher",
    "batches_per_epoch",
    "batches_per_logging",
])

BatchInfo = namedtuple("BatchInfo", [
    "epoch_idx",
    "batch_idx",
    "inputs",
    "labels",
])

def run_batch(run_info, batch_info):
    ri, bi = run_info, batch_info

    training_loss, _ = ri.session.run(
        [ri.graph.cost, ri.graph.optimizer],
        feed_dict = {
            ri.graph.inputs: bi.inputs,
            ri.graph.labels: bi.labels
        }
    )

    should_log = (
        ((bi.batch_idx + 1) % ri.batches_per_logging) == 0
    )
    if should_log:
        print(f"Epoch: {bi.epoch_idx:03.d} | "
              f"Batch: {bi.batch_idx:04.d} / {ri.batches_per_epoch:04.d} | "
              f"Avg Train Loss: {training_loss:.2f} | "
              f"{0} sec / example"
        )

def run_epoch(run_info, epoch_idx):
    batches = run_info.batcher.batches(
        config.BATCH_SIZE, config.WINDOW_SIZE
    )

    for batch_idx, (inputs, labels) in enumerate(batches):
        batch_info = BatchInfo(
            epoch_idx = epoch_idx,
            batch_idx = batch_idx,
            inputs = inputs,
            labels = labels,
        )

        run_batch(run_info, batch_info)

def run(session):
    batcher = preprocessing.Batcher()
    num_batches = batcher.num_batches(config.BATCH_SIZE)

    run_info = RunInfo(
        session = session,
        graph = graph_fns.build_graph(
            vocab_size = batcher.vocab_size(),
            num_embedding_units = config.NUM_EMBEDDING_UNITS,
            num_negative_samples = config.NUM_NEGATIVE_SAMPLES,
        ),
        saver = tf.train.Saver(),
        batcher = batcher,
        batches_per_epoch = num_batches,
        batches_per_logging = int(
            num_batches * config.LOGGING_FREQUENCY
        ),
    )

    for epoch_idx in range(1, config.NUM_EPOCHS + 1):
        run_epoch(run_info, epoch_idx)

with tf.Session() as session:
    run(session)
