from collections import namedtuple
import config
import graph as graph_fns
import preprocessing
import tensorflow as tf
import time

RunInfo = namedtuple("RunInfo", [
    "session",
    "graph",
    "saver",
    "batcher",
    "batch_size",
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

    batch_training_loss, _ = ri.session.run(
        [ri.graph.cost, ri.graph.optimizer],
        feed_dict = {
            ri.graph.inputs: bi.inputs,
            ri.graph.labels: bi.labels
        }
    )

    return batch_training_loss

def log_batches(run_info, batch_info, cumulative_loss, start_time):
    ri, bi = run_info, batch_info
    num_examples = ri.batches_per_logging * ri.batch_size
    end_time = time.time()
    examples_per_sec = num_examples / (end_time - start_time)
    print(f"Epoch: {bi.epoch_idx:03d} | "
          f"Batch: {bi.batch_idx:04d} / {ri.batches_per_epoch:04d} | "
          f"Avg Train Loss: {cumulative_loss:.2f} | "
          f"{examples_per_sec:3.2f} sec / example"
    )

def run_epoch(run_info, epoch_idx):
    batches = run_info.batcher.batches(
        config.BATCH_SIZE, config.WINDOW_SIZE
    )

    cumulative_loss = 0
    start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(batches):
        batch_info = BatchInfo(
            epoch_idx = epoch_idx,
            batch_idx = batch_idx,
            inputs = inputs,
            labels = labels,
        )

        cumulative_loss += run_batch(run_info, batch_info)

        should_log = (
            ((batch_idx + 1) % ri.batches_per_logging) == 0
        )
        if should_log:
            log_batches(batch_info, cumulative_loss, start_time)
            cumulative_loss = 0
            start_time = 0

def run(session):
    batcher = preprocessing.Batcher()
    num_batches = batcher.num_batches(config.BATCH_SIZE)
    graph = graph_fns.build_graph(
        vocab_size = batcher.vocab_size(),
        num_embedding_units = config.NUM_EMBEDDING_UNITS,
        num_negative_samples = config.NUM_NEGATIVE_SAMPLES,
    )

    session.run(tf.global_variables_initializer())

    run_info = RunInfo(
        session = session,
        graph = graph,
        saver = tf.train.Saver(),
        batcher = batcher,
        batch_size = config.BATCH_SIZE,
        batches_per_epoch = num_batches,
        batches_per_logging = int(
            num_batches * config.LOGGING_FREQUENCY
        ),
    )

    for epoch_idx in range(1, config.NUM_EPOCHS + 1):
        run_epoch(run_info, epoch_idx)

with tf.Session() as session:
    run(session)
