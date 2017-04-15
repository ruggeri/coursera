from collections import namedtuple
import config
import graph as graph_fns
import preprocessing
import tensorflow as tf
import time
import validator

RunInfo = namedtuple("RunInfo", [
    "session",
    "graph",
    "saver",
    "batcher",
    "validator",
    "batch_size",
    "window_size",
    "batches_per_epoch",
    "batches_per_logging",
    "batches_per_save",
    "batches_per_validation",
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

    average_loss = cumulative_loss / ri.batches_per_logging
    examples_per_sec = int(num_examples / (end_time - start_time))
    print(f"Epoch: {bi.epoch_idx:03d} | "
          f"Batch: {bi.batch_idx:04d} / {ri.batches_per_epoch:04d} | "
          f"Avg Train Loss: {average_loss:.2f} | "
          f"{examples_per_sec:04d} sec / example"
    )

def save(run_info, batch_info):
    ri, bi = run_info, batch_info

    run_info.saver.save(
        run_info.session,
        f"{config.SAVE_BASENAME}-{bi.epoch_idx:03d}-{bi.batch_idx:04d}"
    )
    print(f"Epoch: {bi.epoch_idx:03d} | "
          f"Batch: {bi.batch_idx:04d} / {ri.batches_per_epoch:04d} | "
          f"Model saved!"
    )

def run_epoch(run_info, epoch_idx):
    batches = run_info.batcher.batches(
        run_info.batch_size, run_info.window_size
    )

    cumulative_loss, start_time = 0, time.time()
    for batch_idx, (inputs, labels) in enumerate(batches, 1):
        batch_info = BatchInfo(
            epoch_idx = epoch_idx,
            batch_idx = batch_idx,
            inputs = inputs,
            labels = labels,
        )

        cumulative_loss += run_batch(run_info, batch_info)

        should_log = (batch_idx % run_info.batches_per_logging) == 0
        if should_log:
            log_batches(
                run_info, batch_info, cumulative_loss, start_time
            )

        should_save = (batch_idx % run_info.batches_per_save) == 0
        if should_save:
            save(run_info, batch_info)
        should_validate = (
            (batch_idx % run_info.batches_per_validation) == 0
        )
        if should_validate:
            run_info.validator.run_and_log(run_info, batch_info)

        if should_log:
            # Doing this at the very end to not include the saving or
            # validation time.
            cumulative_loss, start_time = 0, time.time()

def run(session):
    batcher = preprocessing.Batcher(
        config.SUBSAMPLE_THRESHOLD, config.TEST_MODE
    )
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
        validator = validator.Validator(
            batcher.vocab_size(),
            graph.embedding_matrix
        ),
        batch_size = config.BATCH_SIZE,
        window_size = config.WINDOW_SIZE,
        batches_per_epoch = num_batches,
        batches_per_logging = int(
            num_batches * config.LOGGING_FREQUENCY
        ),
        batches_per_save = int(
            num_batches * config.SAVING_FREQUENCY
        ),
        batches_per_validation = int(
            num_batches * config.VALIDATION_FREQUENCY
        ),
    )

    for epoch_idx in range(1, config.NUM_EPOCHS + 1):
        run_epoch(run_info, epoch_idx)

with tf.Session() as session:
    run(session)
