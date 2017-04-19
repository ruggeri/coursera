from collections import namedtuple
import config
import dataset
import classification_graph
import tensorflow as tf

BatchInfo = namedtuple("BatchInfo", [
    "epoch_idx",
    "batch_idx",
    "inputs",
    "labels",
    "num_batches",
])
RunInfo = namedtuple("RunInfo", [
    "session",
    "graph",
    "dataset",
    "saver",
    "summary_file_writer",
])

def run_batch(run_info, batch_info):
    ri = run_info

    avg_cost, accuracy, summaries, _ = ri.session.run([
        ri.graph.avg_cost,
        ri.graph.accuracy,
        ri.graph.summaries,
        ri.graph.optimization_op,
    ], feed_dict = {
        ri.graph.inputs: batch_info.inputs,
        ri.graph.labels: batch_info.labels,
        ri.graph.keep_probability: config.KEEP_PROBABILITY,
    })
    ri.summary_file_writer.add_summary(
        summaries, global_step = batch_info.batch_idx
    )

    print(f"Epoch: {batch_info.epoch_idx}/{config.NUM_EPOCHS}",
          f"Batch: {batch_info.batch_idx}/{batch_info.num_batches}",
          f"Train loss: {avg_cost:.3f}",
          f"accuracy: {accuracy:.3f}")

def run_validation(run_info, batch_info):
    ri = run_info

    validation_accuracy = 0.0
    (num_batches, batches) = ri.dataset.get_batches("validation")
    for inputs, labels in batches:
        batch_accuracy = ri.session.run(ri.graph.accuracy, feed_dict = {
            ri.graph.inputs: inputs,
            ri.graph.labels: labels,
            ri.graph.keep_prob: 1.0,
        })
        validation_accuracy += batch_accuracy

    validation_accuracy /= num_batches
    print(f"Val acc: {validation_accuracy:.3f}")

def run_epoch(run_info, epoch_idx):
    (num_batches, batches) = run_info.dataset.get_batches("training")
    for batch_idx, (inputs, labels) in enumerate(batches, 1):
        batch_info = BatchInfo(
            epoch_idx = epoch_idx,
            batch_idx = batch_idx,
            inputs = inputs,
            labels = labels,
            num_batches = num_batches
        )

        run_batch(run_info, batch_info)

        if batch_idx % config.BATCHES_PER_VALIDATION == 0:
            run_validation(run_info, batch_info)

def run(session):
    ds = dataset.Dataset()
    run_info = RunInfo(
        session = session,
        graph = classification_graph.build_graph(
            ds.num_code_units(), ds.num_labels()
        ),
        dataset = ds,
        saver = tf.train.Saver(),
        summary_file_writer = tf.summary.FileWriter(config.TBOARD_NAME),
    )

    run_info.summary_file_writer.add_graph(session.graph)

    session.run(tf.global_variables_initializer())
    for epoch_idx in range(config.NUM_EPOCHS):
        run_epoch(run_info, epoch_idx)
        run_info.saver.save(session, config.CHECKPOINT_NAME)

with tf.Session() as session:
    run(session)
