import batcher
from collections import namedtuple
import config
import graph as graph_fns
import numpy as np
import tensorflow as tf

BatchInfo = namedtuple(
    "BatchInfo", [
        "epoch_idx",
        "batch_idx",
        "batch_x",
        "batch_y",
        "states",
    ]
)

RunInfo = namedtuple(
    "RunInfo", [
        "train_batches",
        "validation_batches",
        "num_training_batches",
        "num_validation_batches",
        "batches_per_save",
        "batches_per_validation",
        "graph",
        "saver",
    ]
)

def run_validation(session, run_info, batch_info):
    ri = run_info
    graph = ri.graph

    states = graph_fns.make_initial_states(
        config.BATCH_SIZE, config.NUM_LAYERS, config.NUM_LSTM_UNITS
    )

    total_loss = 0
    total_accuracy = 0
    for (validation_x, validation_y) in ri.validation_batches:
        validation_results = session.run({
            "avg_loss": graph.avg_loss,
            "accuracy": graph.accuracy,
            "final_states": graph.final_states
        }, feed_dict={
            graph.inputs: validation_x,
            graph.outputs: validation_y,
            tuple(graph.initial_states): tuple(states)
        })
        total_loss += (
            validation_results["avg_loss"] / ri.num_validation_batches
        )
        total_accuracy += (
            validation_results["accuracy"] / ri.num_validation_batches
        )
        states = validation_results["final_states"]

    return (total_loss, total_accuracy)

def run_batch(session, run_info, batch_info):
    epoch_idx, batch_idx = batch_info.epoch_idx, batch_info.batch_idx
    graph = run_info.graph

    results = session.run({
        "final_states": graph.final_states,
        "avg_loss": graph.avg_loss,
        "accuracy": graph.accuracy,
        "train_op": graph.train_op
    }, feed_dict={
        graph.inputs: batch_info.batch_x,
        graph.outputs: batch_info.batch_y,
        tuple(graph.initial_states): tuple(batch_info.states)
    })
    states = results["final_states"]

    should_run_validation = (
        (batch_info.batch_idx + 1) % run_info.batches_per_validation
    )
    if should_run_validation == 0:
        validation_loss, validation_accuracy = run_validation(
            session, run_info, batch_info
        )

        print(f"Epoch {epoch_idx}, "
              f"Batch {batch_idx}/{run_info.num_training_batches} | "
              f"Training Avg Loss: {results['avg_loss']:.4f} | "
              f"Training Accuracy: {results['accuracy']:.4f} | "
              f"Valid Avg Loss {validation_loss:0.4f} | "
              f"Valid Accuracy: {validation_accuracy:.4f}")
    else:
        print(f"Epoch {epoch_idx}, "
              f"Batch {batch_idx}/{run_info.num_training_batches} | "
              f"Training Avg Loss: {results['avg_loss']:.4f} | "
              f"Training Accuracy: {results['accuracy']:.4f}")

    should_save = (
        (batch_info.batch_idx + 1) % run_info.batches_per_save == 0
    )
    if should_save:
        save_name = (
            f"{config.SAVER_BASENAME}-"
            f"{epoch_idx:02d}-{batch_idx:04d}.ckpt"
        )
        print(f"Saving model: {save_name}!")
        run_info.saver.save(session, save_name)

    return states

def run_epoch(session, run_info, epoch_idx):
    initial_states = graph_fns.make_initial_states(
        config.BATCH_SIZE, config.NUM_LAYERS, config.NUM_LSTM_UNITS
    )

    states = initial_states[:]
    train_batches = run_info.train_batches
    for batch_idx, (batch_x, batch_y) in enumerate(train_batches):
        batch_info = BatchInfo(
            epoch_idx = epoch_idx,
            batch_idx = batch_idx,
            batch_x = batch_x,
            batch_y = batch_y,
            states = states
        )

        states = run_batch(session, run_info, batch_info)

def run():
    train_batches, validation_batches = batcher.partition_batches(
        batcher.make_batches(
            config.file_reader.one_hot_text(),
            config.BATCH_SIZE,
            config.STEP_SIZE
        )
    )

    num_training_batches = len(train_batches)
    num_validation_batches = len(validation_batches)
    batches_per_save = int(config.SAVE_FREQUENCY * num_training_batches)
    batches_per_validation = int(
        config.VALIDATION_FREQUENCY * num_training_batches
    )

    graph = graph_fns.build_graph(
        config.BATCH_SIZE,
        config.file_reader.vocab_size(),
        config.NUM_LAYERS,
        config.STEP_SIZE,
        config.NUM_LSTM_UNITS
    )

    saver = tf.train.Saver()

    run_info = RunInfo(
        train_batches = train_batches,
        validation_batches = validation_batches,
        num_training_batches = num_training_batches,
        num_validation_batches = num_validation_batches,
        batches_per_save = batches_per_save,
        batches_per_validation = batches_per_validation,
        graph = graph,
        saver = saver
    )

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch_idx in range(config.NUM_EPOCHS):
            run_epoch(session, run_info, epoch_idx)

run()
