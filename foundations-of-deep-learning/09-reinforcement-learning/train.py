from collections import namedtuple
import config
import example
import memory as m
import numpy as np
import play
import pong
import pong_stats

RunInfo = namedtuple("RunInfo", [
    "session",
    "graph",
    "memory",
])

BatchInfo = namedtuple("BatchInfo", [
    "epoch_idx",
    "batch_idx",
    "exploration_rate",
    "reward_decay"
])

def log_batch(batch_info, losses, games):
    epoch_idx, batch_idx = batch_info.epoch_idx, batch_info.batch_idx

    if len(losses) > 0:
        avg_loss = sum(losses) / len(losses)
    else:
        avg_loss = -1

    print(f"Epoch {epoch_idx} | Batch {batch_idx} | "
          f"Avg Loss {avg_loss}")

    reduced_stats = pong_stats.reduce([game.stats for game in games])
    print(reduced_stats)
    print(reduced_stats.p2_bounces / (reduced_stats.p1_bounces + 0.01))
    print(f"exploration rate: {batch_info.exploration_rate:0.2f}")

def train_batch(run_info, batch_info):
    session, graph, memory = (
        run_info.session, run_info.graph, run_info.memory
    )

    if memory.num_points() == 0:
        return None

    examples = memory.training_batch()

    prev_game_states = np.array(
        [example.prev_game_state for example in examples]
    )
    chosen_actions = np.array(
        [example.chosen_action for example in examples]
    )
    collected_rewards = np.array(
        [example.collected_reward for example in examples]
    )
    next_game_states = np.array(
        [example.next_game_state for example in examples]
    )
    did_episodes_end = np.array(
        [example.did_episode_end for example in examples]
    )

    avg_loss, _ = session.run(
        [graph.avg_loss, graph.training_op],
        feed_dict = {
            graph.prev_game_states: prev_game_states,
            graph.chosen_actions: chosen_actions,
            graph.collected_rewards: collected_rewards,
            graph.next_game_states: next_game_states,
            graph.reward_decay: batch_info.reward_decay,
            graph.did_episodes_end: did_episodes_end,
            graph.is_training: True
        }
    )

    return avg_loss

def train_epoch(run_info, epoch_idx):
    games = [
        pong.PongGame(training_mode = config.TRAINING_MODE)
        for
        _ in range(config.GAME_BATCH_SIZE)
    ]

    losses = []
    batch_idxs = range(1, config.NUM_BATCHES_PER_EPOCH + 1)
    for batch_idx in batch_idxs:
        bi = batch_info(epoch_idx, batch_idx)

        if (not run_info.memory.is_full()) or (batch_idx % 10 == 0):
            example.generate_data_all(run_info, bi, games)
        batch_loss = train_batch(run_info, bi)

        if batch_loss:
            losses.append(batch_loss)
        if batch_idx % config.BATCHES_PER_LOG == 0:
            log_batch(bi, losses, games)
            losses = []

def batch_info(epoch_idx, batch_idx):
    exploration_rate = config.EXPLORATION_START_RATE
    exploration_rate *= 1 - config.EXPLORATION_DECAY_RATE

    exploration_rate = max(
        exploration_rate, config.EXPLORATION_RATE_MIN
    )

    reward_decay = config.REWARD_DECAY_START
    reward_decay *= (1 + config.REWARD_DECAY_GROWTH_FACTOR)
    reward_decay = min(reward_decay, config.REWARD_DECAY_MAX)

    return BatchInfo(
        epoch_idx = epoch_idx,
        batch_idx = batch_idx,
        exploration_rate = exploration_rate,
        reward_decay = reward_decay
    )

def train(session, graph, saver):
    ri = RunInfo(
        session = session,
        graph = graph,
        memory = m.Memory(),
    )

    for epoch_idx in range(1, config.NUM_EPOCHS + 1):
        train_epoch(ri, epoch_idx)

        print("Saving!")
        saver.save(session, config.CHECKPOINT_FILENAME)
        if epoch_idx % config.NUM_EPOCHS_PER_EVAL == 0:
            play.evaluate_performance(
                session, graph, training_mode = False
            )
