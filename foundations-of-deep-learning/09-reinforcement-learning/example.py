from collections import namedtuple
import config
import numpy as np
import play
import pong_constants
import pong_state

# TODO: can tweak reward, did_episode_end, and generate_data

Example = namedtuple("Example", [
    "prev_game_state",
    "chosen_action",
    "collected_reward",
    "next_game_state",
    "did_episode_end",
])

def model_state(game_state):
    return np.array([
        game_state.paddle1_pos,
        game_state.paddle2_pos,
        game_state.ball_pos[0],
        game_state.ball_pos[1],
        game_state.ball_vel[0],
        game_state.ball_vel[1],
    ])

def reward(prev_state, prev_stats, next_state, next_stats):
    if config.REWARD_TYPE == "BALL_FOLLOW_REWARD":
        r = ball_follow_reward(
            prev_state, prev_stats, next_state, next_stats
        )
    elif config.REWARD_TYPE == "IDEAL_ANTICIPATION_REWARD":
        r = ideal_anticipation_reward(
            prev_state, prev_stats, next_state, next_stats
        )
    else:
        raise Exception("Unknown reward setting")

    if config.REWARD_BOUNCES:
        r += reward_bounce(
            prev_state, prev_stats, next_state, next_stats
        )

    if np.random.uniform() >= config.REWARD_PROBABILITY:
        return 0.0

    scale_factor = config.REWARD_SCALING_FACTOR
    scale_factor *= 1 + (
        config.SCALE_REWARD_BY_DISTANCE_TO_PADDLE
        * (1 - next_state.ball_pos[1])
    )

    return scale_factor * r

def ball_follow_reward(prev_state, prev_stats, next_state, next_stats):
    prev_distance = pong_state.distance_to_ball(
        prev_state, pong_constants.PLAYER1
    )
    next_distance = pong_state.distance_to_ball(
        next_state, pong_constants.PLAYER1
    )

    return -((next_distance ** config.DISTANCE_ERROR_POWER)
             - (prev_distance) ** config.DISTANCE_ERROR_POWER)

def ideal_anticipation_reward(
        prev_state, prev_stats, next_state, next_stats):
    prev_ideal_distance = pong_state.ideal_distance(
        prev_state, pong_constants.PLAYER1
    )
    next_ideal_distance = pong_state.ideal_distance(
        next_state, pong_constants.PLAYER1
    )

    return -((next_ideal_distance ** config.DISTANCE_ERROR_POWER)
             - (prev_ideal_distance) ** config.DISTANCE_ERROR_POWER)

def reward_bounce(prev_state, prev_stats, next_state, next_stats):
    return (
        (next_stats.p1_bounces - prev_stats.p1_bounces)
        + (prev_stats.p2_points - next_stats.p2_points)
    )

def did_episode_end(prev_state, prev_stats, next_state, next_stats):
    if (next_stats.p2_points - prev_stats.p2_points) == 1:
        return True
    if (next_stats.p1_bounces - prev_stats.p1_bounces) == 1:
        return True
    return False

def generate_data(run_info, batch_info, game):
    session, graph, memory = (
        run_info.session, run_info.graph, run_info.memory
    )
    exploration_rate = batch_info.exploration_rate

    prev_game_state, prev_stats = game.state, game.stats
    if np.random.random() < exploration_rate:
        chosen_action = np.random.randint(0, config.NUM_ACTIONS)
    else:
        chosen_action = play.choose_action(
            session, graph, model_state(prev_game_state)
        )

    play.play_action_idx(game, chosen_action)
    next_game_state, next_stats = game.state, game.stats

    reward_ = reward(
        prev_game_state, prev_stats, next_game_state, next_stats
    )
    did_episode_end_ = did_episode_end(
        prev_game_state, prev_stats, next_game_state, next_stats
    )

    e = Example(
        prev_game_state = model_state(prev_game_state),
        chosen_action = chosen_action,
        collected_reward = reward_,
        next_game_state = model_state(next_game_state),
        did_episode_end = did_episode_end_,
    )

    memory.add_example(e)

UP_ACTION_IDX = 0
DOWN_ACTION_IDX = 1
def action_idx_to_action_name(action_idx):
    if action_idx == UP_ACTION_IDX:
        return pong_constants.ACTION_UP
    elif action_idx == DOWN_ACTION_IDX:
        return pong_constants.ACTION_DOWN
    else:
        raise Exception("Unexpected pong action!")

def action_name_to_action_idx(action_name):
    if action_name == ACTION_UP:
        return UP_ACTION_IDX
    elif action_name == ACTION_DOWN:
        return DOWN_ACTION_IDX
    else:
        raise Exception("Unexpected pong action!")
