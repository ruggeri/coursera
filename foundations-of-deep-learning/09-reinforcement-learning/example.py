from collections import namedtuple

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
    if (next_state.p2_points - prev_state.p2_points) == 1:
        return -1
    if (next_state.p1_bounces - prev_state.p1_bounces) == 1:
        return +1

    distance_change = (
        pong_state.paddle_pos(prev_state, pong_constants.PLAYER2)
        - pong_state.paddle_pos(prev_state, pong_constants.PLAYER1)
    )

    if (bounces_change != 0):
        return +10
    if (score_change != 0):
        return -10
    return -prev_paddle_distance

def did_episode_end(prev_stats, next_stats):
    if (next_state.p2_points - prev_state.p2_points) == 1:
        return True
    if (next_state.p1_bounces - prev_state.p1_bounces) == 1:
        return True
    return False

def generate_data(session, graph, exploration_rate, memory, game):
    prev_game_state, prev_stats = game.state, game.stats
    if np.random.random() < exploration_rate:
        chosen_action = np.random.randint(0, config.NUM_ACTIONS)
    else:
        chosen_action = choose_action(
            session, graph, model_state(prev_game_state)
        )

    play_action(game, chosen_action)
    next_game_state, next_stats = game.state, game.stats

    reward_ = reward(
        prev_game_state, prev_stats, next_game_state, next_stats
    )
    did_episode_end_ = did_episode_end(prev_stats, next_stats)

    e = example.Example(
        prev_game_state = model_state(prev_game_state),
        chosen_action = chosen_action,
        collected_reward = reward_,
        next_game_state = model_state(next_game_state),
        did_episode_end = did_episode_end_,
    )

    memory.add_example(e)
