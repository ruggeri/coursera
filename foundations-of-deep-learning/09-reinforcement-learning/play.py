import asyncio
import config
import example
import numpy as np
import pong
import pong_constants
import pong_stats

def choose_action(session, graph, game_state):
    q_values = session.run(
        graph.q_values,
        feed_dict = {
            graph.prev_game_states: (
                game_state.reshape((1, config.NUM_STATE_DIMENSIONS))
            )
        }
    )

    q_values = q_values.reshape((config.NUM_ACTIONS,))
    best_action = np.argmax(q_values)
    return best_action

# This code tells us to take the locally optimal action.
def choose_best_action(session, graph, game_state):
    import pong_state

    game = pong.PongGame(training_mode = True)
    game.state.paddle1_pos = game_state[0]
    game.state.paddle2_pos = game_state[1]
    game.state.ball_pos = np.array([game_state[2], game_state[3]])
    game.state.ball_vel = np.array([game_state[4], game_state[5]])
    prev_state = game.state
    prev_stats = game.stats
    play_action_idx(game, 0)
    next_state = game.state
    next_stats = game.stats
    reward0 = example.reward(prev_state, prev_stats, next_state, next_stats)

    game = pong.PongGame(training_mode = True)
    game.state.paddle1_pos = game_state[0]
    game.state.paddle2_pos = game_state[1]
    game.state.ball_pos = np.array([game_state[2], game_state[3]])
    game.state.ball_vel = np.array([game_state[4], game_state[5]])
    prev_state = game.state
    prev_stats = game.stats
    play_action_idx(game, 1)
    next_state = game.state
    next_stats = game.stats
    reward1 = example.reward(prev_state, prev_stats, next_state, next_stats)

    return 1 if reward0 < reward1 else 0


def play_action_idx(game, chosen_action_idx):
    chosen_action_name = example.action_idx_to_action_name(
        chosen_action_idx
    )

    game.nudge_paddle(pong_constants.PLAYER1, chosen_action_name)
    game.play_default_move(pong_constants.PLAYER2)
    game.evolve()

def evaluate_performance(session, graph, training_mode):
    asyncio.get_event_loop().run_until_complete(
        async_evaluate_performance(session, graph, training_mode)
    )

async def async_evaluate_performance(
        session, graph, training_mode, callback = None):

    game = pong.PongGame(training_mode = training_mode)

    def total_points(game):
        return pong_stats.total_points(game.stats)

    while total_points(game) < config.NUM_POINTS_PER_EVALUATION:
        prev_stats = game.stats
        prev_game_state = example.model_state(game.state)
        if config.CHOOSE_BEST_ALWAYS:
            chosen_action_idx = choose_best_action(
                session, graph, prev_game_state
            )
        else:
            chosen_action_idx = choose_action(
                session, graph, prev_game_state
            )
        play_action_idx(game, chosen_action_idx)
        next_stats = game.stats

        if callback:
            await callback(game)

        points_did_change = (
            pong_stats.total_points(prev_stats)
            != pong_stats.total_points(next_stats)
        )
        should_log = (
            points_did_change
            and (total_points(game) % config.POINTS_PER_LOG == 0)
        )
        if should_log:
            print(f"eval point #{total_points(game)}")
            print(game.stats)
