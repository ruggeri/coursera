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

def play_action_idx(game, chosen_action_idx):
    chosen_action_name = example.action_idx_to_action_name(
        chosen_action_idx
    )

    game.nudge_paddle(pong_constants.PLAYER1, chosen_action_name)
    game.play_default_move(pong_constants.PLAYER2)
    game.evolve()

def evaluate_performance(
        session, graph, training_mode, callback = None):

    game = pong.PongGame(training_mode = training_mode)

    def total_points(game):
        return pong_stats.total_points(game.stats)

    while total_points(game) < config.NUM_POINTS_PER_EVALUATION:
        prev_game_state = example.model_state(game.state)
        chosen_action_idx = choose_action(
            session, graph, prev_game_state
        )
        play_action_idx(game, chosen_action_idx)

        if callback:
            callback()

        if total_points(game) % config.POINTS_PER_LOG == 0:
            print("eval point #{total_points(game)}")
            print(game.stats)
