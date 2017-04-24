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

def play_action(game, chosen_action):
    game.nudge_paddle(pong_constants.PLAYER1, chosen_action)
    game.play_default_move(pong.PLAYER2)
    game.evolve()

def evaluate_performance(
        session, graph, training_mode, callback = None):

    game = pong.PongGame(training_mode = training_mode)

    def total_points(game):
        return pong_stats.total_points(game.stats)

    while total_points(game) < config.NUM_POINTS_PER_EVALUATION:
        prev_game_state = model_state(game.state)
        chosen_action = choose_action(session, graph, prev_game_state)
        play_action(game, chosen_action)

        if callback:
            await callback()

        if total_points(game) % config.POINTS_PER_LOG == 0:
            print("eval point #{total_points(game)}")
            print(game.stats)
