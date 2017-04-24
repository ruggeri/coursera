# This code tells us to take the locally optimal action.
def choose_action(session, graph, game_state):
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
