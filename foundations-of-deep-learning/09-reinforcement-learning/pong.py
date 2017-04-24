import pong_state
import pong_stats

class PongGame:
    def __init__(self, training_mode):
        self.training_mode = training_mode
        self.stats = pong_stats.new()
        self.state = pong_state.new()

    def score(self):
        return pong_stats.score(self.stats)

    def evolve(self):
        self.state, events = pong_state.evolve(self.state)
        self.stats = pong_stats.add_events(self.stats, events)

    def play_default_move(self, player_num):
        self.state = pong_state.play_default_move(
            self.state, player_num
        )

    def nudge_paddle(self, player_num, action_num):
        self.state = pong_state.nudge_paddle(
            self.state, player_num, action_num
        )
