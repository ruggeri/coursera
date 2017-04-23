from pong_constants import *

class PongState:
    def __init__(self, training_mode = False):
        self.training_mode = training_mode

        self.paddle1_pos = 0.5
        self.paddle2_pos = 0.5
        self.ball_pos, self.ball_vel = (
            initial_conditions(self.training_mode)
        )

    def copy(self):
        ps2 = PongState()

        ps2.training_mode = self.training_mode
        ps2.paddle1_pos = self.paddle1_pos
        ps2.paddle2_pos = self.paddle2_pos
        ps2.ball_pos = self.ball_pos.copy()
        ps2.ball_vel = self.ball_vel.copy()

        return ps2

    def nudge(self, player_num, action_num):
        ps2 = self.copy()
        if (player_num == PLAYER1) and (action_num == ACTION_UP):
            ps2.paddle1_pos -= PADDLE_SPEED
        elif (player_num == PLAYER1) and (action_num == ACTION_DOWN):
            ps2.paddle1_pos += PADDLE_SPEED
        elif (player_num == PLAYER2) and (action_num == ACTION_UP):
            ps2.paddle2_pos -= PADDLE_SPEED
        elif (player_num == PLAYER2) and (action_num == ACTION_DOWN):
            ps2.paddle2_pos += PADDLE_SPEED
        else:
            raise Exception(
                f"unknown player_num {player_num} or "
                f"action_num {action_num}"
            )

        ps2.paddle1_pos = max(ps2.paddle1_pos, 0)
        ps2.paddle1_pos = min(ps2.paddle1_pos, 1.0)
        ps2.paddle2_pos = max(ps2.paddle2_pos, 0)
        ps2.paddle2_pos = min(ps2.paddle2_pos, 1.0)

        return ps2

    def bounce(self, direction):
        ps2 = self.copy()
        if direction == DIR_X:
            ps2.ball_vel[1] *= -1
            ps2.ball_vel[0] += np.random.uniform(
                -BOUNCE_JITTER,
                +BOUNCE_JITTER
            )
        elif direction == DIR_Y:
            ps2.ball_vel[0] *= -1
        else:
            raise Exception("unknown direction")
        return ps2

    def move_ball(self):
        ps2 = self.copy()
        ps2.ball_pos += ps2.ball_vel
        return ps2

def new(training_mode = False):
    return PongState()

def constrain_ball_velocity(ball_vel):
    theta = np.arctan2(ball_vel[0], ball_vel[1])

    # Limit the angle from -pi/2 to pi/2
    theta /= (np.pi / 4)
    if (1 < theta) and (theta < 2):
        theta = 1
    elif (2 < theta) and (theta < 3):
        theta = 3
    elif (-2 < theta) and (theta < -1):
        theta = -1
    elif (-3 < theta) and (theta < -2):
        theta = -3
    theta *= (np.pi / 4)

    ball_vel = np.array([np.sin(theta), np.cos(theta)])
    ball_vel /= np.sqrt(np.sum(ball_vel ** 2))
    ball_vel *= BALL_SPEED

    return ball_vel

def paddle_pos(state, player_num):
    if player_num == PLAYER1:
        return state.paddle1_pos
    elif player_num == PLAYER2:
        return state.paddle2_pos
    else:
        raise Exception("no such player_num")

def nudge_paddle(state, player_num, action_num):
    return state.nudge(player_num, action_num)

def paddle_endpoints(state, player_num):
    paddle_pos_ = paddle_pos(state, player_num)
    return (
        paddle_pos_ - PADDLE_WIDTH / 2,
        paddle_pos_ + PADDLE_WIDTH / 2
    )

def ball_touches_wall(state):
    ball_pos_y = state.ball_pos[0]
    ball_vel_y = state.ball_vel[0]

    if (ball_pos_y < COLLISION_EPSILON) and (ball_vel_y < 0.0):
        return True
    elif ((1 - COLLISION_EPSILON) < ball_pos_y) and (0.0 < ball_vel_y):
        return True
    else:
        return False

def ball_touches_paddle(state, player_num):
    paddle_min, paddle_max = paddle_endpoints(state, player_num)
    if state.ball_pos[0] < paddle_min:
        return False
    elif paddle_max < state.ball_pos[0]:
        return False

    if player_num == PLAYER1:
        x_min, x_max = -np.inf, COLLISION_EPSILON
    elif player_num == PLAYER2:
        x_min, x_max = 1 - COLLISION_EPSILON, +np.inf
    else:
        raise Exception("unknown player_num")

    if (state.ball_pos[1] < x_min) or (x_max < state.ball_pos[1]):
        return False

    if (player_num == PLAYER1) and (0.0 < state.ball_vel[1]):
        return False
    if (player_num == PLAYER2) and (state.ball_vel[1] < 0.0):
        return False

    return True

def play_default_move(state, player_num):
    paddle_pos_ = paddle_pos(state, player_num)

    if state.ball_pos[0] < paddle_pos_:
        return state.nudge(player_num, ACTION_UP)
    else:
        return state.nudge(player_num, ACTION_DOWN)

def did_score_point(state):
    if state.ball_pos[1] < 0:
        return PLAYER2
    elif state.ball_pos[1] > 1:
        return PLAYER1
    else:
        return None

def evolve(state):
    state = state.move_ball()

    if ball_touches_paddle(state, PLAYER1):
        state = state.bounce(DIR_X)

    if ball_touches_paddle(state, PLAYER2):
        state = state.bounce(DIR_X)

    if ball_touches_wall(state):
        state = state.bounce(DIR_Y)

    if did_score_point(state):
        state = PongState()

    return state

def initial_conditions(training_mode):
    ball_pos = np.array([0.5, 0.5])
    ball_vel = constrain_ball_velocity(
        np.random.uniform(low = -1, high = +1, size = (2,))
    )

    if training_mode:
        ball_pos = np.random.uniform(
            low = 0, high = +1, size = (2,)
        )

    if (ball_vel[1] > 0):
        ball_vel[1] *= -1

    return (ball_pos, ball_vel)
