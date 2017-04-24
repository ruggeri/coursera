import numpy as np

ACTION_UP = "ACTION_UP"
ACTION_DOWN = "ACTION_DOWN"
BALL_SPEED = 0.02
BOUNCE_JITTER = 0.01
COLLISION_EPSILON = 0.02
DIR_X = "DIR_X"
DIR_Y = "DIR_Y"
# The idea here is that the maximum angle allowed is 45deg. That means
# the maximum transverse speed is BALL_SPEED * np.sqrt(2)/2. If the
# paddle can move that fast, it can never lose the ball. So we can
# make it a little bit slower.
PADDLE_SPEED = 0.9 * BALL_SPEED * np.sqrt(2) / 2
PADDLE_WIDTH = 0.1
PLAYER1 = "PLAYER1"
PLAYER2 = "PLAYER2"
POLL_FREQUENCY = 0.015
