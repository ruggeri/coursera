from collections import namedtuple
import config
import numpy as np
import tensorflow as tf

Variables = namedtuple("Variables", [
    "weights1",
    "biases1",
    "weights2",
    "biases2",
    "weights3",
    "biases3",
    "weights4",
    "biases4",
    "weights5",
    "biases5",
])

Graph = namedtuple("Graph", [
    # Training inputs
    "prev_game_states",
    "chosen_actions",
    "collected_rewards",
    "next_game_states",
    "did_episodes_end",
    "is_training",

    # Training constants
    "reward_decay",

    # Training operations and loss
    "training_op",
    "avg_loss",

    # Evaluation output
    "q_values",
])

def build_variables():
    weights1 = tf.Variable(
        tf.truncated_normal(
            [config.NUM_STATE_DIMENSIONS, config.NUM_HIDDEN_UNITS],
            stddev = 1 / np.sqrt(
                config.NUM_STATE_DIMENSIONS
                +
                config.NUM_HIDDEN_UNITS
            )
        )
    )
    biases1 = tf.Variable(tf.zeros([config.NUM_HIDDEN_UNITS]))

    weights2 = tf.Variable(
        tf.truncated_normal(
            [config.NUM_HIDDEN_UNITS, config.NUM_HIDDEN_UNITS],
            stddev = 1 / np.sqrt(
                config.NUM_HIDDEN_UNITS
                +
                config.NUM_HIDDEN_UNITS
            )
        )
    )
    biases2 = tf.Variable(tf.zeros([config.NUM_HIDDEN_UNITS]))

    weights3 = tf.Variable(
        tf.truncated_normal(
            [config.NUM_HIDDEN_UNITS, config.NUM_HIDDEN_UNITS],
            stddev = 1 / np.sqrt(
                config.NUM_HIDDEN_UNITS
                +
                config.NUM_HIDDEN_UNITS
            )
        )
    )
    biases3 = tf.Variable(tf.zeros([config.NUM_HIDDEN_UNITS]))

    weights4 = tf.Variable(
        tf.truncated_normal(
            [config.NUM_HIDDEN_UNITS, config.NUM_HIDDEN_UNITS],
            stddev = 1 / np.sqrt(
                config.NUM_HIDDEN_UNITS
                +
                config.NUM_HIDDEN_UNITS
            )
        )
    )
    biases4 = tf.Variable(tf.zeros([config.NUM_HIDDEN_UNITS]))

    weights5 = tf.Variable(
        tf.truncated_normal(
            [config.NUM_HIDDEN_UNITS, config.NUM_ACTIONS],
            stddev = 1 / np.sqrt(
                config.NUM_HIDDEN_UNITS
                +
                config.NUM_ACTIONS
            )
        )
    )
    biases5 = tf.Variable(tf.zeros([config.NUM_ACTIONS]))

    return Variables(
        weights1 = weights1,
        biases1 = biases1,
        weights2 = weights2,
        biases2 = biases2,
        weights3 = weights3,
        biases3 = biases3,
        weights4 = weights4,
        biases4 = biases4,
        weights5 = weights5,
        biases5 = biases5,
    )

def q_values(game_states, variables, is_training):
    fc1 = tf.nn.relu(
        tf.matmul(game_states, variables.weights1) + variables.biases1
    )
    fc1 = tf.layers.batch_normalization(fc1, training = is_training)
    fc2 = tf.nn.relu(
        tf.matmul(fc1, variables.weights2) + variables.biases2
    )
    fc2 = tf.layers.batch_normalization(fc2, training = is_training)
    fc3 = tf.nn.relu(
        tf.matmul(fc2, variables.weights3) + variables.biases3
    )
    fc3 = tf.layers.batch_normalization(fc3, training = is_training)
    fc4 = tf.nn.relu(
        tf.matmul(fc3, variables.weights4) + variables.biases4
    )
    fc4 = tf.layers.batch_normalization(fc4, training = is_training)
    # TODO: Added tanh to capture the idea that reward is either
    # -1 or +1.
    q_values = 100 * tf.tanh(
        tf.matmul(fc4, variables.weights5) + variables.biases5
    )

    return q_values

def build_graph():
    prev_game_states = tf.placeholder(
        tf.float32,
        [None, config.NUM_STATE_DIMENSIONS]
    )

    chosen_actions = tf.placeholder(
        tf.int32, [None]
    )

    collected_rewards = tf.placeholder(
        tf.float32, [None]
    )

    next_game_states = tf.placeholder(
        tf.float32,
        [None, config.NUM_STATE_DIMENSIONS],
        name = "next_game_states"
    )

    did_episodes_end = tf.placeholder(
        tf.bool,
        [None]
    )

    reward_decay = tf.placeholder(
        tf.float32
    )

    is_training = tf.placeholder(tf.bool)

    variables = build_variables()

    # Calculate our expectation of the reward from the chosen action
    # at the previous state.
    prev_game_state_q_values = q_values(prev_game_states, variables, is_training)
    one_hot_chosen_actions = tf.one_hot(
        chosen_actions, config.NUM_ACTIONS
    )
    chosen_action_old_q_values = tf.reduce_sum(
        tf.multiply(prev_game_state_q_values, one_hot_chosen_actions),
        axis = 1
    )

    # Calculate the expectation of the reward from the next state if
    # we go on to choose the best action.
    next_game_state_q_values = q_values(next_game_states, variables, is_training)
    next_game_state_q_values = tf.reduce_max(
        next_game_state_q_values,
        axis = 1
    )
    next_game_state_q_values = tf.multiply(
        next_game_state_q_values,
        (1 - tf.cast(did_episodes_end, tf.float32))
    )
    # TODO: this seems to cause divergence problems!!
    next_game_state_q_values = tf.stop_gradient(
        next_game_state_q_values
    )

    # Calculate the discrepency between (1) the expected value of the
    # action in the prev state and (2) the sum of (a) the received
    # reward and (b) the expected reward from the future state.
    errors = chosen_action_old_q_values - (
        collected_rewards
        + (reward_decay * next_game_state_q_values)
    )

    avg_loss = tf.reduce_mean(tf.square(errors))

    training_op = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(
        avg_loss
    )

    return Graph(
        # Training inputs
        prev_game_states = prev_game_states,
        chosen_actions = chosen_actions,
        collected_rewards = collected_rewards,
        next_game_states = next_game_states,
        did_episodes_end = did_episodes_end,
        is_training = is_training,

        # Training constants
        reward_decay = reward_decay,

        # Training operations and loss
        training_op = training_op,
        avg_loss = avg_loss,

        # Evaluation ouptut
        q_values = prev_game_state_q_values,
    )
