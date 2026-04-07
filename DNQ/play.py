# =============================================================================
# play.py  —  Watch the Trained Agent Play Snake
# =============================================================================
#
# WHAT THIS FILE DOES
# -------------------
# Training is done. The best weights are saved in best_model/snake_dqn.weights.h5
# This file:
#   1. Loads those weights into a fresh Q-network
#   2. Runs the agent with epsilon = 0.0 (NO random moves — pure exploitation)
#   3. Renders every frame so you can watch it play
#   4. Prints the score after each game
#
# WHY epsilon = 0.0 HERE?
# During training, epsilon stayed at 0.01 minimum so the agent occasionally
# explored random moves. Now that training is done, we want to see the best
# the agent can do — pure greedy decisions from the network, no randomness.
#
# =============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # silence TensorFlow logs

import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

from snake_game import SnakeGameAI


# =============================================================================
# CONFIGURATION
# =============================================================================

WEIGHTS_PATH = "best_model/snake_dqn.weights.h5"   # saved during training
NUM_GAMES    = 10      # how many games to watch (set to 0 for infinite)
SPEED_DELAY  = 0.08   # seconds between frames — lower = faster snake
                       # 0.05 = fast, 0.08 = comfortable, 0.15 = slow


# =============================================================================
# LOAD THE TRAINED NETWORK
# =============================================================================
# We rebuild the exact same architecture as agent.py and load the saved weights.
# We don't import DQNAgent here because we don't need the full training setup —
# just the network itself.

def load_network(weights_path, state_size=11, action_size=3):
    """
    Rebuild the Q-network and load the best saved weights.

    WHY REBUILD INSTEAD OF IMPORTING DQNAgent?
    DQNAgent builds TWO networks, a memory buffer, an optimiser, and sets up
    epsilon. For playing we only need the Q-network. Keeping play.py lightweight
    means it loads faster and has no unnecessary dependencies.
    """
    model = Sequential([
        Input(shape=(state_size,)),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(action_size, activation='linear'),
    ])

    if not os.path.exists(weights_path):
        print(f"ERROR: Could not find weights at '{weights_path}'")
        print("Make sure you've run train.py first and best_model/ exists.")
        quit()

    model.load_weights(weights_path)
    print(f"Weights loaded from: {weights_path}")
    return model


def get_action(model, state):
    """
    Pick the best action for the current state — no randomness.

    During training:  action = random  OR  argmax(Q(state))  based on epsilon
    During play:      action = argmax(Q(state))  always

    This is pure exploitation — the agent does exactly what its network
    thinks is best every single step.
    """
    state_tensor = tf.expand_dims(
        tf.convert_to_tensor(state, dtype=tf.float32), axis=0
    )
    q_values = model(state_tensor, training=False)   # shape: (1, 3)
    return int(tf.argmax(q_values[0]).numpy())        # index of best action


# =============================================================================
# PLAY LOOP
# =============================================================================

def play():
    print("=" * 50)
    print("  Snake DQN — Watch Mode")
    print(f"  Speed delay : {SPEED_DELAY}s per frame")
    print(f"  Games       : {'infinite' if NUM_GAMES == 0 else NUM_GAMES}")
    print("=" * 50)

    # Load the trained network
    model = load_network(WEIGHTS_PATH)

    # Create the game with rendering ON — we want to watch every frame
    env = SnakeGameAI(render=True)

    scores    = []
    game_num  = 0

    while True:
        game_num += 1
        if NUM_GAMES > 0 and game_num > NUM_GAMES:
            break

        # -- Start a new game ------------------------------------------------
        state = env.reset()
        done  = False
        score = 0

        print(f"\nGame {game_num} starting...")

        # -- Play one full game ----------------------------------------------
        while not done:
            # Agent picks the best action — no randomness
            action = get_action(model, state)

            # Game executes the action
            next_state, reward, done = env.step(action)

            # Track score
            if reward == 10:
                score += 1

            # Advance state
            state = next_state

            # Control playback speed
            # Without this delay the snake moves so fast you can't see it
            time.sleep(SPEED_DELAY)

        # -- Game over -------------------------------------------------------
        scores.append(score)
        print(f"Game {game_num:>3} finished | Score: {score:>3} | Best so far: {max(scores):>3}")

    # -- Summary after all games ---------------------------------------------
    if scores:
        print("\n" + "=" * 50)
        print(f"  Games played : {len(scores)}")
        print(f"  Best score   : {max(scores)}")
        print(f"  Worst score  : {min(scores)}")
        print(f"  Average      : {sum(scores)/len(scores):.2f}")
        print("=" * 50)

    import pygame
    pygame.quit()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    play()