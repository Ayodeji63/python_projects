# =============================================================================
# train.py  —  The Training Loop
# =============================================================================
#
# DESIGN PHILOSOPHY
# -----------------
# This file is the "conductor" — it owns the loop that connects snake_game.py
# and agent.py together. Neither of those files knows about each other.
# train.py is the only place that talks to both.
#
# The loop follows the exact agent-environment pattern from Ng's notebook:
#
#   for episode in range(MAX_EPISODES):
#       state = env.reset()
#       while not done:
#           action     = agent.get_action(state)
#           next_state, reward, done = env.step(action)
#           agent.remember(...)
#           agent.learn()
#           state = next_state
#       agent.decay_epsilon()
#
# On top of that loop, we add:
#   - Live score plotting (matplotlib) so you can watch the agent improve
#   - Model saving when a new high score is reached
#   - A printed summary after every episode
#
# =============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # silence TensorFlow info/warning logs

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # use TkAgg backend so the plot window works
                                 # alongside the pygame window
import matplotlib.pyplot as plt

from snake_game import SnakeGameAI
from agent import DQNAgent


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

MAX_EPISODES  = 2_000    # Total episodes to train for.
                         # The agent usually shows clear improvement by ~200
                         # and plays reasonably well by ~500-1000.

RENDER_EVERY  = 50       # Show the game window every N episodes so you can
                         # watch progress, but skip rendering in between for
                         # speed. Set to 1 to always render, 0 to never.

SAVE_PATH     = "best_model"   # Folder to save the model weights when a new
                               # high score is reached.

PRINT_EVERY   = 10       # Print a summary line every N episodes.


# =============================================================================
# LIVE PLOT SETUP
# =============================================================================
# WHY MATPLOTLIB?
# We want to see the agent's score improving in real time without stopping
# training. plt.ion() enables "interactive mode" — the plot updates live
# without blocking the training loop.
#
# We track three things:
#   - scores        : raw score each episode (noisy, but shows peaks)
#   - mean_scores   : rolling average over last 100 episodes (smooth trend)
# =============================================================================

def init_plot():
    """Set up the live training plot. Returns (fig, ax, line1, line2)."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("DQN Snake — Training Progress", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_xlim(0, MAX_EPISODES)
    ax.set_ylim(0, 50)        # will auto-expand as scores improve
    ax.grid(True, alpha=0.3)

    line1, = ax.plot([], [], color='steelblue',  alpha=0.4, label='Score per episode')
    line2, = ax.plot([], [], color='darkorange', linewidth=2, label='Mean (last 100)')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show(block=False)
    return fig, ax, line1, line2


def update_plot(fig, ax, line1, line2, scores, mean_scores):
    """Refresh the plot with the latest scores. Called after each episode."""
    x = list(range(1, len(scores) + 1))

    line1.set_data(x, scores)
    line2.set_data(x, mean_scores)

    # Auto-scale y-axis if scores exceed current limit
    current_max = max(scores) if scores else 1
    if current_max > ax.get_ylim()[1] * 0.9:
        ax.set_ylim(0, current_max * 1.2)

    fig.canvas.draw()
    fig.canvas.flush_events()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():

    # -- 1. INITIALISE ENVIRONMENT AND AGENT ---------------------------------
    #
    # We start with render=False for speed. The game window will be turned on
    # selectively every RENDER_EVERY episodes so you can watch without paying
    # the cost of rendering every frame.
    #
    # WHY CREATE THE ENV ONCE AND CALL reset()?
    # Pygame is expensive to initialise. We create one SnakeGameAI instance
    # and call reset() at the start of each episode — same pattern as Ng's:
    #   env = gym.make(...)   ← once
    #   env.reset()           ← each episode
    env   = SnakeGameAI(render=False)
    agent = DQNAgent(
        state_size  = env.get_state_size(),    # 11
        action_size = env.get_action_size(),   # 3
    )

    # -- 2. TRACKING VARIABLES -----------------------------------------------
    scores        = []        # raw score per episode
    mean_scores   = []        # rolling mean over last 100 episodes
    best_score    = 0         # track the all-time best to know when to save

    # -- 3. LIVE PLOT ---------------------------------------------------------
    fig, ax, line1, line2 = init_plot()

    print("=" * 55)
    print("  DQN Snake Training Started")
    print(f"  Episodes : {MAX_EPISODES}")
    print(f"  Rendering: every {RENDER_EVERY} episodes")
    print("=" * 55)

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    for episode in range(1, MAX_EPISODES + 1):

        # -- Decide whether to render this episode ---------------------------
        # We toggle rendering on/off by updating env.render_mode.
        # No need to recreate the environment — just flip the flag.
        should_render = (RENDER_EVERY > 0 and episode % RENDER_EVERY == 0)
        env.render_mode = should_render

        # -- Reset environment at the start of each episode ------------------
        # Returns the initial 11-feature state vector.
        state = env.reset()

        episode_score = 0
        done          = False

        # =====================================================================
        # INNER LOOP — one full episode (until the snake dies or starves)
        # =====================================================================
        while not done:

            # -- Agent picks an action (ε-greedy) ----------------------------
            # Early episodes: mostly random (ε ≈ 1.0)
            # Later episodes: mostly network-driven (ε ≈ 0.01)
            action = agent.get_action(state)

            # -- Environment executes the action -----------------------------
            # Returns the next state, reward (+10/-10/0), and done flag.
            next_state, reward, done = env.step(action)

            # -- Store the experience in replay memory -----------------------
            # This is the "store experience" step from Ng's algorithm diagram.
            agent.remember(state, action, reward, next_state, done)

            # -- Trigger a learning update (every UPDATE_EVERY steps) --------
            # Internally samples a mini-batch and runs one gradient update.
            agent.learn()

            # -- Advance state -----------------------------------------------
            # CRITICAL: state becomes next_state for the next iteration.
            # This is the line: S_t ← S_{t+1} from the RL loop.
            state = next_state

            # -- Track score -------------------------------------------------
            if reward == 10:      # agent ate food
                episode_score += 1

        # =====================================================================
        # END OF EPISODE
        # =====================================================================

        # -- Decay epsilon after each episode --------------------------------
        # The agent gradually shifts from exploring to exploiting.
        agent.decay_epsilon()

        # -- Record scores ---------------------------------------------------
        scores.append(episode_score)
        mean_score = np.mean(scores[-100:])   # rolling mean of last 100
        mean_scores.append(mean_score)

        # -- Save model if new best score ------------------------------------
        # WHY SAVE ON BEST SCORE, NOT EVERY EPISODE?
        # The agent's performance fluctuates. Saving only on improvements
        # means you always have the best weights on disk, even if the agent
        # later gets worse due to continued training instability.
        if episode_score > best_score:
            best_score = episode_score
            os.makedirs(SAVE_PATH, exist_ok=True)
            agent.q_network.save_weights(
                os.path.join(SAVE_PATH, "snake_dqn.weights.h5")
            )

        # -- Update live plot ------------------------------------------------
        update_plot(fig, ax, line1, line2, scores, mean_scores)

        # -- Print summary every PRINT_EVERY episodes -----------------------
        if episode % PRINT_EVERY == 0:
            print(
                f"Episode {episode:>5} | "
                f"Score: {episode_score:>3} | "
                f"Best: {best_score:>3} | "
                f"Mean(100): {mean_score:>5.2f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    print("\n" + "=" * 55)
    print(f"  Training complete!")
    print(f"  Best score achieved : {best_score}")
    print(f"  Final mean (last 100): {mean_scores[-1]:.2f}")
    print(f"  Model saved to      : {SAVE_PATH}/")
    print("=" * 55)

    # Keep the plot open after training finishes
    plt.ioff()
    plt.show()

    pygame_cleanup()


def pygame_cleanup():
    """Cleanly shut down pygame after training."""
    import pygame
    pygame.quit()


# =============================================================================
# ENTRY POINT
# =============================================================================
# WHY if __name__ == "__main__"?
# This ensures train() only runs when you execute this file directly:
#   python train.py          ← runs
#   import train             ← does NOT run automatically
# =============================================================================

if __name__ == "__main__":
    train()