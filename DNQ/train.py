import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # silence TensorFlow info/warning logs

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # use TkAgg backend so the plot window works
                                 # alongside the pygame window
import matplotlib.pyplot as plt

from snake_game import SnakeGameAI
from agent import DQNAgent

MAX_EPISODES = 2000

RENDER_EVERY = 50 

SAVE_PATH = "best_model"

PRINT_EVERY = 10

def init_plot():
    plt.ion()
    fig,ax = plt.subplots(figsize=(10,5))
    ax.set_title("DQN Snake Training Progress")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_xlim(0, MAX_EPISODES)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    
    line1, = ax.plot([], [], color='steelblue', alpha=0.4, label='Score per episode')
    line2, = ax.plot([], [], color='orange', alpha=0.8, label='Running average (100 episodes)')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax, line1, line2

def update_plot(fig, ax, line1, line2, scores, mean_scores):
    x = list(range(1, len(scores) + 1))
    line1.set_data(x, scores)
    line2.set_data(x, mean_scores)
    
    current_max = max(scores) if scores else 1
    if current_max > ax.get_ylim()[1] * 0.9:
        ax.set_ylim(0, current_max * 1.2)
    
    fig.canvas.draw()
    fig.canvas.flush_events()


def train():
    env = SnakeGameAI(render=False)
    agent = DQNAgent(state_size = env.get_state_size(), action_size=env.get_action_size() )
    
    scores = []
    mean_scores = []
    best_score = 0
    
    fig, ax, line1, line2 = init_plot()
    
    print("=" * 55)
    print(" DQN Snake Training Started")
    print(f" Episodes : {MAX_EPISODES}")
    print(f" Rendering: every {RENDER_EVERY} episodes")
    print("=" * 55)
    
    for episode in range(1, MAX_EPISODES + 1):
        should_render = (RENDER_EVERY > 0 and episode
                          % RENDER_EVERY == 0)
        env.render_mode = should_render
        
        state = env.reset()
        
        episode_score = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if reward == 10:
                episode_score += 1
        
        agent.decay_epsilon()
        
        scores.append(episode_score)
        mean_score = np.mean(scores[-100:])
        mean_scores.append(mean_score)
        
        if episode_score > best_score:
            best_score = episode_score
            os.makedirs(SAVE_PATH, exist_ok=True)
            agent.q_network.save_weights(os.path.join(SAVE_PATH, "snake_dpn.weights.h5"))
        
        update_plot(fig, ax, line1, line2, scores, mean_scores)
        
        if episode % PRINT_EVERY == 0:
            print(
                f"Episode {episode:4d} | "
                f"Score: {episode_score:2d} | "
                f"Mean Score (100 eps): {mean_score:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
                f"Best Score: {best_score:2d}"
            )
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

if __name__ == "__main__":
    train()