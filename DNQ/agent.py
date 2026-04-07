import numpy as np
import random
from collections import deque, namedtuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

LR              = 1e-3      # Learning rate (ALPHA in Ng's notebook)
                            # 0.001 is a solid default for Adam on small nets.

GAMMA           = 0.99      # Discount factor — how much future rewards matter.
                            # Ng used 0.995 for LunarLander (long episodes).
                            # Snake episodes are shorter, so 0.99 works well.

MEMORY_SIZE     = 100_000   # Max experiences to store (same as Ng).
                            # Older memories get dropped when full (FIFO).

BATCH_SIZE      = 64        # How many experiences we sample per training step.
                            # 64 is a standard mini-batch size.

MIN_REPLAY_SIZE = 1_000     # Don't start training until we have this many
                            # experiences. Prevents learning from too little data.

EPSILON_START   = 1.0       # Start fully random (100% exploration).
EPSILON_END     = 0.01      # Never go below 1% random (always some exploration).
EPSILON_DECAY   = 0.995     # Multiply epsilon by this after each episode.
                            # At this rate: ~900 episodes to reach ~0.01.

TAU             = 0.005     # Soft update rate (same symbol as Ng uses).
                            # w⁻ ← τ·w + (1-τ)·w⁻
                            # Small τ = target network changes slowly = stable.

UPDATE_EVERY    = 4         # Train every N steps (NUM_STEPS_FOR_UPDATE in Ng).

Experience = namedtuple('Experience',
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)
    
    def add (self, state, action, reward, next_state, done):
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)
    
    def sample(self, batch_size = BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        
        states = tf.convert_to_tensor([e.state       for e in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([e.actio     for e in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([e.reward    for e in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([e.next_state    for e in batch], dtype=tf.float32)
        done_vals = tf.convert_to_tensor([e.done        for e in batch], dtype=tf.float32)
        
        return states, actions, rewards, next_states, done_vals
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.epsilon = EPSILON_START
        
        self.t_step = 0
        self.q_network = self._build_network()
        self.target_network.set_weights(self.q_network.get_weights())
        self.optimizer = Adam(learning_rate=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
    
    
    def _build_network(self):
        model = Sequential([
            Input(shape=(self.state_size,)), # 11 input features
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear'),
        ])
        return model
        