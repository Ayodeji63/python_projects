# =============================================================================
# agent.py  —  The DQN Agent (Andrew Ng's approach, adapted for Snake)
# =============================================================================
#
# DESIGN PHILOSOPHY
# -----------------
# This file is a direct translation of Andrew Ng's Lunar Lander DQN into
# our Snake environment. Every major design decision maps 1-to-1:
#
#   Ng's notebook          →   This file
#   ─────────────────────────────────────────────────────
#   q_network              →   self.q_network
#   target_q_network       →   self.target_network
#   experience (namedtuple)→   Experience (namedtuple)
#   memory buffer (deque)  →   self.memory
#   compute_loss()         →   _compute_loss()
#   soft update (τ)        →   _soft_update_target()
#   ALPHA, GAMMA           →   LR, GAMMA
#
# The only new piece is `get_action()` which implements ε-greedy exploration
# — choosing random actions early on, then trusting the network more over time.
#
# =============================================================================

import numpy as np
import random
from collections import deque, namedtuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
# These are the knobs you can tune. We follow Ng's values where possible
# and explain any deviations.

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


# =============================================================================
# EXPERIENCE NAMEDTUPLE
# =============================================================================
# Identical to Ng's notebook:
#   experience = namedtuple("Experience", field_names=["state","action",
#                            "reward","next_state","done"])
#
# WHY A NAMEDTUPLE?
# We store thousands of (state, action, reward, next_state, done) tuples.
# A namedtuple gives us clean attribute access (exp.reward) instead of
# fragile index access (exp[2]). It's also memory-efficient vs. a dict.
# =============================================================================
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "next_state", "done"]
)


# =============================================================================
# CLASS: ReplayMemory
# =============================================================================
# This is the Experience Replay buffer from Section 6.2 of Ng's notebook.
#
# WHY EXPERIENCE REPLAY?
# When the snake is moving, consecutive states are highly correlated:
#   (going right, step 1) → (going right, step 2) → (going right, step 3)
# If we trained on these in order, the network would overfit to "going right"
# and forget everything else. By storing experiences and sampling randomly,
# we break these correlations and get a healthier training signal.
#
# The deque automatically drops old experiences when it hits maxlen — this
# is the FIFO (first-in, first-out) behaviour we want.
# =============================================================================

class ReplayMemory:

    def __init__(self, max_size=MEMORY_SIZE):
        # deque with maxlen = fixed-size circular buffer
        # When full, adding a new item automatically removes the oldest one.
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """Store one experience tuple in the buffer."""
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

    def sample(self, batch_size=BATCH_SIZE):
        """
        Sample a random mini-batch of experiences.

        Returns 5 TensorFlow tensors:
            states, actions, rewards, next_states, done_vals

        WHY CONVERT TO TENSORS HERE?
        Our loss function (below) uses TensorFlow ops. By converting once
        during sampling, we avoid repeated conversion inside the training loop.
        This mirrors how Ng's notebook feeds data into his Keras model.
        """
        # random.sample picks without replacement — no duplicate experiences
        batch = random.sample(self.buffer, batch_size)

        # 'batch' is a list of Experience namedtuples.
        # We transpose it: instead of [(s1,a1,...), (s2,a2,...)]
        # we want separate arrays: [s1,s2,...], [a1,a2,...], etc.
        states      = tf.convert_to_tensor([e.state      for e in batch], dtype=tf.float32)
        actions     = tf.convert_to_tensor([e.action     for e in batch], dtype=tf.int32)
        rewards     = tf.convert_to_tensor([e.reward     for e in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([e.next_state for e in batch], dtype=tf.float32)
        done_vals   = tf.convert_to_tensor([e.done       for e in batch], dtype=tf.float32)

        return states, actions, rewards, next_states, done_vals

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# CLASS: DQNAgent
# =============================================================================
# The agent that interacts with the Snake environment and learns from it.
# =============================================================================

class DQNAgent:

    def __init__(self, state_size, action_size):
        """
        state_size  : int — number of input features (11 for Snake)
        action_size : int — number of possible actions (3 for Snake)
        """
        self.state_size  = state_size
        self.action_size = action_size

        # ε (epsilon) starts high (explore randomly) and decays over time.
        # This is the ε-greedy exploration strategy.
        self.epsilon = EPSILON_START

        # Step counter — used to decide when to run a training update.
        self.t_step = 0

        # ── BUILD THE TWO NETWORKS ──────────────────────────────────────────
        # This is Exercise 1 from Ng's notebook, adapted for our state/actions.
        #
        # ARCHITECTURE CHOICE:
        # Ng used [64, 64] hidden units for LunarLander (8 inputs, 4 outputs).
        # Our Snake has 11 inputs and 3 outputs — similar complexity.
        # We use [256, 256] because Snake's strategy space (when to turn,
        # when to chase food, how to avoid itself) is richer than it looks.
        # You can experiment with [128, 128] or [64, 64] as well.
        #
        # The final layer uses 'linear' activation — same as Ng — because
        # Q-values are unbounded real numbers (not probabilities).
        self.q_network      = self._build_network()
        self.target_network = self._build_network()

        # Copy q_network weights to target_network so they start identical.
        # If they start differently, the first soft-update is meaningless.
        self.target_network.set_weights(self.q_network.get_weights())

        # Adam optimiser — same as Ng's notebook (ALPHA = LR = 1e-3)
        self.optimizer = Adam(learning_rate=LR)

        # Replay memory buffer
        self.memory = ReplayMemory(MEMORY_SIZE)

    # -------------------------------------------------------------------------
    # _build_network()  (private)
    # -------------------------------------------------------------------------
    # Builds and returns a Keras Sequential model.
    # Identical pattern to Ng's Exercise 1, just different layer sizes.
    # -------------------------------------------------------------------------
    def _build_network(self):
        model = Sequential([
            Input(shape=(self.state_size,)),          # 11 input features
            Dense(256, activation='relu'),             # hidden layer 1
            Dense(256, activation='relu'),             # hidden layer 2
            Dense(self.action_size, activation='linear'),  # output: Q-value per action
        ])
        return model

    # -------------------------------------------------------------------------
    # get_action(state)  — ε-greedy policy
    # -------------------------------------------------------------------------
    # HOW ε-GREEDY WORKS:
    # With probability ε → choose a RANDOM action (exploration)
    # With probability 1-ε → choose the action with the highest Q-value (exploitation)
    #
    # Early in training, ε ≈ 1.0, so the agent explores almost randomly.
    # Over time, ε decays toward EPSILON_END, so the agent trusts its network more.
    #
    # WHY DO WE NEED EXPLORATION AT ALL?
    # If the agent always picks its best known action, it might never discover
    # that a different sequence of actions leads to a much higher reward.
    # Random exploration ensures it tries enough of the state space.
    # -------------------------------------------------------------------------
    def get_action(self, state):
        if random.random() < self.epsilon:
            # EXPLORE: random action
            return random.randint(0, self.action_size - 1)
        else:
            # EXPLOIT: ask the Q-network which action looks best
            # state shape is (11,) — we need (1, 11) for the network
            state_tensor = tf.expand_dims(
                tf.convert_to_tensor(state, dtype=tf.float32), axis=0
            )
            q_values = self.q_network(state_tensor, training=False)  # shape: (1, 3)
            return int(tf.argmax(q_values[0]).numpy())   # index of max Q-value

    # -------------------------------------------------------------------------
    # remember(...)
    # -------------------------------------------------------------------------
    # Stores one experience in replay memory. Simple wrapper so train.py
    # doesn't need to know about the ReplayMemory internals.
    # -------------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    # -------------------------------------------------------------------------
    # learn()
    # -------------------------------------------------------------------------
    # Called every step. Increments the step counter and triggers a training
    # update every UPDATE_EVERY steps (if we have enough memories).
    #
    # WHY NOT TRAIN EVERY SINGLE STEP?
    # Sampling and backpropagating on every step is expensive and also
    # causes the network to overfit to very recent experiences.
    # Ng used NUM_STEPS_FOR_UPDATE = 4 — we do the same.
    # -------------------------------------------------------------------------
    def learn(self):
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # Only train if we have enough experiences to form a mini-batch
            if len(self.memory) >= MIN_REPLAY_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self._train_step(experiences)

    # -------------------------------------------------------------------------
    # _train_step(experiences)  (private)
    # -------------------------------------------------------------------------
    # The actual gradient update. This is the core of DQN learning.
    # Wrapping it in tf.GradientTape() is the TensorFlow way to compute
    # gradients manually — Ng's notebook uses the same pattern.
    # -------------------------------------------------------------------------
    def _train_step(self, experiences):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(experiences)

        # Compute gradients of the loss with respect to q_network weights
        gradients = tape.gradient(loss, self.q_network.trainable_variables)

        # Apply the gradients (this is the actual "learning" step)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

        # After each training step, nudge the target network toward q_network
        self._soft_update_target()

    # -------------------------------------------------------------------------
    # _compute_loss(experiences)  (private)
    # -------------------------------------------------------------------------
    # THIS IS EXERCISE 2 FROM NG'S NOTEBOOK — implemented here for Snake.
    #
    # The Bellman target (y) is:
    #   y = R                                      if episode is done
    #   y = R + γ · max_a'[ Q̂(s', a') ]           otherwise
    #
    # Written compactly using the (1 - done) trick from Ng's notebook:
    #   y = R + γ · max_a'[ Q̂(s', a') ] · (1 - done)
    #
    # WHY (1 - done)?
    # When done=True (the snake died), the episode ended — there is no future
    # reward to add. (1 - 1) = 0 zeros out the future term automatically.
    # When done=False, (1 - 0) = 1, so the future term is included normally.
    # This lets us compute all 64 targets in one vectorised operation.
    #
    # LOSS FUNCTION:
    # We use MSE(y_targets, Q(s,a)) — same as Ng.
    # The Q-network predicts Q-values for ALL 3 actions, but we only care
    # about the Q-value for the action that was actually taken. We use
    # tf.one_hot + masking to extract just those values.
    # -------------------------------------------------------------------------
    def _compute_loss(self, experiences):
        states, actions, rewards, next_states, done_vals = experiences

        # -- Step 1: Compute the target Q-values using the TARGET network --
        # target_network(next_states) → shape (64, 3): Q-values for all actions
        # tf.reduce_max(..., axis=-1) → shape (64,): best Q-value per experience
        max_qsa = tf.reduce_max(
            self.target_network(next_states, training=False), axis=-1
        )

        # -- Step 2: Compute the Bellman targets y (Ng's Exercise 2) --
        # rewards shape: (64,)
        # max_qsa shape: (64,)
        # done_vals shape: (64,)  ← values are 0.0 or 1.0
        y_targets = rewards + GAMMA * max_qsa * (1 - done_vals)

        # -- Step 3: Get the Q(s,a) values for the ACTIONS ACTUALLY TAKEN --
        # q_network(states) → shape (64, 3)
        # We want to extract one Q-value per row — the one for the action taken.
        #
        # tf.one_hot converts action indices to binary masks:
        #   action=0 → [1, 0, 0]
        #   action=1 → [0, 1, 0]
        #   action=2 → [0, 0, 1]
        #
        # Multiplying by q_values zeros out the other actions.
        # tf.reduce_sum collapses each row to the single non-zero value.
        q_values = self.q_network(states, training=True)           # (64, 3)
        action_mask = tf.one_hot(actions, self.action_size)         # (64, 3)
        q_sa = tf.reduce_sum(q_values * action_mask, axis=-1)       # (64,)

        # -- Step 4: MSE between targets and predictions --
        # This is the error the network is trying to minimise.
        loss = MSE(y_targets, q_sa)
        return loss

    # -------------------------------------------------------------------------
    # _soft_update_target()  (private)
    # -------------------------------------------------------------------------
    # Implements the soft update from Ng's Section 6.1:
    #   w⁻ ← τ · w + (1 - τ) · w⁻
    #
    # WHY SOFT UPDATE INSTEAD OF HARD COPY?
    # If we copied q_network → target_network every N steps (hard update),
    # the target values would jump suddenly, causing instability.
    # Soft update moves the target weights very slowly (τ = 0.005),
    # making the training targets change gradually — much more stable.
    # -------------------------------------------------------------------------
    def _soft_update_target(self):
        for q_weight, target_weight in zip(
            self.q_network.trainable_variables,
            self.target_network.trainable_variables
        ):
            # τ · w + (1 - τ) · w⁻
            target_weight.assign(TAU * q_weight + (1 - TAU) * target_weight)

    # -------------------------------------------------------------------------
    # decay_epsilon()
    # -------------------------------------------------------------------------
    # Called once per episode (not per step).
    # Multiplies ε by EPSILON_DECAY, but never lets it fall below EPSILON_END.
    #
    # WHY DECAY PER EPISODE, NOT PER STEP?
    # Per-step decay reduces ε very quickly at the start (when episodes are
    # short and the agent dies fast). Per-episode decay is more predictable —
    # you know roughly how many episodes of exploration you'll get.
    # -------------------------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)