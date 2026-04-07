# =============================================================================
# snake_game.py  —  The Snake Environment for our DQN Agent
# =============================================================================
#
# DESIGN PHILOSOPHY
# -----------------
# Your original snake_game.py was built for a *human* player. A human presses
# arrow keys, watches the screen, and reacts. But our DQN agent is a program —
# it can't press keys, and it doesn't "watch" the screen. It needs to:
#
#   1. Receive the current game STATE as a vector of numbers
#   2. Choose an ACTION (an integer: 0, 1, or 2)
#   3. Get back a REWARD and the NEXT STATE
#
# This is exactly the "agent-environment loop" from Andrew Ng's notebook.
# We're building our own mini Gym environment, just without the gym wrapper.
#
# The key method we're adding is:  step(action) → (next_state, reward, done)
# That single method is what connects the game to the agent.
#
# =============================================================================

import pygame
import numpy as np
import random
from collections import namedtuple

# -----------------------------------------------------------------------------
# CONSTANTS
# We define these at the top so they're easy to tune later.
# SIZE is the pixel size of each grid cell — same as your original code.
# -----------------------------------------------------------------------------
SIZE   = 40          # each grid square is 40x40 pixels
WIDTH  = 600         # game window width  (600 / 40 = 15 columns)
HEIGHT = 600         # game window height (600 / 40 = 15 rows)
COLS   = WIDTH  // SIZE   # → 15
ROWS   = HEIGHT // SIZE   # → 15

# Colors — using plain RGB tuples so we don't depend on image files.
# Your original code loaded .jpg files for the snake and apple. That's fine
# for a human game, but for training we want zero file-loading friction.
BLACK      = (0,   0,   0)
GREEN      = (0,   200, 0)
DARK_GREEN = (0,   150, 0)
RED        = (200, 0,   0)
WHITE      = (255, 255, 255)
BG_COLOR   = (30,  30,  30)   # dark background — easier to see training

# -----------------------------------------------------------------------------
# DIRECTION ENCODING
# We represent direction as a simple Point (named tuple of dx, dy).
# This lets us do direction arithmetic cleanly — e.g., turning left/right
# is just a rotation of the (dx, dy) vector, not a string comparison.
#
# WHY NOT USE STRINGS LIKE YOUR ORIGINAL CODE?
# Your original used if/elif chains on strings ('up', 'down', 'left', 'right').
# That works for human input, but for the agent we need to rotate directions
# mathematically. A Point(dx, dy) makes that trivial.
# -----------------------------------------------------------------------------
Point = namedtuple('Point', ['x', 'y'])

# The four possible directions as (dx, dy) movement per step.
# We store them in clockwise order so that turning is just an index shift.
DIRECTIONS = [
    Point( 0, -SIZE),   # index 0 → UP    (y decreases going up on screen)
    Point( SIZE,  0),   # index 1 → RIGHT
    Point( 0,  SIZE),   # index 2 → DOWN
    Point(-SIZE,  0),   # index 3 → LEFT
]

# The agent's 3 possible actions:
#   0 → go straight (keep current direction)
#   1 → turn right  (clockwise)
#   2 → turn left   (counter-clockwise)
#
# WHY 3 ACTIONS, NOT 4?
# In your original game, the player had 4 arrow keys. But "U-turns" are
# illegal in Snake — if you're going right, you can't instantly go left.
# Instead of filtering illegal moves, we redesign the action space:
# the agent always says "straight / turn right / turn left" *relative*
# to wherever it's currently heading. This makes every action always valid.
STRAIGHT    = 0
TURN_RIGHT  = 1
TURN_LEFT   = 2


# =============================================================================
# CLASS: SnakeGameAI
# =============================================================================
# We renamed the class from `Game` to `SnakeGameAI` to make it clear this
# version is designed for an AI agent, not a human player.
# =============================================================================

class SnakeGameAI:

    def __init__(self, render=True):
        # ------------------------------------------------------------------
        # render=True  → show the pygame window (use during evaluation)
        # render=False → run headless, no window  (use during fast training)
        #
        # WHY HAVE A RENDER FLAG?
        # Training runs thousands of episodes. Drawing every frame to screen
        # is slow. With render=False, we skip all pygame drawing calls and
        # the game runs much faster. We only turn rendering on when we want
        # to *watch* the agent play after training.
        #
        # WHY TRACK _display_initialised SEPARATELY?
        # render_mode can be toggled mid-training (train.py flips it every
        # RENDER_EVERY episodes). But creating a real pygame display window
        # is a one-time OS operation — we can't just flip a flag and have
        # the window appear. We track whether the real display has been
        # created yet, and create it on demand the first time _draw() is
        # called with render_mode=True.
        # ------------------------------------------------------------------
        self.render_mode         = render
        self._display_created    = False   # has pygame.display.set_mode been called?

        pygame.init()
        self.font = pygame.font.SysFont('arial', 22)

        # Always start with an off-screen surface.
        # If render=True from the start, we upgrade it to a real window immediately.
        self.display = pygame.Surface((WIDTH, HEIGHT))

        if self.render_mode:
            self._create_display()

        # Kick off the first episode
        self.reset()

    def _create_display(self):
        """
        Create the real pygame display window.

        WHY A SEPARATE METHOD?
        pygame.display.set_mode() must only be called once per process, and
        only when we actually want a visible window. By isolating it here,
        both __init__ and _draw() can call it safely — it's guarded by the
        _display_created flag so it only runs once.
        """
        if not self._display_created:
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake — DQN Training")
            self._display_created = True

    # -------------------------------------------------------------------------
    # reset()
    # -------------------------------------------------------------------------
    # Called at the start of every new episode (every time the snake dies).
    # It re-initialises all game state and returns the first observation.
    #
    # WHY RETURN THE STATE HERE?
    # In Andrew Ng's notebook: `current_state = env.reset()`
    # We follow the same convention. The agent needs the initial state to
    # pick its very first action.
    # -------------------------------------------------------------------------
    def reset(self):
        # Start the snake in the middle of the board, length 3, heading right
        self.direction_idx = 1   # index into DIRECTIONS → RIGHT
        head = Point(COLS // 2 * SIZE, ROWS // 2 * SIZE)

        # The body is a list of Points, head first.
        # We build a length-3 snake by placing segments to the LEFT of head.
        self.snake = [
            head,
            Point(head.x - SIZE,     head.y),
            Point(head.x - 2 * SIZE, head.y),
        ]

        self.score    = 0
        self.steps    = 0         # steps taken in this episode
        self.max_steps = 100 * len(self.snake)  # starvation limit (see below)

        self._place_food()
        return self._get_state()  # return initial state vector

    # -------------------------------------------------------------------------
    # _place_food()  (private helper)
    # -------------------------------------------------------------------------
    # Places food at a random grid position that is NOT occupied by the snake.
    # Underscore prefix = private method, not meant to be called externally.
    # -------------------------------------------------------------------------
    def _place_food(self):
        while True:
            x = random.randint(0, COLS - 1) * SIZE
            y = random.randint(0, ROWS - 1) * SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break   # valid position found

    # -------------------------------------------------------------------------
    # step(action)  ← THE MOST IMPORTANT METHOD
    # -------------------------------------------------------------------------
    # This is the heart of the environment. The agent calls this once per
    # time step. It:
    #   1. Applies the action (moves the snake)
    #   2. Checks for collisions / food
    #   3. Computes the reward
    #   4. Returns (next_state, reward, done)
    #
    # This mirrors env.step(action) from Andrew Ng's Gym environment exactly.
    # -------------------------------------------------------------------------
    def step(self, action):
        self.steps += 1

        # -- 1. HANDLE QUIT EVENTS (so the window doesn't freeze) ------------
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # -- 2. MOVE THE SNAKE -----------------------------------------------
        self._move(action)   # updates self.snake[0] (the head)

        # -- 3. CHECK FOR DEATH ----------------------------------------------
        done   = False
        reward = 0

        if self._is_collision() or self.steps > self.max_steps:
            # REWARD DESIGN — collision is heavily penalised.
            # -10 is large enough to outweigh the small per-step costs and
            # signal clearly that dying is the worst outcome.
            #
            # WHY max_steps?
            # Without a step limit, a snake that learns to loop endlessly
            # (avoiding death but never eating) would train forever.
            # We kill episodes that go on too long — "starvation" penalty.
            done   = True
            reward = -10
            return self._get_state(), reward, done

        # -- 4. CHECK FOR FOOD -----------------------------------------------
        if self.snake[0] == self.food:
            # REWARD DESIGN — eating food is strongly positive.
            # +10 matches the collision penalty in magnitude.
            self.score += 1
            reward      = 10
            self._place_food()
            # DO NOT remove the tail → snake grows by 1
            self.max_steps = 100 * len(self.snake)  # update starvation limit
        else:
            # Normal move: remove the tail segment (snake stays same length)
            self.snake.pop()

        # -- 5. RENDER -------------------------------------------------------
        if self.render_mode:
            self._draw()

        # -- 6. RETURN (next_state, reward, done) ----------------------------
        # This is identical to the return signature of env.step() in gym.
        return self._get_state(), reward, done

    # -------------------------------------------------------------------------
    # _move(action)  (private helper)
    # -------------------------------------------------------------------------
    # Translates the agent's action (0/1/2) into a new head position.
    #
    # HOW TURNING WORKS:
    # DIRECTIONS is ordered clockwise: UP(0) → RIGHT(1) → DOWN(2) → LEFT(3)
    # Turning right = index + 1 (mod 4)
    # Turning left  = index - 1 (mod 4)
    # Going straight = index unchanged
    # -------------------------------------------------------------------------
    def _move(self, action):
        if action == TURN_RIGHT:
            self.direction_idx = (self.direction_idx + 1) % 4
        elif action == TURN_LEFT:
            self.direction_idx = (self.direction_idx - 1) % 4
        # else STRAIGHT → direction_idx unchanged

        direction = DIRECTIONS[self.direction_idx]
        new_head  = Point(
            self.snake[0].x + direction.x,
            self.snake[0].y + direction.y,
        )
        # Insert new head at the front of the snake list
        self.snake.insert(0, new_head)

    # -------------------------------------------------------------------------
    # _is_collision(point=None)  (private helper)
    # -------------------------------------------------------------------------
    # Returns True if a given point hits a wall or the snake's own body.
    # If no point is given, checks the current head position.
    #
    # WHY ACCEPT AN OPTIONAL POINT?
    # The state function (_get_state) needs to check hypothetical positions
    # — "if I go straight, would I die?" — without actually moving. So we
    # make collision-checking reusable for any point.
    # -------------------------------------------------------------------------
    def _is_collision(self, point=None):
        if point is None:
            point = self.snake[0]

        # Wall collision
        if point.x < 0 or point.x >= WIDTH:
            return True
        if point.y < 0 or point.y >= HEIGHT:
            return True

        # Self collision (skip head — it was just placed there)
        if point in self.snake[1:]:
            return True

        return False

    # -------------------------------------------------------------------------
    # _get_state()  ← THE SECOND MOST IMPORTANT METHOD
    # -------------------------------------------------------------------------
    # Converts the current game situation into an 11-element numpy array
    # that the neural network can process.
    #
    # WHY 11 FEATURES?
    # The neural network can't "see" the grid. It only gets numbers.
    # We hand-craft features that capture the most useful information:
    #
    #   [0..2]  — DANGER: is there a collision directly ahead / right / left?
    #   [3..6]  — DIRECTION: which way is the snake currently moving?
    #   [7..10] — FOOD: is the food up / down / left / right of the head?
    #
    # All values are 0 or 1 (binary). This keeps the input space small and
    # makes the network's job easier — it doesn't have to learn coordinate
    # systems, just patterns in these 11 boolean signals.
    #
    # COMPARISON TO ANDREW NG'S LUNAR LANDER:
    # LunarLander used 8 continuous state variables (positions, velocities).
    # Our Snake uses 11 binary variables. Binary inputs are simpler to learn
    # from and train faster for a game of this complexity.
    # -------------------------------------------------------------------------
    def _get_state(self):
        head = self.snake[0]
        d    = self.direction_idx

        # Compute the three points the snake could move to next
        # (straight, right-of-current-direction, left-of-current-direction)
        dir_straight = DIRECTIONS[d]
        dir_right    = DIRECTIONS[(d + 1) % 4]
        dir_left     = DIRECTIONS[(d - 1) % 4]

        pt_straight = Point(head.x + dir_straight.x, head.y + dir_straight.y)
        pt_right    = Point(head.x + dir_right.x,    head.y + dir_right.y)
        pt_left     = Point(head.x + dir_left.x,     head.y + dir_left.y)

        state = [
            # --- DANGER (3 features) ---
            # Is there a wall or body collision one step ahead?
            int(self._is_collision(pt_straight)),   # danger straight
            int(self._is_collision(pt_right)),      # danger right
            int(self._is_collision(pt_left)),       # danger left

            # --- CURRENT DIRECTION (4 features, one-hot) ---
            # Only one of these will be 1 at any time.
            int(d == 3),   # moving left?
            int(d == 1),   # moving right?
            int(d == 0),   # moving up?
            int(d == 2),   # moving down?

            # --- FOOD LOCATION relative to head (4 features) ---
            int(self.food.x < head.x),   # food is to the left
            int(self.food.x > head.x),   # food is to the right
            int(self.food.y < head.y),   # food is above
            int(self.food.y > head.y),   # food is below
        ]

        return np.array(state, dtype=np.float32)

    # -------------------------------------------------------------------------
    # _draw()  (private helper)
    # -------------------------------------------------------------------------
    # Draws the current game state to the pygame window.
    # Only called when render_mode=True. Pure visuals — no game logic here.
    #
    # KEY FIX: we call _create_display() first because render_mode can be
    # toggled by train.py mid-training. If we started headless (render=False),
    # no real display window exists yet. _create_display() creates it on demand
    # the first time _draw() is called with render_mode=True.
    # Without this, pygame.display.flip() crashes: "Display mode not set"
    # -------------------------------------------------------------------------
    def _draw(self):
        self._create_display()   # ← safe to call multiple times (guarded by flag)

        self.display.fill(BG_COLOR)

        # Draw each snake segment (two-tone to make the body visible)
        for i, pt in enumerate(self.snake):
            color = GREEN if i == 0 else DARK_GREEN   # head is brighter
            pygame.draw.rect(self.display, color,
                             pygame.Rect(pt.x + 2, pt.y + 2, SIZE - 4, SIZE - 4))

        # Draw food
        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.x + 4, self.food.y + 4,
                                     SIZE - 8, SIZE - 8))

        # Draw score
        score_text = self.font.render(f"Score: {self.score}  Steps: {self.steps}",
                                      True, WHITE)
        self.display.blit(score_text, (8, 8))

        pygame.display.flip()

    # -------------------------------------------------------------------------
    # get_state_size() / get_action_size()  — convenience methods
    # -------------------------------------------------------------------------
    # The agent needs to know the input/output dimensions to build its network.
    # These mirror Ng's: state_size = env.observation_space.shape
    #                                 num_actions = env.action_space.n
    # -------------------------------------------------------------------------
    def get_state_size(self):
        return 11   # our 11-feature vector

    def get_action_size(self):
        return 3    # straight, right, left