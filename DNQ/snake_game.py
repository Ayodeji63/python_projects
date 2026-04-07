import pygame
import numpy as np
import random
from collections import namedtuple


SIZE = 40
WIDTH = 600

HEIGHT = 600
COLS = WIDTH // SIZE # 600 // 40 = 15
ROWS = HEIGHT // SIZE # 600 // 40 = 15

BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
RED = (200, 0, 0)
WHITE = (255, 255, 255)
BG_COLOR = (30, 30, 30)

Point = namedtuple('Point', ['x', 'y'])

DIRECTIONS = [
    Point(0, -SIZE), # index 0: up
    Point(SIZE, 0), # index 1: right
    Point(0, SIZE), # index 2: down
    Point(-SIZE, 0)  # index 3: left
]

STRAIGHT = 0
TURN_RIGHT = 1
TURN_LEFT = 2

class SnakeGameAI:
    def __init__(self, render=True):
        
        self.render_mode = render
        self._display_created = False
        
        pygame.init()
        self.font = pygame.font.SysFont('arial', 22)
        
        self.display = pygame.Surface((WIDTH, HEIGHT))
        
        if self.render_mode:
            self._create_display()
        
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
    
    def reset(self):
        self.direction_idx = 1
        # head = 15 // 2 = 7 * 40 = 280
        head = Point(COLS // 2 * SIZE, ROWS // 2 * SIZE)
        self.snake = [
            head,
            Point(head.x - SIZE, head.y),
            Point(head.x - 2 * SIZE, head.y)
        ]
        
        self.score = 0
        self.steps = 0
        self.max_steps = 100 * len(self.snake)
        
        self._place_food()
        return self._get_state()
    
    def _place_food(self):
        while True:
            x = random.randint(0, COLS - 1) * SIZE
            y = random.randint(0, ROWS - 1) * SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
    
    def step(self, action):
        self.steps += 1
        
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                    
        self._move(action)
        
        done = False
        reward = 0
        
        if self._is_collision() or self.steps > self.max_steps:
            done = True
            reward = -10
            return self._get_state(), reward, done
        
        if self.snake[0] == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.max_steps = 100 * len(self.snake)
        else:
            self.snake.pop()
        
        if self.render_mode:
            self._draw()
        
        return self._get_state(), reward, done
    
    
    
    
    def _move(self, action):
        if action == TURN_RIGHT:
            self.direction_idx = (self.direction_idx + 1) % 4
        elif action == TURN_LEFT:
            self.direction_idx = (self.direction_idx - 1) % 4
        
        direction = DIRECTIONS[self.direction_idx]
        new_head = Point(
            self.snake[0].x + direction.x,
            self.snake[0].y + direction.y, 
        )
        self.snake.insert(0, new_head)
        
    def _is_collision(self, point=None):
        if point is None:
            point = self.snake[0]
        
        # Wall collision
        if point.x < 0 or point.x >= WIDTH:
            return True
        if point.y < 0 or point.y >= HEIGHT:
            return True
        
        if point in self.snake[1:]:
            return True
        
        return False
    
    def _get_state(self):
        head = self.snake[0]
        d = self.direction_idx
        
        dir_straight = DIRECTIONS[d]
        dir_right = DIRECTIONS[(d + 1) % 4]
        dir_left = DIRECTIONS[(d - 1) % 4]
        
        pt_straight = Point(head.x + dir_straight.x , head.y * dir_straight.y)
        pt_right = Point(head.x + dir_right.x, head.y + dir_right.y)
        pt_left = Point(head.x + dir_left.x, head.y + dir_left.y)
        
        state = [
            int(self._is_collision(pt_straight)), # danger straight
            int(self._is_collision(pt_right)), # danger right
            int(self._is_collision(pt_left)), # danger left
            
            int(d == 3),
            int(d == 1),
            int(d == 0),
            int(d == 2),
            
            int(self.food.x < head.x), # food left
            int(self.food.x > head.x), # food right
            int(self.food.y < head.y), # food up
            int(self.food.y > head.y) # food down
        ]
        
        return np.array(state, dtype=np.float32)
    
    
    def _draw(self):
        self._create_display()
        self.display.fill(BG_COLOR)
        
        for i, pt in enumerate(self.snake):
            color = GREEN if i == 0 else DARK_GREEN
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x + 4, pt.y + 4, SIZE - 8, SIZE - 8))
        
        # Draw Food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x + 4, self.food.y + 4, SIZE - 8, SIZE - 8))
        
        # Draw Score
        score_text = self.font.render(f"Score: {self.score} Steps: {self.steps}", True, WHITE)
        self.display.blit(score_text, (8, 8))
        
        pygame.display.flip()
    
    def get_state_size(self):
        return 11
    
    def get_action_size(self):
        return 3
            
            