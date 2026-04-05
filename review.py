import pygame
import time

class Snake:
    def __init__(self, parent_screen):
        self.parent_screen = parent_screen
        self.block = pygame.image.load("resources/block.jpg").convert()
        self.x = 100
        self.y = 100
        self.direction = 'down'
        
    def draw(self):
        self.parent_screen.fill((100, 100, 5))
        self.parent_screen.blit(self.block,(self.x, self.y))
        pygame.display.flip()
    
    def move_left(self):
        self.direction = 'left'
    
    def move_right(self):
        self.direction = 'right'
        
    def move_up(self):
        self.direction = 'up'
        
    def move_down(self):
        self.direction = 'down'
    
    def walking(self):
        if self.direction == 'up':
            self.y -= 10
        if self.direction == 'down':
            self.y += 10
        if self.direction == 'left':
            self.x -= 10
        if self.direction == 'right':
            self.x += 10
        self.draw()
        

class Game:
    def __init__(self):
        pygame.init()
        self.surface = pygame.display.set_mode((500, 500))
        self.surface.fill((110, 110, 5))
        self.snake = Snake(self.surface)
        self.snake.draw()
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_UP:
                       self.snake.move_up()
                    if event.key == pygame.K_DOWN:
                        self.snake.move_down()
                    if event.key == pygame.K_LEFT:
                        self.snake.move_left()
                    if event.key == pygame.K_RIGHT:
                        self.snake.move_right()
                    self.surface.fill((110, 110, 5))
                    self.surface.blit(self.snake.block, (self.snake.x, self.snake.y))
                    pygame.display.flip()
                elif event.type == pygame.QUIT:
                    running = False
            self.snake.walking()
            time.sleep(0.2)
        

if __name__ == "__main__":
    game = Game()
    game.run()
    
