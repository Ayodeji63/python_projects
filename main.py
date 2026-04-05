import pygame
from pygame.locals import *
import time
import random   

SIZE = 40
BACKGROUND_COLOR = 110, 110, 5

class Apple:
    def __init__(self, parent_screen):
        self.parent_screen = parent_screen
        self.image = pygame.image.load("resources/apple.jpg").convert()
        self.x = SIZE*3
        self.y = SIZE*3
    
    def draw(self):
        self.parent_screen.blit(self.image, (self.x, self.y))
        pygame.display.flip()
    
    def move(self):
        self.x = SIZE * random.randint(0, 14)
        self.y = SIZE * random.randint(0, 14)
        

class Snake:
    def __init__(self, parent_screen, length):
        self.parent_screen = parent_screen
        self.block = pygame.image.load("resources/block.jpg").convert()
        self.length = length
        self.x = [40] * length
        self.y = [40] * length
        self.direction = 'down' 
        self.is_collided = False
    
    def increase_length(self):
        self.length += 1
        self.x.append(-1)
        self.y.append(-1)
        
    def move_left(self):
        self.direction = 'left'
    
    def move_right(self):
        self.direction = 'right'
    
    def move_up(self):
        self.direction = 'up'
    
    def move_down(self):
        self.direction = 'down'
    
    def walk(self):
        
        for i in range(self.length-1, 0, -1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]
        
        if self.direction == 'up':
            self.y[0] -= SIZE
        if self.direction == 'down':
            self.y[0] += SIZE
        if self.direction == 'left':
            self.x[0] -= SIZE
        if self.direction == 'right':
            self.x[0] += SIZE
        self.draw()
    
    def draw(self):
        self.parent_screen.fill((BACKGROUND_COLOR))
        for i in range(self.length):
            self.parent_screen.blit(self.block, (self.x[i], self.y[i]))
        pygame.display.flip()   

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake Game")
        
        pygame.mixer.init()
        self.surface = pygame.display.set_mode((700, 700))
        self.surface.fill((BACKGROUND_COLOR))
        self.snake = Snake(self.surface, 1)
        self.snake.draw()
        self.apple = Apple(self.surface)
        self.apple.draw()
    
    def is_collision(self, x1, y1, x2, y2):
        if x1 >= x2 and x1 < x2 + SIZE:
            if y1 >= y2 and y1 < y2 + SIZE:
                return True
        return False
    
    def play_background_music(self):
        pygame.mixer.music.load("resources/bg_music_1.mp3")
        pygame.mixer.music.play()
    def play_sound(self, sound_name):
        sound = pygame.mixer.Sound(f"resources/{sound_name}.mp3")
        pygame.mixer.Sound.play(sound)
    
    def display_score(self):
        font = pygame.font.SysFont('arial', 30)
        score = font.render(f"Score: {self.snake.length - 1}", True, (255, 255, 255))
        self.surface.blit(score, (550, 10))
    
    def play(self):
        self.snake.walk()
        self.apple.draw()
        self.display_score()
        pygame.display.flip()
        
        if self.is_collision(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.play_sound("ding")
            self.snake.increase_length()
            self.apple.move()
            
        
        # snake colliding with itself
        for i in range(3, self.snake.length):
            if self.is_collision(self.snake.x[0], self.snake.y[0], self.snake.x[i], self.snake.y[i]):
                self.play_sound("crash")
                raise "Collision Occured"
    
    def show_game_over(self):
        self.surface.fill(BACKGROUND_COLOR)
        font = pygame.font.SysFont('arial', 30)
        line1 = font.render(f"Game is over! Your score is {self.snake.length}", True, (255, 255, 255))
        self.surface.blit(line1, (100, 150))
        line2 = font.render("To play again press Enter.", True, (255, 255, 255))
        self.surface.blit(line2, (100, 180))
        line3 = font.render("To exit press Escape!", True, (255, 255, 255))
        self.surface.blit(line3, (100, 200))
        pygame.display.flip()
    
    def reset(self):
        self.snake = Snake(self.surface, 1)
        self.apple = Apple(self.surface)
    
    def run(self):
            running = True
            pause = False
            while running:
                for event in pygame.event.get():
                    if event.type == KEYDOWN: # type: ignore
                        if event.key == K_ESCAPE: # type: ignore
                            running = False
                        if event.key == K_RETURN: # type: ignore
                            pause = False
                            
                        if not pause:   
                            if event.key == K_UP:
                                self.snake.move_up()
                            if event.key == K_DOWN:
                                self.snake.move_down()
                            if event.key == K_LEFT:
                                self.snake.move_left()
                            if event.key == K_RIGHT:
                                self.snake.move_right()
                    elif event.type == QUIT: # type: ignore
                        running = False
                try:
                    if not pause:
                        self.play()
                        time.sleep(0.2)
                except Exception as e:
                    self.show_game_over()
                    self.reset()
                    pause = True
                
                
            
if __name__ == "__main__":

    game = Game()
    game.run()
    
    

    
    