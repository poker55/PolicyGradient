import pygame
import numpy as np
import math
import sys

# Import constants from constants.py
from constants import *

class BreakoutEnv:
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Breakout Policy Gradients")
            self.clock = pygame.time.Clock()
        
        # Handle Pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        self.reset()
    
    def reset(self):
        # Reset paddle to center
        self.paddle_x = SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2
        
        # Reset ball position and velocity
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT - 100
        self.ball_dx = BALL_SPEED * math.sin(ANGLE)
        self.ball_dy = -BALL_SPEED * math.cos(ANGLE)
        
        # Reset game stats
        self.score = 0
        self.steps = 0
        self.game_started = True
        
        # Recreate all bricks
        self.bricks = []
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                x = col * (BRICK_WIDTH + BRICK_PADDING) + BRICK_PADDING
                y = row * (BRICK_HEIGHT + BRICK_PADDING) + BRICK_PADDING
                self.bricks.append({
                    'rect': pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT),
                    'color': BRICK_COLORS[row],
                    'active': True
                })
        
        return self._get_state()
    
    def _get_state(self):
        state = [
            self.paddle_x / SCREEN_WIDTH,           # Normalized paddle x position
            self.ball_x / SCREEN_WIDTH,             # Normalized ball x position
            self.ball_y / SCREEN_HEIGHT,            # Normalized ball y position
            self.ball_dx / BALL_SPEED,              # Normalized ball x velocity
            self.ball_dy / BALL_SPEED,              # Normalized ball y velocity
        ]
        
        # Add normalized distances to the closest active bricks
        closest_bricks = float('inf'), float('inf')
        for brick in self.bricks:
            if brick['active']:
                dx = brick['rect'].centerx - self.ball_x
                dy = brick['rect'].centery - self.ball_y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < closest_bricks[0]:
                    closest_bricks = (dist, math.atan2(dy, dx))
        
        state.extend([
            closest_bricks[0] / math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2),  # Normalized distance to closest brick
            math.sin(closest_bricks[1]),                                        # Sin of angle to closest brick
            math.cos(closest_bricks[1])                                         # Cos of angle to closest brick
        ])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        reward = 0
        done = False
        
        # Apply continuous action (paddle velocity)
        self.paddle_x = np.clip(
            self.paddle_x + action * PADDLE_SPEED,
            0,
            SCREEN_WIDTH - PADDLE_WIDTH
        )
        
        # Update ball position
        new_ball_x = self.ball_x + self.ball_dx
        new_ball_y = self.ball_y + self.ball_dy
        
        # Wall collisions
        if new_ball_x <= BALL_SIZE or new_ball_x >= SCREEN_WIDTH - BALL_SIZE:
            self.ball_dx = -self.ball_dx
            new_ball_x = self.ball_x + self.ball_dx
        
        if new_ball_y <= BALL_SIZE:
            self.ball_dy = -self.ball_dy
            new_ball_y = self.ball_y + self.ball_dy
        
        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_x, SCREEN_HEIGHT - 50, PADDLE_WIDTH, PADDLE_HEIGHT)
        if (new_ball_y + BALL_SIZE >= paddle_rect.top and 
            paddle_rect.left <= new_ball_x <= paddle_rect.right):
            
            hit_pos = (new_ball_x - paddle_rect.left) / PADDLE_WIDTH
            angle = (hit_pos - 0.5) * math.pi / 3
            
            speed = math.sqrt(self.ball_dx ** 2 + self.ball_dy ** 2)
            self.ball_dx = speed * math.sin(angle)
            self.ball_dy = -speed * math.cos(angle)
            
            new_ball_y = paddle_rect.top - BALL_SIZE
            reward += 1  # Reward for hitting the ball
        
        # Brick collisions
        for brick in self.bricks:
            if brick['active'] and brick['rect'].collidepoint(new_ball_x, new_ball_y):
                brick['active'] = False
                self.ball_dy = -self.ball_dy
                reward += 10  # Reward for breaking a brick
                self.score += 10
                break
        
        # Game over conditions
        if new_ball_y > SCREEN_HEIGHT:
            done = True
            reward = -50  # Penalty for losing the ball
        
        if not any(brick['active'] for brick in self.bricks):
            done = True
            reward += 100  # Bonus for clearing all bricks
        
        self.ball_x = new_ball_x
        self.ball_y = new_ball_y
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_state(), reward, done
    
    def render(self):
        self.screen.fill((0, 0, 0))
        
        # Draw bricks
        for brick in self.bricks:
            if brick['active']:
                pygame.draw.rect(self.screen, brick['color'], brick['rect'])
        
        # Draw paddle
        pygame.draw.rect(self.screen, WHITE, 
                        (self.paddle_x, SCREEN_HEIGHT - 50, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Draw ball
        pygame.draw.circle(self.screen, WHITE, 
                         (int(self.ball_x), int(self.ball_y)), BALL_SIZE)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)