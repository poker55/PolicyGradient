import torch
import math
# Constants
SCREEN_WIDTH = 800#pixels
SCREEN_HEIGHT = 600
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 10
BALL_SIZE = 8
BRICK_WIDTH = 60
BRICK_HEIGHT = 20
BRICK_PADDING = 4
BRICK_ROWS = 5
BRICK_COLS = 12
BALL_SPEED = 5
ANGLE = math.radians(30)

# Colors
WHITE = (255, 255, 255)
BRICK_COLORS = [
    (255, 107, 107),
    (255, 165, 107),
    (255, 215, 107),
    (107, 255, 107),
    (107, 107, 255)
]

# Training Parameters
LEARNING_RATE = 0.01  # Reduced learning rate for stability
GAMMA = 0.99
MAX_EPISODES = 1000
PADDLE_SPEED = 100  # Reduced paddle speed for smoother control
MIN_STD = 0.1  # Minimum standard deviation for exploration
MAX_STEPS_PER_EPISODE = 3000  # Maximum steps before forcing reset

EXPLORATION_NOISE = 0.5  # Higher initial noise
NOISE_DECAY = 0.995     # Slower decay
MIN_NOISE = 0.1        # Keep some minimum noise