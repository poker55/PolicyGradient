from train import train_agent

if __name__ == "__main__":
    train_agent()

    #The game sometimes ends even when the ball did not miss the paddle. It ends because the episode limit is reached.
    #Training (1000 episodes total)
    #└── Episode (max 3000 steps)
    #    ├── Game 1 (until ball is lost)
    #    │   ├── Step 1 (one paddle movement)
    #    │   ├── Step 2
    #    │   └── ... (until ball is lost)
    #    ├── Game 2
    #    │   ├── Step 500
    #    │   ├── Step 501
    #    │   └── ... (until ball is lost)
    #    └── ... (until 3000 steps or episode ends)