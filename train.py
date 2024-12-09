import torch
import torch.optim as optim
import time
from tqdm import tqdm
from environment import BreakoutEnv
from policy import PolicyNetwork
from constants import *
import pygame
from torch.distributions import Normal

def train_agent():
    try:
        print("Initializing training environment...")
        print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        env = BreakoutEnv(render_mode="human")
        state_size = len(env.reset())
        print("State size:", state_size)
        
        policy = PolicyNetwork(state_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        #optimizer = optim.SGD(policy.parameters(), lr=0.01, momentum=0.9)
        print("Policy network initialized successfully")
        
        # Initialize progress tracking
        pbar = tqdm(range(MAX_EPISODES), desc="Training Progress")
        running_reward = 0
        start_time = time.time()
        
        # New episode structure variables
        games_per_episode = 5  # Number of complete games that form one episode
        max_steps_per_game = 10000  # More generous step limit per individual game
        best_score = 0
        #exploration_noise = EXPLORATION_NOISE

        for episode in pbar:
            episode_log_probs = []
            episode_rewards = []
            episode_total_reward = 0
            total_steps = 0
            games_completed = 0
            #exploration_noise = max(MIN_NOISE, exploration_noise * NOISE_DECAY)
            # Run multiple complete games to form one episode
            while games_completed < games_per_episode:
                state = env.reset()
                game_rewards = []
                game_log_probs = []
                steps_in_game = 0
                game_score = 0
                
                # Play one complete game
                while steps_in_game < max_steps_per_game:
                    steps_in_game += 1
                    total_steps += 1
                    
                    # Get action from policy
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )
                    
                    mean_std = policy(state_tensor)
                    mean, std = mean_std[0, 0], torch.exp(mean_std[0, 1])
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    # Clip action to valid range [-1, 1]
                    #noise = torch.rand(1).item() * exploration_noise
                    #action = action + noise
                    action = torch.clamp(action, -1, 1)
                    
                    # Step environment
                    next_state, reward, done = env.step(action.item())
                    
                    # Enhanced reward shaping
                    ball_x = next_state[1] * SCREEN_WIDTH
                    paddle_x = next_state[0] * SCREEN_WIDTH
                    paddle_center = paddle_x + PADDLE_WIDTH/2
                    
                    # Reward for good positioning (inversely proportional to distance)
                    positioning_reward = 1.0 / (1.0 + abs(ball_x - paddle_center) / SCREEN_WIDTH)
                    #positioning_reward = - abs(ball_x - paddle_center) / SCREEN_WIDTH
                    reward += positioning_reward * 5
                    
                    # Additional reward for maintaining volley
                    #if steps_in_game > 100:
                    #    reward += 0.01  # Small bonus for surviving longer
                    
                    game_log_probs.append(log_prob)
                    game_rewards.append(reward)
                    game_score += reward
                    
                    if done:  # Game naturally ended (ball lost or all bricks broken)
                        print(f"\nGame {games_completed + 1} completed! Score: {env.score} Steps: {steps_in_game}")
                        break
                        
                    state = next_state
                    
                    # Handle Pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nClosing game window...")
                            pygame.quit()
                            return
                
                # After game completion
                games_completed += 1
                episode_log_probs.extend(game_log_probs)
                episode_rewards.extend(game_rewards)
                episode_total_reward += game_score
                
                # Track best performance
                if env.score > best_score:
                    best_score = env.score
                    print(f"\nNew best score: {best_score}!")
                
            # Policy update after completing all games in the episode
            if len(episode_rewards) > 0:
                try:
                    # Calculate returns with episode-level normalization
                    R = 0
                    policy_loss = []
                    returns = []
                    
                    for r in episode_rewards[::-1]:
                        R = r + GAMMA * R
                        returns.insert(0, R)
                    
                    returns = torch.FloatTensor(returns).to(
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )
                    
                    # Normalize returns across the entire episode
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                    
                    # Calculate policy loss
                    for log_prob, R in zip(episode_log_probs, returns):
                        policy_loss.append(-log_prob * R)
                    
                    # Update policy
                    optimizer.zero_grad()
                    policy_loss = torch.stack(policy_loss).sum()
                    policy_loss.backward()
                    optimizer.step()
                    
                    # Update running reward and progress bar
                    running_reward = 0.05 * episode_total_reward + (1 - 0.05) * running_reward
                    
                    pbar.set_description(
                        f"Episode {episode + 1}/{MAX_EPISODES} | "
                        f"Games: {games_completed} | "
                        f"Avg Reward: {episode_total_reward/games_completed:.1f} | "
                        f"Best Score: {best_score}"
                    )
                    
                except Exception as e:
                    print(f"\nError during policy update: {e}")
                    continue
            
            # Save checkpoint every 50 episodes
            if (episode + 1) % 50 == 0:
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'running_reward': running_reward,
                    'best_score': best_score
                }, f'breakout_pg_checkpoint_{episode + 1}.pt')
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        torch.save({
            'episode': episode,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'running_reward': running_reward,
            'best_score': best_score
        }, 'breakout_pg_interrupted.pt')
    finally:
        print("\nTraining completed or interrupted. Final statistics:")
        print(f"Total episodes completed: {episode + 1}")
        print(f"Final running reward: {running_reward:.1f}")
        print(f"Best score achieved: {best_score}")
        print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")