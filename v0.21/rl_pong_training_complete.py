"""
Complete Non-Visual Pong Agent Training - All Parameters from Config
Contains FeatureExtractor, PolicyGradientAgent, and Training Loop in one file.

All hyperparameters are pulled from TrainingConfig for centralized management.

Usage:
    python pong_training_complete.py

Author: [Your Name]
Date: 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from env_custom_pong_simulator import CustomPongSimulator
from rl_policy_gradient_agent import PolicyGradientAgent
from config import EnvConfig, TrainingConfig, PolicyGradientAgentConfig


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    """Extracts features from Pong game state for RL agent"""
    
    def __init__(self, env):
        self.env = env
        self.feature_names = [
            'player_y', 'cpu_y', 'ball_x', 'ball_y',
            'ball_vx', 'ball_vy', 'player_score', 'cpu_score',
        ]
        self.num_features = len(self.feature_names)
    
    def extract(self):
        """Extract feature vector from current game state"""
        game_info = self.env.get_game_info()
        ball_pos = game_info['ball_position']
        ball_vel = game_info['ball_velocity']
        
        features = np.array([
            self.env.player_y,
            self.env.cpu_y,
            ball_pos[0],
            ball_pos[1],
            ball_vel[0],
            ball_vel[1],
            self.env.player_score,
            self.env.cpu_score,
        ], dtype=np.float32)
        
        return features.reshape(1, -1)
    
    def normalize(self, features):
        """Normalize features to [-1, 1] range for better learning"""
        normalized = features.copy()
        
        normalized[0, 0] = (features[0, 0] - self.env.height/2) / (self.env.height/2)
        normalized[0, 1] = (features[0, 1] - self.env.height/2) / (self.env.height/2)
        normalized[0, 2] = (features[0, 2] - self.env.width/2) / (self.env.width/2)
        normalized[0, 3] = (features[0, 3] - self.env.height/2) / (self.env.height/2)
        normalized[0, 4] = features[0, 4] / 4.0
        normalized[0, 5] = features[0, 5] / 4.0
        normalized[0, 6] = (features[0, 6] - 10.5) / 10.5
        normalized[0, 7] = (features[0, 7] - 10.5) / 10.5
        
        return normalized
    
    def get_feature_description(self, features):
        """Get human-readable description of features"""
        lines = ["Game Features:"]
        for i, name in enumerate(self.feature_names):
            value = features[0, i]
            lines.append(f"  {name:15s}: {value:7.2f}")
        return "\n".join(lines)



# =============================================================================
# TRAINER
# =============================================================================

class PongTrainer:
    """Manages training loop for Pong agent - all config from TrainingConfig"""
    
    def __init__(self, agent, env, feature_extractor, max_episodes=None):
        self.agent = agent
        self.env = env
        self.feature_extractor = feature_extractor
        self.max_episodes = max_episodes or TrainingConfig.MAX_EPISODES
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.running_reward = None
        self.best_reward = -float('inf')
    
    def play_episode(self, render=False):
        """Play one episode and collect trajectory"""
        state = self.env.reset()
        trajectory = []
        total_reward = 0
        step_count = 0
        
        while True:
            features = self.feature_extractor.extract()
            features_normalized = self.feature_extractor.normalize(features)
            
            action, action_prob = self.agent.select_action(features_normalized)
            state, reward, done, info = self.env.step(action)
            
            trajectory.append((features_normalized, action, reward))
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        return total_reward, step_count, trajectory
    
    def train(self, verbose_freq=None):
        """Main training loop - uses TrainingConfig for all parameters"""
        verbose_freq = verbose_freq or TrainingConfig.PRINT_EVERY_N_EPISODES
        
        print(f"\n{'='*60}")
        print(f"TRAINING PONG AGENT")
        print(f"{'='*60}")
        print(f"Agent: {self.agent.get_network_summary()}")
        print(f"Learning Rate: {self.agent.learning_rate}")
        print(f"Discount Factor: {self.agent.discount}")
        print(f"Max Episodes: {self.max_episodes}")
        print(f"RMSprop Decay: {self.agent.rmsprop_decay}")
        print(f"Running Reward Decay: {TrainingConfig.RUNNING_REWARD_DECAY}")
        print(f"Reward Structure: Score={TrainingConfig.REWARD_SCORE}, Hit={TrainingConfig.REWARD_BALL_HIT}, Loss={TrainingConfig.REWARD_OPPONENT_SCORE}")
        print(f"{'='*60}\n")
        
        for episode in range(1, self.max_episodes + 1):
            total_reward, episode_length, trajectory = self.play_episode()
            
            self.agent.train_on_episode(trajectory)
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            
            if self.running_reward is None:
                self.running_reward = total_reward
            else:
                self.running_reward = (TrainingConfig.RUNNING_REWARD_DECAY * self.running_reward + 
                                      (1 - TrainingConfig.RUNNING_REWARD_DECAY) * total_reward)
            
            if total_reward > self.best_reward:
                self.best_reward = total_reward
            
            if episode % verbose_freq == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Running Avg: {self.running_reward:7.2f} | "
                      f"Best: {self.best_reward:7.2f} | "
                      f"Length: {episode_length}")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"Final Running Average: {self.running_reward:.2f}")
        print(f"Best Episode Reward: {self.best_reward:.2f}")
        print(f"{'='*60}\n")
    
    def plot_learning_curve(self, window=None):
        """Plot learning progress - uses TrainingConfig for window size"""
        window = window or TrainingConfig.PLOT_WINDOW_SIZE
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Raw rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, 
                                     np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.episode_rewards)), moving_avg, 
                   label=f'Moving Avg (window={window})', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Learning Curve - Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Episode length
        ax = axes[0, 1]
        ax.plot(self.episode_lengths, alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Duration Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Distribution of rewards
        ax = axes[1, 0]
        ax.hist(self.episode_rewards, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(self.episode_rewards), color='r', 
                  linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Training loss
        ax = axes[1, 1]
        if len(self.agent.loss_history) > 0:
            ax.plot(self.agent.loss_history, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title('Policy Gradient Loss')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = TrainingConfig.PLOT_SAVE_PATH
        plt.savefig(save_path, dpi=100)
        print(f"✓ Learning curve saved to '{save_path}'")
        plt.show()
    
    def evaluate(self, num_episodes=None):
        """Evaluate trained agent - uses TrainingConfig for num_episodes"""
        num_episodes = num_episodes or TrainingConfig.EVAL_EPISODES
        
        print(f"\n{'='*60}")
        print(f"EVALUATING TRAINED AGENT ({num_episodes} episodes)")
        print(f"{'='*60}\n")
        
        eval_rewards = []
        eval_wins = 0
        eval_lengths = []
        
        for ep in range(num_episodes):
            reward, length, _ = self.play_episode()
            eval_rewards.append(reward)
            eval_lengths.append(length)
            
            if self.env.player_score > self.env.cpu_score:
                eval_wins += 1
            
            print(f"Eval Episode {ep+1:2d}: Reward={reward:7.2f}, "
                  f"Player={self.env.player_score:2d}, CPU={self.env.cpu_score:2d}, "
                  f"Length={length}")
        
        print(f"\n{'='*60}")
        print(f"Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"Reward Range: [{np.min(eval_rewards):.2f}, {np.max(eval_rewards):.2f}]")
        print(f"Win Rate: {eval_wins/num_episodes:.1%}")
        print(f"Mean Episode Length: {np.mean(eval_lengths):.0f} steps")
        print(f"{'='*60}\n")