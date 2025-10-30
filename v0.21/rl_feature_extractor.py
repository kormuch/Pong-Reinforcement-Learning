"""
Feature Extractor for Non-Visual Pong Agent
Converts raw game state into interpretable features for RL learning.

Author: [Your Name]
Date: 2025
"""
import numpy as np


class FeatureExtractor:
    """Extracts features from Pong game state"""
    
    def __init__(self, env):
        """
        Initialize feature extractor.
        
        Args:
            env: CustomPongSimulator instance
        """
        self.env = env
        self.feature_names = [
            'player_y',
            'cpu_y',
            'ball_x',
            'ball_y',
            'ball_vx',
            'ball_vy',
            'player_score',
            'cpu_score',
        ]
        self.num_features = len(self.feature_names)
    
    def extract(self):
        """
        Extract feature vector from current game state.
        
        Returns:
            np.ndarray: Feature vector of shape (1, num_features)
        """
        game_info = self.env.get_game_info()
        ball_pos = game_info['ball_position']
        ball_vel = game_info['ball_velocity']
        
        features = np.array([
            self.env.player_y,           # Player paddle Y position
            self.env.cpu_y,              # CPU paddle Y position
            ball_pos[0],                 # Ball X position
            ball_pos[1],                 # Ball Y position
            ball_vel[0],                 # Ball X velocity
            ball_vel[1],                 # Ball Y velocity
            self.env.player_score,       # Player score
            self.env.cpu_score,          # CPU score
        ], dtype=np.float32)
        
        return features.reshape(1, -1)
    
    def normalize(self, features):
        """
        Normalize features to reasonable ranges.
        
        This helps the network learn more effectively by scaling inputs
        to roughly [-1, 1] range.
        
        Args:
            features: Raw feature vector from extract()
        
        Returns:
            np.ndarray: Normalized features
        """
        normalized = features.copy()
        
        # Normalize positions (0-192 for Y, 0-160 for X) to [-1, 1]
        normalized[0, 0] = (features[0, 0] - self.env.height/2) / (self.env.height/2)  # player_y
        normalized[0, 1] = (features[0, 1] - self.env.height/2) / (self.env.height/2)  # cpu_y
        normalized[0, 2] = (features[0, 2] - self.env.width/2) / (self.env.width/2)    # ball_x
        normalized[0, 3] = (features[0, 3] - self.env.height/2) / (self.env.height/2)  # ball_y
        
        # Velocities are already small, just scale slightly
        normalized[0, 4] = features[0, 4] / 4.0  # ball_vx
        normalized[0, 5] = features[0, 5] / 4.0  # ball_vy
        
        # Scores (0-21) to [-1, 1]
        normalized[0, 6] = (features[0, 6] - 10.5) / 10.5  # player_score
        normalized[0, 7] = (features[0, 7] - 10.5) / 10.5  # cpu_score
        
        return normalized
    
    def get_feature_description(self, features):
        """
        Get human-readable description of features.
        
        Useful for debugging and understanding what the agent "sees".
        
        Args:
            features: Feature vector
        
        Returns:
            str: Formatted description
        """
        lines = ["Game Features:"]
        for i, name in enumerate(self.feature_names):
            value = features[0, i]
            lines.append(f"  {name:15s}: {value:7.2f}")
        return "\n".join(lines)
    
    def get_relative_position(self):
        """
        Get relative position of ball vs paddles (useful feature).
        
        Returns:
            tuple: (dy_player, dy_cpu) - how far ball is above/below each paddle
        """
        game_info = self.env.get_game_info()
        ball_y = game_info['ball_position'][1]
        
        dy_player = ball_y - self.env.player_y
        dy_cpu = ball_y - self.env.cpu_y
        
        return dy_player, dy_cpu