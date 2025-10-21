"""
Collision Detection for Atari-Authentic Pong
Handles paddle collisions and scoring logic.

Author: [Your Name]
Date: 2025
"""
import numpy as np


class CollisionDetector:
    """Detects and resolves paddle collisions and scoring events"""
    
    def __init__(self, court_width, court_height, ball_size, paddle_width, paddle_height):
        """
        Initialize collision detector.
        
        Args:
            court_width, court_height: Court dimensions
            ball_size: Size of ball
            paddle_width, paddle_height: Paddle dimensions
        """
        self.court_width = court_width
        self.court_height = court_height
        self.ball_size = ball_size
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
    
    def check_player_paddle_collision(self, ball_x, ball_y, player_paddle_x, player_y):
        """
        Check if ball collides with player paddle (right side).
        
        Args:
            ball_x, ball_y: Ball position
            player_paddle_x: X position of player paddle
            player_y: Y position of player paddle
        
        Returns:
            bool: True if collision detected
        """
        # X-axis overlap
        ball_right = ball_x + self.ball_size
        paddle_left = player_paddle_x
        paddle_right = player_paddle_x + self.paddle_width
        
        x_collision = (ball_right >= paddle_left and ball_x <= paddle_right)
        
        # Y-axis overlap
        ball_bottom = ball_y + self.ball_size
        paddle_top = player_y
        paddle_bottom = player_y + self.paddle_height
        
        y_collision = (ball_bottom > paddle_top and ball_y < paddle_bottom)
        
        return x_collision and y_collision
    
    def check_cpu_paddle_collision(self, ball_x, ball_y, cpu_paddle_x, cpu_y):
        """
        Check if ball collides with CPU paddle (left side).
        
        Args:
            ball_x, ball_y: Ball position
            cpu_paddle_x: X position of CPU paddle
            cpu_y: Y position of CPU paddle
        
        Returns:
            bool: True if collision detected
        """
        # X-axis overlap
        ball_left = ball_x
        ball_right = ball_x + self.ball_size
        paddle_left = cpu_paddle_x
        paddle_right = cpu_paddle_x + self.paddle_width
        
        x_collision = (ball_right >= paddle_left and ball_left <= paddle_right)
        
        # Y-axis overlap
        ball_bottom = ball_y + self.ball_size
        paddle_top = cpu_y
        paddle_bottom = cpu_y + self.paddle_height
        
        y_collision = (ball_bottom > paddle_top and ball_y < paddle_bottom)
        
        return x_collision and y_collision
    
    def check_scoring(self, ball_x):
        """
        Check if ball went off-screen (scoring event).
        
        Args:
            ball_x: Ball X position
        
        Returns:
            str or None: 'player_scored', 'cpu_scored', or None
        """
        if ball_x <= 0:
            return 'player_scored'
        elif ball_x >= self.court_width:
            return 'cpu_scored'
        return None
    
    def get_paddle_hit_offset(self, ball_y, paddle_y):
        """
        Get normalized offset of where ball hit paddle.
        Used for angle calculations.
        
        Args:
            ball_y: Ball Y position
            paddle_y: Paddle Y position
        
        Returns:
            float: Normalized offset (-1.0 to +1.0)
                -1.0 = top edge
                 0.0 = center
                +1.0 = bottom edge
        """
        paddle_center = paddle_y + self.paddle_height / 2
        ball_center = ball_y + self.ball_size / 2
        
        # Normalized offset
        offset = (ball_center - paddle_center) / (self.paddle_height / 2)
        
        # Clamp to valid range
        return np.clip(offset, -1.0, 1.0)