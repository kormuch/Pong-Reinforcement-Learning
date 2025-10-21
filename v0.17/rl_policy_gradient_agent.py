"""
Policy Gradient Agent for Non-Visual Pong Learning
Simple neural network agent using REINFORCE algorithm.

Author: [Your Name]
Date: 2025
"""
import numpy as np


class PolicyGradientAgent:
    """
    Policy Gradient Agent using REINFORCE algorithm.
    
    The agent learns a policy: P(action | state) = how likely each action is given current state
    
    Training: Maximize reward by adjusting policy toward high-reward actions
    """
    
    def __init__(self, input_size=None, hidden_size=None, learning_rate=None, discount=None):
        """
        Initialize agent with neural network.
        Parameters from PolicyGradientAgentConfig if not provided.
        
        Args:
            input_size: Size of feature vector (8 for Pong)
            hidden_size: Number of hidden neurons
            learning_rate: Learning rate for gradient descent
            discount: Discount factor for future rewards (gamma)
        """
        from config import PolicyGradientAgentConfig
        
        self.input_size = input_size or PolicyGradientAgentConfig.INPUT_SIZE
        self.hidden_size = hidden_size or PolicyGradientAgentConfig.HIDDEN_SIZE
        self.output_size = PolicyGradientAgentConfig.OUTPUT_SIZE
        self.learning_rate = learning_rate or PolicyGradientAgentConfig.LEARNING_RATE
        self.discount = discount or PolicyGradientAgentConfig.DISCOUNT_FACTOR
        
        self.rmsprop_decay = PolicyGradientAgentConfig.RMSPROP_DECAY
        self.eps = PolicyGradientAgentConfig.RMSPROP_EPSILON
        
        # Initialize network weights (small random values)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        
        # RMSprop accumulators
        self.W1_acc = np.zeros_like(self.W1)
        self.b1_acc = np.zeros_like(self.b1)
        self.W2_acc = np.zeros_like(self.W2)
        self.b2_acc = np.zeros_like(self.b2)
        
        # History for learning analysis
        self.loss_history = []
    
    def forward(self, features):
        """
        Forward pass through network.
        
        Args:
            features: Input features of shape (1, input_size)
        
        Returns:
            action_probs: Probability distribution over actions (1, 3)
        """
        # Hidden layer with ReLU activation
        self.hidden = np.maximum(0, np.dot(features, self.W1) + self.b1)
        
        # Output layer with softmax
        logits = np.dot(self.hidden, self.W2) + self.b2
        self.action_probs = self._softmax(logits)
        
        return self.action_probs
    
    def _softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def select_action(self, features):
        """
        Select action based on current policy.
        
        Args:
            features: Input features
        
        Returns:
            action: Selected action (0, 1, or 2)
            action_prob: Probability of selected action (for loss calculation)
        """
        probs = self.forward(features)
        action = np.random.choice(self.output_size, p=probs[0])
        action_prob = probs[0, action]
        
        return action, action_prob
    
    def train_on_episode(self, trajectory):
        """
        Train agent on completed episode using REINFORCE.
        
        REINFORCE: Loss = -sum(log(P(a|s)) * G_t)
        Where G_t is discounted future reward
        
        Args:
            trajectory: List of (features, action, reward) tuples
        """
        if len(trajectory) == 0:
            return
        
        # Calculate discounted cumulative rewards
        discounted_rewards = self._calculate_discounted_rewards(trajectory)
        
        # Normalize rewards (helps learning stability)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Accumulate gradients
        dW1, db1, dW2, db2 = 0, 0, 0, 0
        
        for i, (features, action, _) in enumerate(trajectory):
            # Forward pass
            probs = self.forward(features)
            
            # Policy gradient loss
            # We want to increase log probability of actions that led to high rewards
            action_prob = probs[0, action]
            log_prob = np.log(action_prob + 1e-8)
            loss = -log_prob * discounted_rewards[i]
            
            # Backward pass
            # Gradient of loss w.r.t. action logits
            action_one_hot = np.zeros((1, self.output_size))
            action_one_hot[0, action] = 1
            
            d_logits = (probs - action_one_hot) * discounted_rewards[i]
            
            # Backprop through output layer
            dW2 += np.dot(self.hidden.T, d_logits)
            db2 += d_logits
            
            # Backprop through hidden layer
            d_hidden = np.dot(d_logits, self.W2.T)
            d_hidden[self.hidden <= 0] = 0  # ReLU gradient
            
            dW1 += np.dot(features.T, d_hidden)
            db1 += d_hidden
        
        # Average gradients
        dW1 /= len(trajectory)
        db1 /= len(trajectory)
        dW2 /= len(trajectory)
        db2 /= len(trajectory)
        
        # RMSprop optimization
        self._rmsprop_update(dW1, db1, dW2, db2)
        
        # Track loss
        mean_loss = np.mean([-np.log(probs[0, trajectory[i][1]] + 1e-8) * discounted_rewards[i] 
                             for i in range(len(trajectory))])
        self.loss_history.append(mean_loss)
    
    def _calculate_discounted_rewards(self, trajectory):
        """
        Calculate discounted cumulative rewards.
        
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        
        Args:
            trajectory: List of (features, action, reward) tuples
        
        Returns:
            np.ndarray: Discounted rewards for each step
        """
        discounted_rewards = np.zeros(len(trajectory))
        cumulative_reward = 0
        
        # Process in reverse order
        for t in reversed(range(len(trajectory))):
            _, _, reward = trajectory[t]
            cumulative_reward = reward + self.discount * cumulative_reward
            discounted_rewards[t] = cumulative_reward
        
        return discounted_rewards
    
    def _rmsprop_update(self, dW1, db1, dW2, db2):
        """
        RMSprop optimization step.
        
        Args:
            dW1, db1, dW2, db2: Gradients for each parameter
        """
        # Accumulate squared gradients
        self.W1_acc = self.rmsprop_decay * self.W1_acc + (1 - self.rmsprop_decay) * (dW1 ** 2)
        self.b1_acc = self.rmsprop_decay * self.b1_acc + (1 - self.rmsprop_decay) * (db1 ** 2)
        self.W2_acc = self.rmsprop_decay * self.W2_acc + (1 - self.rmsprop_decay) * (dW2 ** 2)
        self.b2_acc = self.rmsprop_decay * self.b2_acc + (1 - self.rmsprop_decay) * (db2 ** 2)
        
        # Update weights
        self.W1 -= self.learning_rate * dW1 / (np.sqrt(self.W1_acc) + self.eps)
        self.b1 -= self.learning_rate * db1 / (np.sqrt(self.b1_acc) + self.eps)
        self.W2 -= self.learning_rate * dW2 / (np.sqrt(self.W2_acc) + self.eps)
        self.b2 -= self.learning_rate * db2 / (np.sqrt(self.b2_acc) + self.eps)
    
    def get_action_names(self):
        """Return action names for debugging"""
        return ['stay', 'up', 'down']
    
    def get_network_summary(self):
        """Get summary of network architecture"""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'total_params': (self.input_size * self.hidden_size + 
                           self.hidden_size + 
                           self.hidden_size * self.output_size + 
                           self.output_size)
        }