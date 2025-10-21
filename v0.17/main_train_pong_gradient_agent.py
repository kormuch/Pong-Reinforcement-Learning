"""
RL Training Pipeline for Non-Visual Policy Gradient Agent
Complete training workflow in a single reusable function.

Usage:
    from rl_training_pipeline import train_policy_gradient_agent
    agent, trainer = train_policy_gradient_agent()

Author: [Your Name]
Date: 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from env_custom_pong_simulator import CustomPongSimulator
from rl_feature_extractor import FeatureExtractor
from rl_policy_gradient_agent import PolicyGradientAgent
from rl_pong_training_complete import PongTrainer
from config import EnvConfig, TrainingConfig, PolicyGradientAgentConfig


def train_policy_gradient_agent():
    """
    Complete training pipeline for non-visual policy gradient agent.
    
    All configuration comes from config.py classes:
    - EnvConfig: Environment settings
    - PolicyGradientAgentConfig: Agent architecture
    - TrainingConfig: Training parameters
    
    Returns:
        tuple: (agent, trainer) - Trained agent and trainer with metrics
    """
    print("\n" + "=" * 60)
    print("PONG RL TRAINING PIPELINE - POLICY GRADIENT AGENT")
    print("=" * 60)
    
    # Initialize environment
    print("\n✓ Initializing Pong environment...")
    env = CustomPongSimulator(**EnvConfig.get_env_params())
    print(f"  Court: {EnvConfig.WIDTH}×{EnvConfig.HEIGHT}")
    print(f"  CPU Difficulty: {EnvConfig.COMPUTER_DIFFICULTY}")
    
    # Create feature extractor
    print("\n✓ Creating feature extractor...")
    feature_extractor = FeatureExtractor(env)
    print(f"  Features: {', '.join(feature_extractor.feature_names)}")
    print(f"  Input Size: {feature_extractor.num_features}")
    
    # Create agent
    print("\n✓ Creating Policy Gradient agent...")
    agent = PolicyGradientAgent(
        input_size=feature_extractor.num_features,
        hidden_size=PolicyGradientAgentConfig.HIDDEN_SIZE,
        learning_rate=PolicyGradientAgentConfig.LEARNING_RATE,
        discount=PolicyGradientAgentConfig.DISCOUNT_FACTOR
    )
    print(f"  Architecture: {agent.get_network_summary()['input_size']} → "
          f"{agent.get_network_summary()['hidden_size']} → "
          f"{agent.get_network_summary()['output_size']}")
    print(f"  Total Parameters: {agent.get_network_summary()['total_params']}")
    print(f"  Learning Rate: {PolicyGradientAgentConfig.LEARNING_RATE}")
    print(f"  Discount Factor: {PolicyGradientAgentConfig.DISCOUNT_FACTOR}")
    
    # Create trainer
    print("\n✓ Creating trainer...")
    trainer = PongTrainer(
        agent, 
        env, 
        feature_extractor,
        max_episodes=TrainingConfig.MAX_EPISODES
    )
    
    # Train agent
    print("\n✓ Starting training...")
    trainer.train(verbose_freq=TrainingConfig.PRINT_EVERY_N_EPISODES)
    
    # Save agent
    if TrainingConfig.SAVE_AFTER_TRAINING:
        print("\n✓ Saving trained agent...")
        agent.save(TrainingConfig.MODEL_SAVE_PATH)
    
    # Generate plots
    if TrainingConfig.GENERATE_PLOTS:
        print("\n✓ Generating training plots...")
        trainer.plot_learning_curve(window=TrainingConfig.PLOT_WINDOW_SIZE)
    
    # Evaluate agent
    if TrainingConfig.EVAL_AFTER_TRAINING:
        print("\n✓ Evaluating trained agent...")
        trainer.evaluate(num_episodes=TrainingConfig.EVAL_EPISODES)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60 + "\n")
    
    return agent, trainer


if __name__ == "__main__":
    train_policy_gradient_agent()