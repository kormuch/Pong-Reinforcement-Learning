import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from env_custom_pong_simulator import CustomPongSimulator
from rl_feature_extractor import FeatureExtractor
from rl_policy_gradient_agent import PolicyGradientAgent
from rl_pong_training_complete import PongTrainer
from config import EnvConfig, TrainingConfig, PolicyGradientAgentConfig

# Load and save config
TrainingConfig.load_from_json("config/config_training.json")
TrainingConfig.save_active_config()


def train_policy_gradient_agent():
    print("\n" + "=" * 60)
    print("PONG RL TRAINING PIPELINE - POLICY GRADIENT AGENT")
    print("=" * 60)
    
    # Initialize environment
    print("\n✓ Initializing Pong environment...")
    env = CustomPongSimulator(**EnvConfig.get_env_params())
    print(f"  Court: {EnvConfig.WIDTH}×{EnvConfig.HEIGHT}")
    print(f"  CPU Difficulty: {EnvConfig.OPPONENT_DIFFICULTY}")
    
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
    summary = agent.get_network_summary()
    print(f"  Architecture: {summary['input_size']} → {summary['hidden_size']} → {summary['output_size']}")
    print(f"  Total Parameters: {summary['total_params']}")
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
    
    # Generate and save plots
    if TrainingConfig.GENERATE_PLOTS:
        print("\n✓ Generating and saving training plots...")
        plt.figure()
        trainer.plot_learning_curve(window=TrainingConfig.PLOT_WINDOW_SIZE)

        # === Handle file or folder path automatically ===
        base_path = TrainingConfig.PLOT_SAVE_PATH
        base_dir, base_ext = os.path.splitext(base_path)

        if base_ext:  # It is a file path
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            plot_path = base_path
        else:  # It is a folder path
            os.makedirs(base_path, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(base_path, f"training_curve_{timestamp}.png")

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Plot saved to: {plot_path}")
    
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
