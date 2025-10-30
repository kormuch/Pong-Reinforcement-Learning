# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

import json
import os
import datetime

json_path="config/config_training.json"

class TrainingConfig:
    """Training loop configuration - All hyperparameters for agent learning"""
    
    # =========================================================================
    # DEFAULT PARAMETERS
    # =========================================================================
    
    # REWARD STRUCTURE
    REWARD_SCORE = 1.0
    REWARD_OPPONENT_SCORE = -1.0
    REWARD_BALL_HIT = 0.1
    
    # TRAINING LOOP
    MAX_EPISODES = 1000
    BATCH_SIZE = 1
    PRINT_EVERY_N_EPISODES = 50
    RUNNING_REWARD_DECAY = 0.99
    
    # EVALUATION
    EVAL_EPISODES = 10
    EVAL_AFTER_TRAINING = True
    
    # MODEL PERSISTENCE
    MODEL_SAVE_PATH = "models/policy_gradient_agent.npz"
    SAVE_AFTER_TRAINING = True
    LOAD_EXISTING_MODEL = False
    
    # PLOTTING
    GENERATE_PLOTS = True
    PLOT_WINDOW_SIZE = 50
    # Construct the plot save path with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    PLOT_SAVE_PATH = f"plots/pong_training_results_{timestamp}.png"
    
    # EARLY STOPPING
    USE_EARLY_STOPPING = False
    EARLY_STOP_THRESHOLD = 0.5
    EARLY_STOP_PATIENCE = 100
    
    # =========================================================================
    # LOAD CONFIG FROM JSON
    # =========================================================================
    
    @classmethod
    def load_from_json(cls, json_path=json_path):
        """
        Override default parameters from a JSON file if it exists.
        """
        if not os.path.exists(json_path):
            print(f"⚠ No config file found at {json_path}. Using defaults.")
            return
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        for key, value in data.items():
            key_upper = key.upper()
            if hasattr(cls, key_upper):
                setattr(cls, key_upper, value)
                print(f"✓ Loaded {key_upper} = {value}")
            else:
                print(f"⚠ Unknown config key: {key}")
    
    @classmethod
    def save_active_config(cls, save_dir="logs"):
        """Save the current effective config to a timestamped JSON file."""
        os.makedirs(save_dir, exist_ok=True)
        config_dict = {k: getattr(cls, k) for k in dir(cls)
                       if k.isupper() and not k.startswith("__")}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"used_config_{timestamp}.json")
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"✓ Saved active config to {path}")
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @classmethod
    def get_reward_structure(cls):
        return {
            "score": cls.REWARD_SCORE,
            "opponent_score": cls.REWARD_OPPONENT_SCORE,
            "ball_hit": cls.REWARD_BALL_HIT,
        }
    
    @classmethod
    def get_training_params(cls):
        return {
            "max_episodes": cls.MAX_EPISODES,
            "batch_size": cls.BATCH_SIZE,
            "print_every": cls.PRINT_EVERY_N_EPISODES,
            "running_reward_decay": cls.RUNNING_REWARD_DECAY,
            "eval_episodes": cls.EVAL_EPISODES,
        }


# Auto-load from JSON when module is imported
TrainingConfig.load_from_json()