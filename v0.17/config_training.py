# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

class TrainingConfig:
    """Training loop configuration - All hyperparameters for agent learning"""
    
    # =========================================================================
    # REWARD STRUCTURE (Atari-like with shaping)
    # =========================================================================
    REWARD_SCORE = 2.0           # Reward for scoring a point
    REWARD_OPPONENT_SCORE = -1.0 # Penalty when opponent scores
    REWARD_BALL_HIT = 1.0        # Small reward for returning ball (reward shaping)
    
    # =========================================================================
    # TRAINING LOOP PARAMETERS
    # =========================================================================
    MAX_EPISODES = 2000           # Maximum episodes to train
    BATCH_SIZE = 1                # Episodes per policy update (1 for REINFORCE)
    PRINT_EVERY_N_EPISODES = 50   # Progress reporting frequency
    RUNNING_REWARD_DECAY = 0.99   # Decay for running average calculation
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    EVAL_EPISODES = 10            # Number of evaluation episodes
    EVAL_AFTER_TRAINING = True    # Run evaluation after training completes
    
    # =========================================================================
    # MODEL PERSISTENCE
    # =========================================================================
    MODEL_SAVE_PATH = 'pong_agent.npz'  # File to save trained agent
    SAVE_AFTER_TRAINING = True           # Auto-save after training
    LOAD_EXISTING_MODEL = False          # Load pre-trained model if exists
    
    # =========================================================================
    # PLOTTING AND VISUALIZATION
    # =========================================================================
    GENERATE_PLOTS = True         # Generate learning curves
    PLOT_WINDOW_SIZE = 50         # Moving average window
    PLOT_SAVE_PATH = 'pong_training_results.png'
    
    # =========================================================================
    # EARLY STOPPING (optional)
    # =========================================================================
    USE_EARLY_STOPPING = False
    EARLY_STOP_THRESHOLD = 0.5    # Stop if running reward exceeds this
    EARLY_STOP_PATIENCE = 100     # Episodes to wait before stopping
    
    @classmethod
    def get_reward_structure(cls):
        """Returns reward dictionary"""
        return {
            'score': cls.REWARD_SCORE,
            'opponent_score': cls.REWARD_OPPONENT_SCORE,
            'ball_hit': cls.REWARD_BALL_HIT,
        }
    
    @classmethod
    def get_training_params(cls):
        """Returns dictionary of training parameters"""
        return {
            'max_episodes': cls.MAX_EPISODES,
            'batch_size': cls.BATCH_SIZE,
            'print_every': cls.PRINT_EVERY_N_EPISODES,
            'running_reward_decay': cls.RUNNING_REWARD_DECAY,
            'eval_episodes': cls.EVAL_EPISODES,
        }