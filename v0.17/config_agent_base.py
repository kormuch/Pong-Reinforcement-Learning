# =============================================================================
# BASE AGENT CONFIGURATION
# =============================================================================

class AgentConfig:
    """Base agent configuration - common parameters for all agents"""
    
    # Output size (same for all agents - Pong has 3 actions)
    OUTPUT_SIZE = 3  # Actions: stay, up, down
    
    # Common learning parameters
    DISCOUNT_FACTOR = 0.99  # Reward discount factor (gamma)
    
    @classmethod
    def get_agent_params(cls):
        """Returns dictionary of common agent parameters"""
        return {
            'output_size': cls.OUTPUT_SIZE,
            'discount_factor': cls.DISCOUNT_FACTOR,
        }