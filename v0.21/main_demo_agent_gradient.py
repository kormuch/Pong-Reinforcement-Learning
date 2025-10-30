"""
Demo: Watch Trained Policy Gradient Agent Play Pong (with GUI)
Completely standalone - loads agent and displays in GUI.

Usage:
    python main_demo_agent_gradient.py

Author: [Your Name]
Date: 2025
"""

import sys
import pygame
import numpy as np

# Import environment and GUI
from env_custom_pong_simulator import CustomPongSimulator
from config import EnvConfig, GUIConfig, VisualConfig, PolicyGradientAgentConfig

# Import AI components
from rl_feature_extractor import FeatureExtractor
from rl_policy_gradient_agent import PolicyGradientAgent


class AIDemoGUI:
    """GUI to watch trained AI agent play Pong"""
    
    def __init__(self, model_path='models/policy_gradient_v2.npz'):
        """Initialize demo with AI agent"""
        
        print("=" * 60)
        print("PONG AI AGENT DEMO (GUI)")
        print("=" * 60)
        
        # Step 1: Create environment
        print("\nStep 1: Creating environment...")
        self.env = CustomPongSimulator(**EnvConfig.get_env_params())
        print(f"  ✓ Court: {self.env.width}×{self.env.height}")
        print(f"  ✓ CPU Difficulty: {EnvConfig.OPPONENT_DIFFICULTY}")
        
        # Step 2: Load AI agent
        print("\nStep 2: Loading AI agent...")
        self.feature_extractor = FeatureExtractor(self.env)
        
        self.agent = PolicyGradientAgent(
            input_size=PolicyGradientAgentConfig.INPUT_SIZE,
            hidden_size=PolicyGradientAgentConfig.HIDDEN_SIZE,
            learning_rate=PolicyGradientAgentConfig.LEARNING_RATE,
            discount=PolicyGradientAgentConfig.DISCOUNT_FACTOR
        )
        
        try:
            self.agent.load(model_path)
            print(f"  ✓ Model loaded: {model_path}")
            print(f"  ✓ Network: {self.agent.input_size} → {self.agent.hidden_size} → {self.agent.output_size}")
        except FileNotFoundError:
            print(f"  ❌ Model not found: {model_path}")
            print("  Please train an agent first!")
            sys.exit(1)
        
        # Step 3: Setup Pygame GUI
        print("\nStep 3: Initializing GUI...")
        pygame.init()
        
        self.pixel_size = GUIConfig.PIXEL_SIZE
        self.fps = GUIConfig.GUI_FPS
        
        self.render_width = VisualConfig.RENDER_WIDTH
        self.render_height = VisualConfig.RENDER_HEIGHT
        
        self.display_width = self.render_width * self.pixel_size
        self.display_height = self.render_height * self.pixel_size
        
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("Atari Pong - AI Agent Demo")
        
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Colors
        self.COLOR_WHITE = GUIConfig.COLOR_WHITE
        self.COLOR_BLACK = GUIConfig.COLOR_BLACK
        self.COLOR_GREEN = GUIConfig.COLOR_GREEN
        self.COLOR_RED = GUIConfig.COLOR_RED
        
        # Game state
        self.running = True
        self.game_over = False
        self.winner = None
        self.show_debug = True  # Show AI info by default
        
        print("  ✓ GUI ready")
        print("\n" + "=" * 60)
        print("Controls:")
        print("  R   : Restart game")
        print("  D   : Toggle debug info")
        print("  ESC : Quit")
        print("=" * 60)
        
        # Reset game
        self.reset_game()
    
    def reset_game(self):
        """Reset game to initial state"""
        self.state = self.env.reset()
        self.game_over = False
        self.winner = None
    
    def get_ai_action(self):
        """Get action from AI agent"""
        features = self.feature_extractor.extract()
        action, action_prob = self.agent.select_action(features)
        return action, action_prob
    
    def handle_events(self):
        """Handle keyboard events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                if event.key == pygame.K_r:
                    print("\n🔄 Restarting game...")
                    self.reset_game()
                
                if event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
    
    def update_game(self, action):
        """Update game state"""
        if not self.game_over:
            self.state, reward, done, info = self.env.step(action)
            
            if done:
                self.game_over = True
                if info['done_reason'] == 'player_won':
                    self.winner = 'AI'
                    print(f"\n🎉 AI WON! Score: {info['player_score']}-{info['cpu_score']}")
                elif info['done_reason'] == 'cpu_won':
                    self.winner = 'CPU'
                    print(f"\n😞 AI LOST. Score: {info['player_score']}-{info['cpu_score']}")
                else:
                    self.winner = 'DRAW'
                    print(f"\n🤝 DRAW. Score: {info['player_score']}-{info['cpu_score']}")
    
    def render_game(self, current_action, action_prob):
        """Render game to screen"""
        # Get rendered image from environment
        rendered_image = self.env.render('rgb_array')
        
        # Handle grayscale
        if VisualConfig.COLOR_MODE == 'grayscale':
            if rendered_image.shape[-1] == 1:
                rendered_image = np.squeeze(rendered_image, axis=-1)
            rendered_rgb = np.stack([rendered_image, rendered_image, rendered_image], axis=-1)
        else:
            rendered_rgb = rendered_image
        
        # Scale up
        small_surface = pygame.surfarray.make_surface(
            np.transpose(rendered_rgb, (1, 0, 2))
        )
        scaled_surface = pygame.transform.scale(
            small_surface,
            (self.display_width, self.display_height)
        )
        
        self.screen.blit(scaled_surface, (0, 0))
        
        # Draw AI debug info
        if self.show_debug:
            self._draw_ai_info(current_action, action_prob)
        
        # Draw game over overlay
        if self.game_over:
            self._draw_game_over_overlay()
        
        pygame.display.flip()
    
    def _draw_ai_info(self, action, action_prob):
        """Draw AI decision info"""
        action_names = ['STAY', 'UP', 'DOWN']
        
        info_lines = [
            "🤖 AI AGENT",
            f"Action: {action_names[action]}",
            f"Confidence: {action_prob:.1%}",
            f"Score: {self.env.player_score}-{self.env.cpu_score}",
        ]
        
        y_offset = 10
        for line in info_lines:
            text_surface = self.font_small.render(line, True, self.COLOR_GREEN)
            bg_rect = text_surface.get_rect(topleft=(10, y_offset))
            bg_rect.inflate_ip(10, 5)
            pygame.draw.rect(self.screen, self.COLOR_BLACK, bg_rect)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25
    
    def get_ai_action(self):
        """Get action from AI agent"""
        features = self.feature_extractor.extract()
        features = self.feature_extractor.normalize(features)  # ← ADD THIS LINE!
        action, action_prob = self.agent.select_action(features)
        return action, action_prob
    
    def _draw_game_over_overlay(self):
        """Draw game over screen"""
        overlay = pygame.Surface((self.display_width, self.display_height))
        overlay.set_alpha(180)
        overlay.fill(self.COLOR_BLACK)
        self.screen.blit(overlay, (0, 0))
        
        if self.winner == 'AI':
            text = "AI WINS!"
            color = self.COLOR_GREEN
        elif self.winner == 'CPU':
            text = "CPU WINS!"
            color = self.COLOR_RED
        else:
            text = "DRAW!"
            color = self.COLOR_WHITE
        
        winner_surface = self.font_large.render(text, True, color)
        winner_rect = winner_surface.get_rect(
            center=(self.display_width // 2, self.display_height // 2 - 30)
        )
        self.screen.blit(winner_surface, winner_rect)
        
        score_text = f"{self.env.cpu_score} - {self.env.player_score}"
        score_surface = self.font_medium.render(score_text, True, self.COLOR_WHITE)
        score_rect = score_surface.get_rect(
            center=(self.display_width // 2, self.display_height // 2 + 20)
        )
        self.screen.blit(score_surface, score_rect)
        
        restart_surface = self.font_small.render("Press R to restart", True, self.COLOR_WHITE)
        restart_rect = restart_surface.get_rect(
            center=(self.display_width // 2, self.display_height // 2 + 60)
        )
        self.screen.blit(restart_surface, restart_rect)
    
    def run(self):
        """Main game loop"""
        print("\n▶ Starting AI demo...\n")
        
        while self.running:
            # Handle events
            self.handle_events()
            
            # Get AI action
            action, action_prob = self.get_ai_action()
            
            # Update game
            self.update_game(action)
            
            # Render
            self.render_game(action, action_prob)
            
            # Control frame rate
            self.clock.tick(self.fps)
        
        # Cleanup
        pygame.quit()
        print("\n✓ Demo complete!")
        print(f"Final Score: AI {self.env.player_score} - {self.env.cpu_score} CPU\n")


def main():
    """Entry point"""
    try:
        demo = AIDemoGUI(model_path='models/policy_gradient_v2.npz')
        demo.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted")
        pygame.quit()
        sys.exit(0)
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()