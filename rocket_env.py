import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rocket import Rocket
import cv2


class RocketEnv(gym.Env):
    """
    A gymnasium compatible environment that wraps the Rocket simulation.
    
    The environment implements the standard gym interface for reinforcement learning.
    It wraps the Rocket class to provide standardized:
    - observation space
    - action space
    - step and reset functions
    - reward calculation
    - rendering
    """
    metadata = {'render_modes': ['human'], 'render_fps': 20}

    def __init__(self, task='hover', rocket_type='falcon', render_mode=None):
        super().__init__()
        
        # Store initialization parameters
        self.task = task
        self.rocket_type = rocket_type
        self.render_mode = render_mode
        
        # Create the rocket instance
        self.max_steps = 800
        self.rocket = Rocket(max_steps=self.max_steps, task=task, rocket_type=rocket_type)
        
        # Define scaling factors for normalization
        self.scaling_factors = {
            'x': self.rocket.world_x_max,     # Max x position
            'y': self.rocket.world_y_max,     # Max y position
            'vx': 50.0,                       # Typical max velocity
            'vy': 50.0,                       # Typical max velocity
            'theta': np.pi,                   # Full rotation
            'vtheta': 2*np.pi,               # Two rotations per second
            't': self.max_steps,              # Max timesteps
            'phi': np.pi/4                    # Max nozzle angle
        }
        
        # Define action space (9 discrete actions from Rocket's action_table)
        self.action_space = spaces.Discrete(len(self.rocket.action_table))
        
        # Define observation space (8 continuous values normalized between -1 and 1)
        # [x, y, vx, vy, theta, vtheta, t, phi]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0, 
            shape=(8,),
            dtype=np.float32
        )

    def normalize_observation(self, obs):
        """
        Normalize observation to [-1, 1] range using predefined scaling factors.
        
        Args:
            obs: Raw observation array [x, y, vx, vy, theta, vtheta, t, phi]
            
        Returns:
            Normalized observation array with all values clipped to [-1, 1]
        """
        normalized = np.array([
            obs[0] / self.scaling_factors['x'],
            obs[1] / self.scaling_factors['y'],
            obs[2] / self.scaling_factors['vx'],
            obs[3] / self.scaling_factors['vy'],
            obs[4] / self.scaling_factors['theta'],
            obs[5] / self.scaling_factors['vtheta'],
            obs[6] / self.scaling_factors['t'],
            obs[7] / self.scaling_factors['phi']
        ], dtype=np.float32)
        
        return np.clip(normalized, -1.0, 1.0)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        
        Args:
            seed: The seed for random number generation
            options: Additional options for reset (unused)
            
        Returns:
            observation: The initial observation (normalized)
            info: Additional information
        """
        super().reset(seed=seed)

        # Ensure all randomness is synchronised with seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset the rocket simulation
        observation = self.rocket.reset()
        
        # Normalize the observation
        observation = self.normalize_observation(observation)
        
        # Additional info dict
        info = {}
        
        return observation, info

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take (integer between 0 and 8)
            
        Returns:
            observation: The current observation (normalized)
            reward: The reward for the action
            terminated: Whether the episode has ended
            truncated: Whether the episode was artificially terminated
            info: Additional information
        """
        # Execute action in rocket simulation
        observation, reward, done, info = self.rocket.step(action)

        
        # Normalize the observation
        observation = self.normalize_observation(observation)
        
        # Check if max steps reached
        truncated = self.rocket.step_id >= self.max_steps
        
        # Update info dict with additional data
        info.update({
            'x': self.rocket.state['x'],
            'y': self.rocket.state['y'],
            'theta': self.rocket.state['theta'],
            'task': self.task
        })
        
        return observation, reward, done, truncated, info

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "human":
            self.rocket.render(wait_time=50)  # 20 FPS

    def close(self):
        """
        Clean up resources.
        """
        cv2.destroyAllWindows()