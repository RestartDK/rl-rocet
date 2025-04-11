import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from rocket_env import RocketEnv

def make_env(task='landing', rocket_type='starship', render_mode=None):
    """Create a wrapped, monitored environment."""
    def _init():
        env = RocketEnv(task=task, rocket_type=rocket_type, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':
    # Configuration
    task = 'landing'
    rocket_type = 'starship'
    total_timesteps = 10_000_000  # Keep increased training time
    save_freq = 10000
    
    # Create log directory
    log_dir = f"./sb3_logs/{task}_{rocket_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = f"./sb3_checkpoints/{task}_{rocket_type}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create and wrap the environment
    env = DummyVecEnv([make_env(task=task, rocket_type=rocket_type)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(task=task, rocket_type=rocket_type, render_mode='human')])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="rocket_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{checkpoint_dir}/best_model",
        log_path=f"{checkpoint_dir}/eval_logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Initialize the model with original hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,  # Original learning rate
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Original gamma
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Original simpler architecture
        )
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save the final model
    model.save(f"{checkpoint_dir}/final_model")
    
    # Close environments
    env.close()
    eval_env.close() 