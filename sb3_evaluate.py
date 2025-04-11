import numpy as np
from stable_baselines3 import PPO
from rocket_env import RocketEnv

def evaluate_model(model_path, task='landing', rocket_type='starship', n_eval_episodes=10):
    """
    Evaluate a trained model
    """
    # Create environment
    env = RocketEnv(task=task, rocket_type=rocket_type, render_mode='human')
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Run evaluation
    episode_rewards = []
    successful_landings = 0
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            
            if info.get('already_crash', False):
                print(f"Episode {episode + 1}: Crash!")
                break
                
            if info.get('already_landing', False):
                print(f"Episode {episode + 1}: Successful landing!")
                successful_landings += 1
                break
            
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")
    
    env.close()
    
    # Print statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = (successful_landings / n_eval_episodes) * 100
    
    print(f"\nResults after {n_eval_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Successful landings: {successful_landings}/{n_eval_episodes}")
    
    return mean_reward, std_reward, success_rate

if __name__ == '__main__':
    # Configuration
    task = 'landing'
    rocket_type = 'starship'
    model_path = f"./sb3_checkpoints/{task}_{rocket_type}/best_model/best_model"
    
    # Evaluate the model
    mean_reward, std_reward, success_rate = evaluate_model(
        model_path=model_path,
        task=task,
        rocket_type=rocket_type,
        n_eval_episodes=10
    ) 