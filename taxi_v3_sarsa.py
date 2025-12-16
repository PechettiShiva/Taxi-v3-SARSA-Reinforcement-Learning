import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time

# Configuration parameters
config = {
    'alpha': 0.5,          # learning rate
    'gamma': 0.99,         # discount factor
    'epsilon': 1.0,        # initial exploration rate
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    'episodes': 1000,
    
    'render_interval': 100  # render every 100 episodes during training
}

# Create Taxi environment
env = gym.make("Taxi-v3", render_mode="ansi")

# Seeding for reproducibility
np.random.seed(42)
env.action_space.seed(42)

n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros([n_states, n_actions])

def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # explore
    else:
        return np.argmax(q_table[state])   # exploit

def train_agent():
    rewards_per_episode = []
    epsilon = config['epsilon']
    
    for episode in range(1, config['episodes'] + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # choose first action
        action = choose_action(state, Q, epsilon)
        
        # Decide whether to render this episode
        should_render = (episode % config['render_interval'] == 0) or (episode == config['episodes'])
        
        while not done:
            if should_render:
                print(env.render())
                time.sleep(0.05)  # small delay for visibility
            
            # Take step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Choose next action (for SARSA update)
            next_action = choose_action(next_state, Q, epsilon)
            
            # SARSA update rule
            Q[state, action] += config['alpha'] * (
                reward + config['gamma'] * Q[next_state, next_action] - Q[state, action]
            )
            
            # Move to next state and action
            state = next_state
            action = next_action
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        
        # Decay epsilon
        epsilon = max(config['epsilon_min'], epsilon * config['epsilon_decay'])
        
        if episode % 100 == 0 or episode == config['episodes']:
            print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {epsilon:.3f}")
    
    return rewards_per_episode

def test_agent():
    test_env = gym.make("Taxi-v3", render_mode="human")  # always render during test
    state, _ = test_env.reset()
    done = False
    total_reward = 0
    
    actions = {0: "South", 1: "North", 2: "East", 3: "West", 4: "Pick Up", 5: "Drop off"}
    
    print("\nTesting the trained agent...\n")
    while not done:
        action = np.argmax(Q[state])
        print(f"Action: {actions[action]}")
        state, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    print(f"\nFinal Reward: {total_reward}")
    test_env.close()

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("SARSA: Rewards over Episodes")
    plt.show()

if __name__ == "__main__":
    rewards = train_agent()
    plot_rewards(rewards)
    test_agent()



