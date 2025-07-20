import gymnasium as gym
import numpy as np
import time

# Create CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Discretize state space
state_bins = [
    np.linspace(-2.4, 2.4, 20),  # Cart position
    np.linspace(-3.0, 3.0, 20),  # Cart velocity
    np.linspace(-0.21, 0.21, 20),  # Pole angle (~12 degrees)
    np.linspace(-3.0, 3.0, 20)  # Angular velocity
]


def discretize_state(state):
    state_indices = []
    for i in range(4):
        state_indices.append(np.digitize(state[i], state_bins[i]) - 1)
    return tuple(state_indices)


# Q-Learning parameters
q_table = np.zeros([30] * 4 + [env.action_space.n])  # 20^4 states, 2 actions
learning_rate = 0.1
discount = 0.99
exploration_rate = 0.8
min_exploration = 0.01
exploration_decay = 0.99
episodes = 600
successes = 0  # Episodes with >=100 steps
steps_history = []  # Track steps per episode

print("Training Q-Learning agent for CartPole...")
print("Actions: 0=Left, 1=Right")
print("State: [Cart Position, Cart Velocity, Pole Angle, Angular Velocity]")

for episode in range(episodes):
    state = env.reset()[0]
    state = discretize_state(state)
    total_reward = 0
    exploration_rate = max(min_exploration, exploration_rate * exploration_decay)

    for step in range(200):
        # Choose action
        if np.random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        # Take action
        next_state, reward, done, truncated, info = env.step(action)
        next_state = discretize_state(next_state)
        # Update Q-table
        old_q = q_table[state][action]
        q_table[state][action] += learning_rate * (
                reward + discount * np.max(q_table[next_state]) - q_table[state][action]
        )
        total_reward += reward
        # Print details for later episodes
        if episode >= 1900:  # Episodes 1901-2000
            env.render()
            print(f"\nEpisode {episode + 1}, Step {step + 1}:")
            print(f"  State: {list(state)}")
            print(f"  Action: {action} ({['Left', 'Right'][action]})")
            print(f"  Reward: {reward}")
            print(f"  Total Reward: {total_reward}")
            print(f"  Q-value for state, action {action}: {old_q:.3f} -> {q_table[state][action]:.3f}")
            time.sleep(0.02)
        state = next_state
        if done or truncated:
            break

    # Track steps and success
    steps_history.append(step + 1)
    if step + 1 >= 100:
        successes += 1
    # Print reward and steps for every episode
    print(f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {step + 1}")

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        avg_steps = np.mean(steps_history[-100:])
        avg_reward = np.mean(total_reward)
        print(f"\nProgress: Episode {episode + 1}, Success Rate (>=100 steps): {successes / (episode + 1) * 100:.1f}%, Avg Steps: {avg_steps:.1f}")
        print(avg_reward)

env.close()
print("\nTraining done! Q-table saved.")
np.save("cartpole_q_table.npy", q_table)