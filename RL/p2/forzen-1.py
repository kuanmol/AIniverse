import gymnasium as gym
import numpy as np
import time

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Q-Learning parameters
q_table = np.zeros((16, 4))  # 16 states, 4 actions
learning_rate = 0.1  # How fast to learn
discount = 0.99  # Future reward importance
exploration_rate = 0.5  # Start with moderate exploration
min_exploration = 0.01  # Minimum exploration
exploration_decay = 0.995  # Decay exploration
episodes = 1000  # Train for 1000 episodes
successes = 0  # Track goal-reaching
steps_to_goal = []  # Track steps to goal

print("Training Q-Learning agent for FrozenLake...")
print("Grid: S=Start (0), F=Frozen, H=Hole, G=Goal (15)")
print("Actions: 0=Left, 1=Down, 2=Right, 3=Up")

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    exploration_rate = max(min_exploration, exploration_rate * exploration_decay)

    for step in range(100):
        # Choose action: explore or exploit
        if np.random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        # Take action
        next_state, reward, done, truncated, info = env.step(action)
        # Update Q-table
        old_q = q_table[state, action]
        q_table[state, action] += learning_rate * (
                reward + discount * np.max(q_table[next_state]) - q_table[state, action]
        )
        total_reward += reward
        # Print details for later episodes
        if episode >= 900:  # Episodes 901-1000
            env.render()
            row, col = divmod(state, 4)
            action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
            print(f"\nEpisode {episode + 1}, Step {step + 1}:")
            print(f"  State: {state} (Row {row}, Col {col})")
            print(f"  Action: {action} ({action_names[action]})")
            print(f"  Reward: {reward} (1.0 = goal)")
            print(f"  Total Reward: {total_reward}")
            print(f"  Q-value for state {state}, action {action}: {old_q:.3f} -> {q_table[state, action]:.3f}")
            time.sleep(0.02)
        state = next_state
        if reward == 1.0:
            successes += 1
            steps_to_goal.append(step + 1)
        if done or truncated:
            break

    # Print reward for every episode
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    # Print progress every 50 episodes
    if (episode + 1) % 50 == 0:
        avg_steps = np.mean(steps_to_goal) if steps_to_goal else float('inf')
        print(
            f"\nProgress: Episode {episode + 1}, Success Rate: {successes / (episode + 1) * 100:.1f}%, Avg Steps to Goal: {avg_steps:.1f}")

# Print final Q-table
print("\nFinal Q-Table (state, actions: Left, Down, Right, Up):")
for state in range(16):
    row, col = divmod(state, 4)
    print(f"State {state} (Row {row}, Col {col}): {q_table[state].round(3)}")

env.close()
print("\nTraining done! Q-table saved.")
np.save("q_table.npy", q_table)