import gymnasium as gym
import time

# Create the FrozenLake environment with a visual window
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Run 3 episodes to see trial-and-error (like RL learning)
for episode in range(3):
    print(f"\n=== Starting Episode {episode + 1} ===")
    print("Goal: Agent learns to reach G (state 15) for +1 reward, avoid holes (H)")
    print("Grid: S=Start (0), F=Frozen, H=Hole, G=Goal (15)")
    print("Actions: 0=Left, 1=Down, 2=Right, 3=Up")

    # Start a new episode
    state = env.reset()[0]  # Initial state (0, top-left)
    total_reward = 0  # Track rewards (like RL's goal)

    # Run up to 100 steps per episode
    for step in range(100):
        env.render()  # Show agent moving on grid
        action = env.action_space.sample()  # Random action (like trial in RL)

        # Take action, get environment's response
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Print state, action, reward to connect to RL concepts
        row, col = divmod(state, 4)  # Convert state to row, column
        action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
        print(f"Step {step + 1}:")
        print(f"  State: {state} (Row {row}, Col {col})")
        print(f"  Action: {action} ({action_names[action]})")
        print(f"  Reward: {reward} (1.0 = goal reached)")
        print(f"  Total Reward: {total_reward}")

        state = next_state  # Update for next step

        # Check if episode ended
        if done or truncated:
            reason = "reached goal" if reward == 1.0 else "fell in hole" if done else "time limit reached"
            print(f"\nEpisode {episode + 1} ended after {step + 1} steps. Reason: {reason}")
            print(f"Final Total Reward: {total_reward}")
            break

        time.sleep(0.1)  # Slow down to see movement

    env.reset()  # Reset for next episode

env.close()
print("\nAll episodes finished! RL is about learning to get more rewards (e.g., reach goal).")