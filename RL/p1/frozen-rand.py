import gymnasium as gym
import time

# Create the FrozenLake environment with a visual window
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Start a new episode
state = env.reset()[0]  # Initial state (0, top-left corner)
total_reward = 0  # Track total reward

print("Starting FrozenLake episode...")
print("Grid: S=Start (0), F=Frozen, H=Hole, G=Goal (15)")
print("State: Number (0-15) for grid position (row * 4 + col)")
print("Actions: 0=Left, 1=Down, 2=Right, 3=Up")

# Run for up to 100 steps
for step in range(100):
    env.render()  # Show the agent moving on the grid
    action = env.action_space.sample()  # Random action: 0 (left), 1 (down), 2 (right), 3 (up)

    # Take action, get new state, reward, and whether episode ended
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward

    # Print detailed info
    row, col = divmod(state, 4)  # Convert state number to row, column
    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    print(f"\nStep {step + 1}:")
    print(f"  State: {state} (Row {row}, Col {col})")
    print(f"  Action: {action} ({action_names[action]})")
    print(f"  Reward: {reward} (1.0 means reached goal)")
    print(f"  Total Reward: {total_reward}")

    state = next_state  # Update state

    # Check why episode ended
    if done or truncated:
        reason = "reached goal" if reward == 1.0 else "fell in hole" if done else "time limit reached"
        print(f"\nEpisode ended after {step + 1} steps. Reason: {reason}")
        print(f"Final Total Reward: {total_reward}")
        break

    time.sleep(0.1)  # Slow down to see the window

env.close()
print("Episode finished!")