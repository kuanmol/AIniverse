import gymnasium as gym
import time  # For slowing down to see the window

# Create the CartPole environment with a visual window
env = gym.make("CartPole-v1", render_mode="human")


# Start a new episode
state = env.reset()[0]  # Initial state: [cart position, cart velocity, pole angle, pole angular velocity]
total_reward = 0  # Track total reward (steps pole stays balanced)

print("Starting CartPole episode...")
print("State format: [Cart Position, Cart Velocity, Pole Angle (radians), Pole Angular Velocity]")

# Run for up to 200 steps
for step in range(50):
    env.render()  # Show the cart and pole moving in the window
    action = env.action_space.sample()  # Random action: 0 (left) or 1 (right)

    # Take action, get new state, reward, and whether episode ended
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward

    # Print detailed state info
    print(f"\nStep {step + 1}:")
    print(f"  State: {state}")
    print(f"    Cart Position: {state[0]:.3f} (0 is center, negative = left, positive = right)")
    print(f"    Cart Velocity: {state[1]:.3f} (negative = moving left, positive = moving right)")
    print(f"    Pole Angle: {state[2]:.3f} radians (~{state[2] * 57.3:.1f} degrees, 0 is vertical)")
    print(f"    Pole Angular Velocity: {state[3]:.3f} (how fast pole is tilting)")
    print(f"  Action: {action} ({'left' if action == 0 else 'right'})")
    print(f"  Reward: {reward} (1.0 means pole is still balanced)")
    print(f"  Total Reward: {total_reward} (steps pole stayed up)")

    state = next_state  # Update state for next step

    # Check why episode ended
    if done or truncated:
        reason = "pole fell (angle too large or cart too far)" if done else "time limit reached"
        print(f"\nEpisode ended after {step + 1} steps. Reason: {reason}")
        print(f"Final Total Reward: {total_reward}")
        break

    # Slow down to see the window (optional, comment out if too slow)
    time.sleep(0.10)

env.close()  # Close the window
print("Episode finished!")