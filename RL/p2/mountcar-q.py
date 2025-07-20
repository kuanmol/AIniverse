import gymnasium as gym
import numpy as np
import time

# 1) Create the MountainCar environment
env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=1000)

# 2) Discretize the continuous state into bins
n_bins = 120
pos_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], n_bins)
vel_bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], n_bins)
state_bins = [pos_bins, vel_bins]

def discretize_state(obs: np.ndarray) -> tuple[int, ...]:
    """
    Turn a 2‑element observation [position, velocity]
    into a tuple of bin indices, each in [0..n_bins-1].
    """
    return tuple(
        np.digitize(o, b) - 1
        for o, b in zip(obs, state_bins)
    )

# 3) Q‑table initialization
action_size = env.action_space.n   # 3 actions: push left, no push, push right
q_table = np.zeros((n_bins, n_bins, action_size), dtype=np.float64)

# 4) Hyperparameters
learning_rate    = 0.1
discount_factor  = 0.99
epsilon          = 0.9       # initial exploration rate
min_epsilon      = 0.01
epsilon_decay    = 0.9995

n_episodes       = 5000
max_steps        = 1000

# 5) Training loop
successes = 0
for ep in range(1, n_episodes + 1):
    # correct unpacking of reset()
    obs, _ = env.reset()
    state = discretize_state(obs)

    # decay ε
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    total_reward = 0.0

    for t in range(1, max_steps + 1):
        # ε‑greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))

        # correct unpacking of step()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_obs)

        # Q‑learning update
        best_next_q = np.max(q_table[next_state])
        td_target = reward + discount_factor * best_next_q
        td_error  = td_target - q_table[state + (action,)]
        q_table[state + (action,)] += learning_rate * td_error

        state = next_state
        total_reward += reward

        if terminated or truncated:
            # count success only if goal reached
            if terminated and next_obs[0] >= env.goal_position:
                successes += 1
            break

    # logging every 100 episodes
    if ep % 100 == 0:
        avg_success_rate = successes / ep * 100
        print(f"Episode {ep:4d} | ε={epsilon:.3f} | Success Rate: {avg_success_rate:5.1f}% | Last reward: {total_reward:.1f}")

# 6) Save Q‑table
np.save("mountaincar_q_table.npy", q_table)
print("\nTraining complete. Now running 3 evaluation episodes...")


