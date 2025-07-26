import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment
env = gym.make("BipedalWalker-v3", render_mode="human", max_episode_steps=1600)
state_size = env.observation_space.shape[0]  # 24
action_size = 8  # Discretized: 2 levels for 3 effective actions

# State normalization (refined)
state_bounds = np.array(
    [np.pi / 2, 2.0, 2.0, 2.0, np.pi / 2, 2.0, np.pi / 2, 2.0, np.pi / 2, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Discretize actions: 3 effective actions (left leg pair, right leg pair, balance), 2 levels each
action_map = [
    np.array([-1, -1, -1, -1], dtype=np.float32),  # All back
    np.array([1, 1, 1, 1], dtype=np.float32),  # All forward
    np.array([-1, -1, 1, 1], dtype=np.float32),  # Left back, right forward
    np.array([1, 1, -1, -1], dtype=np.float32),  # Left forward, right back
    np.array([0, 0, 0, 0], dtype=np.float32),  # Neutral
    np.array([-1, 1, -1, 1], dtype=np.float32),  # Alternate back/forward
    np.array([1, -1, 1, -1], dtype=np.float32),  # Alternate forward/back
    np.array([0, 0, 1, -1], dtype=np.float32),  # Balance adjust
]


# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# Agent (Double DQN)
class Agent:
    def __init__(self, state_size, action_size, buffer_size=200000, batch_size=256, gamma=0.99, lr=1e-4, tau=1e-2):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.t_steps = 0
        self.update_every = 1

    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state / state_bounds).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > epsilon:
            return torch.argmax(q_values).item()
        else:
            action_idx = np.random.choice(self.action_size)
            # Add noise to action
            action = action_map[action_idx] + np.random.normal(0, 0.1, size=4).astype(np.float32)
            action = np.clip(action, -1, 1)
            return action_idx

    def step(self, state, action, reward, next_state, done):
        state = state / state_bounds
        next_state = next_state / state_bounds
        reward = np.clip(reward, -1.0, 1.0)
        self.memory.add(state, action, reward, next_state, done)
        self.t_steps = (self.t_steps + 1) % self.update_every
        if self.t_steps == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            loss = self.learn(experiences)
            return loss
        return None

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # Double DQN
        next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.item()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


# Pre-fill replay buffer
agent = Agent(state_size, action_size, lr=1e-4, tau=1e-2)
state, _ = env.reset()
for _ in range(5000):
    action_idx = np.random.choice(action_size)
    action = action_map[action_idx]
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    agent.memory.add(state / state_bounds, action_idx, np.clip(reward, -1.0, 1.0), next_state / state_bounds, done)
    state = next_state
    if done:
        state, _ = env.reset()

# Training Loop
n_episodes = 2000
max_t = 1600
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
scores = []
scores_window = deque(maxlen=100)
episode_block_scores = []

for episode in range(1, n_episodes + 1):
    state, _ = env.reset()
    total_reward = 0
    epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** (episode - 1)))
    loss = None

    for t in range(max_t):
        action_idx = agent.act(state, epsilon)
        action = action_map[action_idx]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        loss = agent.step(state, action_idx, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            if episode >= 1900:  # Detailed logs for Episodes 1900-2000
                print(f"\nEpisode {episode}, Step {t + 1}:")
                print(f"  State: {state}")
                print(f"  Action: {action_idx} ({action})")
                print(f"  Reward: {reward:.3f}")
                print(f"  Total Reward: {total_reward:.1f}")
            break

    scores.append(total_reward)
    scores_window.append(total_reward)
    episode_block_scores.append(total_reward)

    print(
        f"Episode {episode}, "
        f"Score: {total_reward:.1f}, "
        f"Avg100: {np.mean(scores_window):.1f}, "
        f"Epsilon: {epsilon:.2f}, "
        f"Loss: {loss:.4f}" if loss is not None else f"Loss: N/A"
    )

    # Print average and max score for each 100-episode block
    if episode % 100 == 0:
        block_avg = np.mean(episode_block_scores[-100:])
        block_max = np.max(episode_block_scores[-100:])
        print(f"Episodes {episode - 99}–{episode}: Avg Score: {block_avg:.1f}, Max Score: {block_max:.1f}")

    if np.mean(scores_window) >= 300 and episode >= 100:
        print(f"✅ Solved in {episode} episodes! Avg score: {np.mean(scores_window):.1f}")
        torch.save(agent.qnetwork_local.state_dict(), 'bipedalwalker_dqn.pth')
        break

env.close()
