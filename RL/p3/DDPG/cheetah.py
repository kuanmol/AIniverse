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

# Frame-skipping wrapper for 2x faster rendering
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            next_state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return next_state, total_reward / 100, terminated, truncated, info  # Normalize reward

# Create environment
env = gym.make("HalfCheetah-v4", render_mode="human", max_episode_steps=1000)
env = FrameSkip(env, skip=2)  # 2x faster rendering
state_size = env.observation_space.shape[0]  # 17
action_size = env.action_space.shape[0]     # 6
action_low = torch.tensor(env.action_space.low, device=device)
action_high = torch.tensor(env.action_space.high, device=device)

# State normalization
class StateNormalizer:
    def __init__(self):
        self.mean = np.zeros(state_size)
        self.std = np.ones(state_size)
        self.n = 0

    def update(self, state):
        self.n += 1
        delta = state - self.mean
        self.mean += delta / self.n
        delta2 = state - self.mean
        self.std = np.sqrt((self.n - 1) / self.n * self.std**2 + delta * delta2 / self.n)
        self.std = np.clip(self.std, 1e-6, np.inf)

    def normalize(self, state):
        return (state - self.mean) / self.std

# Actor Network (Biased toward forward motion)
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Outputs in [-1, 1]
        # Bias for forward motion: +0.1 for hind legs (0-2), 0.0 for front leg (3)
        motion_bias = torch.zeros(action_size, device=state.device)  # Match action_size
        motion_bias[:3] = 0.1  # Hind legs
        motion_bias[3] = 0.0  # Front leg (neutralized from 0.05)
        action = action + motion_bias
        return action

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Ornstein-Uhlenbeck Noise with device support
class OUNoise:
    def __init__(self, action_size, mu=0.0, theta=0.15, sigma=0.15):
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
        self.decay_rate = 0.002  # Faster decay

    def reset(self):
        self.state = np.copy(self.mu)
        self.sigma = max(0.05, self.sigma - self.decay_rate)  # Decay to 0.05

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return dx  # Return noise increment

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
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# DDPG Agent with adjusted flipping penalty
class DDPGAgent:
    def __init__(self, state_size, action_size, buffer_size=1000000, batch_size=256, gamma=0.99, tau=5e-3, lr_actor=1e-4, lr_critic=1e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.noise = OUNoise(action_size)
        self.normalizer = StateNormalizer()
        self.t_steps = 0
        self.update_every = 1

    def act(self, state, add_noise=True):
        state = torch.from_numpy(self.normalizer.normalize(state)).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state)
        self.actor_local.train()
        if add_noise:
            noise = torch.tensor(self.noise.sample(), device=device, dtype=torch.float32)
            action = action + noise
        return torch.clamp(action, action_low, action_high).cpu().numpy().squeeze()

    def step(self, state, action, reward, next_state, done):
        self.normalizer.update(state)
        state = self.normalizer.normalize(state)
        next_state = self.normalizer.normalize(next_state)
        # Penalty for flipping (lowered threshold to 0.3)
        flip_penalty = -1.0 if abs(state[2]) > 0.3 else 0.0  # Adjusted from 0.5
        adjusted_reward = reward + flip_penalty
        self.memory.add(state, action, adjusted_reward, next_state, done)
        self.t_steps = (self.t_steps + 1) % self.update_every
        if self.t_steps == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            loss_critic = self.learn(experiences)
            return loss_critic
        return None

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # Update critic
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)  # Increased from 0.5
        self.critic_optimizer.step()
        # Update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update target networks
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)
        return critic_loss.item()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# Pre-fill replay buffer
agent = DDPGAgent(state_size, action_size, lr_critic=1e-4)
state, _ = env.reset()
for _ in range(5000):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    agent.memory.add(state, action, reward / 100, next_state, done)
    state = next_state
    if done:
        state, _ = env.reset()

# Training Loop
n_episodes = 1000
max_t = 1000
scores = []
scores_window = deque(maxlen=100)
episode_block_scores = []

for episode in range(1, n_episodes + 1):
    state, _ = env.reset()
    agent.noise.reset()
    total_reward = 0
    loss = None

    for t in range(max_t):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        loss = agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            if episode >= 900:  # Detailed logs for Episodes 900-1000
                print(f"\nEpisode {episode}, Step {t + 1}:")
                print(f"  State: {state}")
                print(f"  Action: {action}")
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
        f"Loss: {loss:.4f}" if loss is not None else f"Loss: N/A"
    )

    if episode % 100 == 0:
        block_avg = np.mean(episode_block_scores[-100:])
        block_max = np.max(episode_block_scores[-100:])
        print(f"Episodes {episode-99}–{episode}: Avg Score: {block_avg:.1f}, Max Score: {block_max:.1f}")

    if np.mean(scores_window) >= 30 and episode >= 100:  # Scaled target (3000 / 100)
        print(f"✅ Solved in {episode} episodes! Avg score: {np.mean(scores_window):.1f}")
        torch.save(agent.actor_local.state_dict(), 'halfcheetah_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'halfcheetah_critic.pth')
        break

env.close()