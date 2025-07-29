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
env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=200)
state_size = env.observation_space.shape[0]  # 2
action_size = env.action_space.n  # 3

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states = torch.FloatTensor(np.array([e[0] for e in experiences])).to(device)
        actions = torch.LongTensor(np.array([e[1] for e in experiences])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences])).to(device)
        dones = torch.FloatTensor(np.array([e[4] for e in experiences]).astype(np.uint8)).unsqueeze(1).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# Agent
class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=32, gamma=0.95, lr=5e-4, tau=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.t_step = 0

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def act(self, state, epsilon=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # Enhanced reward shaping: +0.5 * velocity if moving right
        velocity = next_state[1]
        shaped_reward = reward + 0.5 * velocity if velocity > 0 else reward
        self.memory.add(state, action, shaped_reward, next_state, done)
        self.t_step += 1
        if self.t_step % 5 == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            loss = self.learn(experiences)
            self.soft_update(self.q_network, self.target_network)
            return loss
        return None

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            q_targets_next = self.target_network(next_states).max(1)[0].unsqueeze(1)
            q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        q_expected = self.q_network(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Pre-fill Replay Buffer
agent = DQNAgent(state_size, action_size, buffer_size=10000, batch_size=32, gamma=0.95, lr=5e-4, tau=1e-3)
state, _ = env.reset()
print(f"Initial State: {state}")
for _ in range(1000):  # Pre-fill with 1000 random steps
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    agent.step(state, action, reward, next_state, done)
    state = next_state
    if done:
        state, _ = env.reset()

# Training Loop
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.997

for episode in range(1, 501):
    state = np.array(state)
    total_reward = 0
    steps = 0
    for t in range(200):
        env.render()
        action = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        loss = agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1
        if terminated:
            break
        if truncated and steps == 199:
            break
    score = total_reward
    final_position = state[0] if steps > 0 else state[0]  # Log final position
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode}, Score: {score:.2f}, Epsilon: {epsilon:.2f}, Steps: {steps}, Loss: {loss:.4f}" if loss is not None else f"Episode {episode}, Score: {score:.2f}, Epsilon: {epsilon:.2f}, Steps: {steps}, Final Position: {final_position:.2f}")
    if score >= 0.0 and episode >= 100:
        print(f"✅ Solved in {episode} episodes! Score: {score:.2f}")
        break

env.close()