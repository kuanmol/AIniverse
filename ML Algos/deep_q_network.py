import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from gym.wrappers import RecordVideo


# --- DQN Definition ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


# --- Hyperparameters ---
env_name = 'CartPole-v1'
learning_rate = 0.001
gamma = 0.99
buffer_size = 10000
batch_size = 32
epsilon = 0.1
target_update_rate = 100
solve_threshold = 5  # average over 100

# --- Environment & Networks ---
train_env = gym.make(env_name)
input_size = train_env.observation_space.shape[0]
output_size = train_env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

replay_buffer = []


# --- Training Function ---
def train(num_episodes):
    step_count = 0
    reward_history = []

    for episode in range(num_episodes):
        state, _ = train_env.reset()
        state = np.array(state)
        done = False
        total_reward = 0

        while not done:
            # ε-greedy
            if random.random() < epsilon:
                action = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.tensor(state, dtype=torch.float32, device=device)
                    action = policy_net(st).argmax().item()

            next_state, reward, done, truncated, _ = train_env.step(action)
            done = done or truncated
            next_state = np.array(next_state)
            total_reward += reward

            # store
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)
            state = next_state

            # learn
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device)

                q_curr = policy_net(states).gather(1, actions.unsqueeze(1))
                q_next = target_net(next_states).max(1)[0].detach()
                q_target = rewards + gamma * q_next * (1 - dones)

                loss = criterion(q_curr, q_target.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_count += 1
                if step_count % target_update_rate == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        reward_history.append(total_reward)
        if len(reward_history) >= 20:
            avg100 = np.mean(reward_history[-100:])
            print(f"Episode {episode}, Reward {total_reward}, Avg100 {avg100:.2f}")
            if avg100 >= solve_threshold:
                print("Solved! Stopping training.")
                torch.save(policy_net.state_dict(), "dqn_cartpole_solved.pth")
                break
        else:
            print(f"Episode {episode}, Reward {total_reward}")


# --- Evaluation (with optional rendering) ---
def evaluate(env, model, episodes=5, render=False):
    model.eval()
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state)
        done = False
        ep_reward = 0
        while not done:
            if render:
                env.render()
                time.sleep(0.02)
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32, device=device)
                action = model(st).argmax().item()
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            state = np.array(next_state)
            ep_reward += reward
        rewards.append(ep_reward)
        print(f"Episode {ep + 1} Reward {ep_reward}")
    avg = np.mean(rewards)
    print(f"Average Reward: {avg}")


# --- Run Training ---
train(num_episodes=1000)

# --- Record Video of Evaluation ---
video_env = RecordVideo(
    gym.make(env_name, render_mode="rgb_array"),
    video_folder="videos/",
    episode_trigger=lambda x: True
)
evaluate(video_env, policy_net, episodes=5, render=False)
video_env.close()
print("Videos saved to ./videos/")
