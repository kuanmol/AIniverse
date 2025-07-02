import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym

env = gym.make('CartPole-v1')
state_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

learning_rate = 0.01
gamma = 0.99


def build_policy_model():
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=(state_shape,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(num_actions, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate))
    return model


policy_model = build_policy_model()


def choose_action(state):
    state = np.array(state).reshape([1, state_shape])
    probabilities = policy_model.predict(state)
    return np.random.choice(num_actions, p=probabilities[0])


def discount_rewards(rewards):
    discounted = np.zeros_like(rewards)
    cumulative = 0
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[i]
        discounted[i] = cumulative
    return discounted - np.mean(discounted)


def train_on_episode(state, action, reward):
    discounted_rewards = discount_rewards(reward)
    with tf.GradientTape() as tape:
        action_probabilities = policy_model(tf.convert_to_tensor(state, dtype=tf.float32), training=True)
        action_index = tf.stack([tf.range(len(action)), action], axis=1)
        selected_action_probabilities = tf.gather_nd(action_probabilities, action_index)
        loss = -tf.reduce_sum(tf.math.log(selected_action_probabilities) * discounted_rewards)
    gradients = tape.gradient(loss, policy_model.trainable_variables)
    policy_model.optimizer.apply_gradients(zip(gradients, policy_model.trainable_variables))


num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_states, episode_actions, episode_rewards = [], [], []
    while True:
        action = choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        state = next_state
        if done:
            episode_states=np.stack(episode_states)
            train_on_episode(episode_states, np.array(episode_actions), np.array(episode_rewards))
            print(f"Episode: {episode+1}, Total Reward: {sum(episode_rewards)}")
            break

