import gymnasium as gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Hyperparameters
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LEARNING_RATE = 0.001
BUFFER_SIZE = 2000
BATCH_SIZE = 64
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TARGET_UPDATE_FREQ = 10
MAX_EPISODES = 500
LOG_DIR = "logs/dqn_cartpole"
MODEL_PATH = "dqn_cartpole_model.h5"

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def size(self):
        return len(self.buffer)

# Neural network model
def build_q_network(input_shape, action_space):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='selu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='selu'),
        tf.keras.layers.Dense(action_space, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# DQN Agent
class DQNAgent:
    def __init__(self, input_shape, action_space):
        self.action_space = action_space
        self.epsilon = 1.0
        self.q_network = build_q_network(input_shape, action_space)
        self.target_network = build_q_network(input_shape, action_space)
        self.update_target_network()
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.q_network.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Predict Q-values
        q_values_next = self.target_network.predict(next_states, verbose=0)
        q_values_next = np.max(q_values_next, axis=1)
        targets = rewards + (1 - dones) * GAMMA * q_values_next

        # Update Q-values for actions taken
        q_values = self.q_network.predict(states, verbose=0)
        for i, action in enumerate(actions):
            q_values[i][action] = targets[i]

        # Train the Q-network
        self.q_network.fit(states, q_values, batch_size=batch_size, verbose=0)

        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

# Training function
def train():
    env = gym.make(ENV_NAME)
    input_shape = (env.observation_space.shape[0],)
    action_space = env.action_space.n
    agent = DQNAgent(input_shape, action_space)

    rewards_history = []
    
    # Create TensorBoard writer
    summary_writer = tf.summary.create_file_writer(LOG_DIR)

    # Initialize progress bar
    pbar = tqdm(range(MAX_EPISODES), desc="Training")

    for episode in pbar:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            agent.replay_buffer.store((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if agent.replay_buffer.size() > BATCH_SIZE:
                agent.update(BATCH_SIZE)

        rewards_history.append(total_reward)

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Update progress bar with metrics
        pbar.set_postfix({
            'Reward': f'{total_reward:.1f}',
            'Epsilon': f'{agent.epsilon:.3f}'
        })

        # Log metrics to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('reward', total_reward, step=episode)
            tf.summary.scalar('epsilon', agent.epsilon, step=episode)
            summary_writer.flush()

        if total_reward >= 200:
            print("\nSolved! Training complete.")
            break

    # Save the model
    agent.q_network.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Plot training progress
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

# Function to play with the trained model
def play():
    env = gym.make(ENV_NAME, render_mode="human")
    model = tf.keras.models.load_model(MODEL_PATH)
    state, _ = env.reset()  # Unpack the tuple
    state = np.array(state, dtype=np.float32)
    total_reward = 0
    done = False

    while not done:
        q_values = model.predict(state[np.newaxis], verbose=0)
        action = np.argmax(q_values[0])
        next_state, reward, terminated, truncated, _ = env.step(action)  # Updated step return values
        done = terminated or truncated  # Combine termination conditions
        state = np.array(next_state, dtype=np.float32)
        total_reward += reward

    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    # train()
    # Uncomment the following line to watch the trained model play
    play()
