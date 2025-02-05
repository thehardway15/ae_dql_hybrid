import ale_py
import numpy as np
import gymnasium as gym
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Hyperparameters
ENV_NAME = "ALE/Breakout-v5"
GAMMA = 0.99
LEARNING_RATE = 0.00025
BUFFER_SIZE = 100000
BATCH_SIZE = 32
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
TARGET_UPDATE_FREQ = 1000
TRAIN_START = 1000
MAX_FRAMES = 500000
EVALUATION_INTERVAL = 10000
EVAL_EPISODES = 10
PRIORITY_EPSILON = 1e-6
MODEL_PATH = "dqn_breakout_model.h5"
RESULTS_CSV = "training_results.csv"
LOG_DIR = "logs/dqn_breakout/" + datetime.now().strftime("%Y%m%d-%H%M%S")

gym.register_envs(ale_py)

# Preprocess frame
def preprocess_frame(frame):
    frame = tf.image.rgb_to_grayscale(frame)
    frame = tf.image.resize(frame, [84, 84], method='nearest')
    frame = np.array(frame, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    frame = np.squeeze(frame)  # Remove any extra dimensions
    return frame

# Build the Q-Network
def build_q_network(input_shape, action_space):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(action_space, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=0.95, epsilon=0.01)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Main DQN Agent with Prioritized Replay Buffer
class DQNAgent:
    def __init__(self, input_shape, action_space):
        self.action_space = action_space
        self.epsilon = 1.0
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.priorities = deque(maxlen=BUFFER_SIZE)
        self.q_network = build_q_network(input_shape, action_space)
        self.target_network = build_q_network(input_shape, action_space)
        self.target_network.set_weights(self.q_network.get_weights())
        self.update_counter = 0
        
        # TensorBoard setup
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
        self.tensorboard.set_model(self.q_network)
        self.train_summary_writer = tf.summary.create_file_writer(LOG_DIR)
        self.episode_reward = 0
        self.episode_loss = []
        self.episode_count = 0

    def remember(self, state, action, reward, next_state, done, error):
        # Remove the batch dimension before storing
        state = np.squeeze(state, axis=0)  # Shape: (84, 84, 4)
        next_state = np.squeeze(next_state, axis=0)  # Shape: (84, 84, 4)
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(error + PRIORITY_EPSILON)
        
        # Update episode reward
        self.episode_reward += reward
        if done:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('episode_reward', self.episode_reward, step=self.episode_count)
                if self.episode_loss:  # If we have any losses
                    avg_loss = np.mean(self.episode_loss)
                    tf.summary.scalar('episode_loss', avg_loss, step=self.episode_count)
                tf.summary.scalar('epsilon', self.epsilon, step=self.episode_count)
            self.episode_count += 1
            self.episode_reward = 0
            self.episode_loss = []

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = tf.convert_to_tensor(state, dtype=tf.float32)  # Convert to tensor
        q_values = self.q_network(state, training=False)  # Use direct call instead of predict
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(range(len(self.memory)), size=BATCH_SIZE, p=probabilities)
        batch = [self.memory[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)
        # Stack states and next_states into batches
        states = tf.convert_to_tensor(states, dtype=tf.float32)  # Shape: (batch_size, 84, 84, 4)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)  # Shape: (batch_size, 84, 84, 4)
        
        q_values = self.q_network(states, training=False)  # Use direct call
        target_q_values = self.target_network(next_states, training=False)  # Use direct call

        for i in range(BATCH_SIZE):
            q_update = rewards[i]
            if not dones[i]:
                q_update += GAMMA * np.amax(target_q_values[i])
            error = abs(q_values[i][actions[i]] - q_update)
            self.priorities[indices[i]] = error + PRIORITY_EPSILON
            q_values = q_values.numpy()  # Convert to numpy for updating
            q_values[i][actions[i]] = q_update
            q_values = tf.convert_to_tensor(q_values)  # Convert back to tensor

        history = self.q_network.fit(states, q_values, batch_size=BATCH_SIZE, verbose=0)
        self.episode_loss.append(history.history['loss'][0])

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
    
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

# Main training loop
def train():
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    input_shape = (84, 84, 4)  # Shape for stacked frames
    action_space = env.action_space.n

    agent = DQNAgent(input_shape, action_space)
    total_frames = 0
    rewards_history = []
    episode = 0

    print(f"TensorBoard logs directory: {LOG_DIR}")
    print("To view training progress, run:")
    print(f"tensorboard --logdir {LOG_DIR}")

    # Initialize first state
    state, _ = env.reset()
    state = preprocess_frame(state)  # Shape: (84, 84)
    state = np.stack([state] * 4, axis=-1)  # Shape: (84, 84, 4)
    state = np.expand_dims(state, axis=0)   # Shape: (1, 84, 84, 4)

    # Initialize progress bar
    pbar = tqdm(total=MAX_FRAMES, desc=f"Training Episode {episode}")
    last_total_frames = 0

    while total_frames < MAX_FRAMES:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_frame(next_state)  # Shape: (84, 84)

        # Get the last 3 frames from current state and add the new frame
        current_frames = state[0, :, :, 1:]  # Shape: (84, 84, 3)
        next_state_stacked = np.concatenate([current_frames, np.expand_dims(next_state, axis=-1)], axis=-1)  # Shape: (84, 84, 4)
        next_state_stacked = np.expand_dims(next_state_stacked, axis=0)  # Shape: (1, 84, 84, 4)

        error = abs(reward)
        agent.remember(state, action, reward, next_state_stacked, done, error)

        state = next_state_stacked
        total_frames += 1

        # Update progress bar
        pbar.update(total_frames - last_total_frames)
        last_total_frames = total_frames
        pbar.set_postfix({
            'Epsilon': f'{agent.epsilon:.3f}',
            'Reward': f'{agent.episode_reward:.1f}'
        })

        if done:
            episode += 1
            pbar.set_description(f"Training Episode {episode}")
            state, _ = env.reset()
            state = preprocess_frame(state)  # Shape: (84, 84)
            state = np.stack([state] * 4, axis=-1)  # Shape: (84, 84, 4)
            state = np.expand_dims(state, axis=0)   # Shape: (1, 84, 84, 4)

        if total_frames > TRAIN_START:
            agent.replay()

        if total_frames % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if total_frames % EVALUATION_INTERVAL == 0:
            avg_reward = evaluate_agent(agent, env)
            rewards_history.append((total_frames, avg_reward))
            pbar.write(f"Frames: {total_frames}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # Close progress bar
    pbar.close()

    # Save the trained model
    agent.q_network.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save results to CSV
    results_df = pd.DataFrame(rewards_history, columns=["Frames", "AvgReward"])
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved to {RESULTS_CSV}")

    # Plot the training progress
    frames, avg_rewards = zip(*rewards_history)
    plt.plot(frames, avg_rewards)
    plt.xlabel("Frames")
    plt.ylabel("Average Reward")
    plt.title("Training Progress")
    plt.show()

# Evaluation function
def evaluate_agent(agent, env, episodes=EVAL_EPISODES):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)  # Shape: (84, 84)
        state = np.stack([state] * 4, axis=-1)  # Shape: (84, 84, 4)
        state = np.expand_dims(state, axis=0)   # Shape: (1, 84, 84, 4)
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)  # Use the same act method as in training
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            next_state = preprocess_frame(next_state)  # Shape: (84, 84)
            # Get the last 3 frames from current state and add the new frame
            current_frames = state[0, :, :, 1:]  # Shape: (84, 84, 3)
            next_state_stacked = np.concatenate([current_frames, np.expand_dims(next_state, axis=-1)], axis=-1)  # Shape: (84, 84, 4)
            state = np.expand_dims(next_state_stacked, axis=0)  # Shape: (1, 84, 84, 4)
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

def play_final_agent(model_path, env_name=ENV_NAME, episodes=1):
    env = gym.make(env_name, render_mode="human")
    model = tf.keras.models.load_model(model_path)
    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)  # Shape: (84, 84)
        state = np.stack([state] * 4, axis=-1)  # Shape: (84, 84, 4)
        state = np.expand_dims(state, axis=0)   # Shape: (1, 84, 84, 4)
        done = False
        total_reward = 0
        while not done:
            env.render()  # Render the environment
            action = np.argmax(model.predict(state, verbose=0)[0])
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            next_state = preprocess_frame(next_state)  # Shape: (84, 84)
            # Get the last 3 frames from current state and add the new frame
            current_frames = state[0, :, :, 1:]  # Shape: (84, 84, 3)
            next_state_stacked = np.concatenate([current_frames, np.expand_dims(next_state, axis=-1)], axis=-1)  # Shape: (84, 84, 4)
            state = np.expand_dims(next_state_stacked, axis=0)  # Shape: (1, 84, 84, 4)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    env.close()

if __name__ == "__main__":
    train()
    # play_final_agent(MODEL_PATH)
