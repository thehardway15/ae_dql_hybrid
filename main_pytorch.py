import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

gym.register_envs(ale_py)

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
MODEL_PATH = "dqn_breakout_pytorch.pth"
LOG_DIR = "logs/dqn_breakout_pytorch/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Set device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def preprocess_frame(frame):
    # Convert to grayscale and resize
    frame = np.array(frame).mean(axis=2).astype(np.float32)  # Convert to grayscale
    # Resize to exactly 84x84
    frame = frame[::2, ::2]  # Downsample by factor of 2
    # Further downsample to exactly 84x84 if needed
    if frame.shape != (84, 84):
        new_frame = np.zeros((84, 84), dtype=np.float32)
        new_frame[:frame.shape[0], :frame.shape[1]] = frame[:84, :84]
        frame = new_frame
    frame = frame / 255.0  # Normalize
    return frame

# Neural Network Model
class DQN(nn.Module):
    def __init__(self, input_shape, action_space):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # (4, 84, 84) -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (32, 20, 20) -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (64, 9, 9) -> (64, 7, 7)
            nn.ReLU()
        )
        
        # Calculate the size of flattened features
        x = torch.zeros(1, *input_shape)
        conv_out = self.conv(x)
        self.fc_input_dim = int(np.prod(conv_out.shape[1:]))  # Multiply all dimensions except batch
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )
        
        print(f"Conv output shape: {conv_out.shape}")
        print(f"Flattened dim: {self.fc_input_dim}")

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch
        return self.fc(x)

# Replay Buffer with Prioritization
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.eps = 1e-6

    def store(self, transition, error):
        self.buffer.append(transition)
        self.priorities.append(error + self.eps)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(dones).to(DEVICE),
            indices
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + self.eps

    def size(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, input_shape, action_space):
        self.action_space = action_space
        self.epsilon = 1.0
        self.q_network = DQN(input_shape, action_space).to(DEVICE)
        self.target_network = DQN(input_shape, action_space).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=LEARNING_RATE, alpha=0.95, eps=0.01)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        
        # TensorBoard setup
        self.writer = SummaryWriter(LOG_DIR)
        self.total_frames = 0
        self.episode_reward = 0
        self.episode_count = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_space)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return
        
        states, actions, rewards, next_states, dones, indices = self.replay_buffer.sample(batch_size)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + (1 - dones) * GAMMA * next_q_values
        
        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss and update priorities
        errors = torch.abs(q_values - targets).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, errors)
        
        # Update network
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Log metrics
        self.writer.add_scalar('Loss/train', loss.item(), self.total_frames)
        
        # Update epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def log_episode(self, reward):
        self.episode_reward = reward
        self.writer.add_scalar('Reward/train', reward, self.episode_count)
        self.writer.add_scalar('Epsilon', self.epsilon, self.episode_count)
        self.episode_count += 1

# Training function
def train():
    env = gym.make(ENV_NAME)
    input_shape = (4, 84, 84)  # (channels, height, width)
    action_space = env.action_space.n
    agent = DQNAgent(input_shape, action_space)
    
    print(f"Training on device: {DEVICE}")
    print(f"TensorBoard logs: {LOG_DIR}")
    
    state, _ = env.reset()
    state = preprocess_frame(state)
    state = np.stack([state] * 4, axis=0)  # Stack 4 frames
    
    pbar = tqdm(total=MAX_FRAMES, desc="Training")
    rewards_history = []
    episode_reward = 0
    
    while agent.total_frames < MAX_FRAMES:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = preprocess_frame(next_state)
        next_state = np.concatenate([state[1:], np.expand_dims(next_state, 0)], axis=0)
        
        error = abs(reward)
        agent.replay_buffer.store((state, action, reward, next_state, done), error)
        
        state = next_state
        episode_reward += reward
        agent.total_frames += 1
        pbar.update(1)
        
        if agent.total_frames > TRAIN_START:
            agent.update(BATCH_SIZE)
        
        if agent.total_frames % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        if done:
            agent.log_episode(episode_reward)
            rewards_history.append(episode_reward)
            episode_reward = 0
            state, _ = env.reset()
            state = preprocess_frame(state)
            state = np.stack([state] * 4, axis=0)
        
        if agent.total_frames % EVALUATION_INTERVAL == 0:
            avg_reward = evaluate_agent(agent, env)
            pbar.write(f"Frames: {agent.total_frames}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
            agent.writer.add_scalar('Reward/eval', avg_reward, agent.total_frames)
    
    pbar.close()
    torch.save(agent.q_network.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

def evaluate_agent(agent, env, episodes=EVAL_EPISODES):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)
        state = np.stack([state] * 4, axis=0)
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_frame(next_state)
            state = np.concatenate([state[1:], np.expand_dims(next_state, 0)], axis=0)
            total_reward += reward
            
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

def play():
    env = gym.make(ENV_NAME, render_mode="human")
    input_shape = (4, 84, 84)
    action_space = env.action_space.n
    
    model = DQN(input_shape, action_space).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    state, _ = env.reset()
    state = preprocess_frame(state)
    state = np.stack([state] * 4, axis=0)
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action = model(state_tensor).argmax().item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = preprocess_frame(next_state)
        state = np.concatenate([state[1:], np.expand_dims(next_state, 0)], axis=0)
        total_reward += reward
    
    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    train()
    # Uncomment to watch the trained model play
    # play()
