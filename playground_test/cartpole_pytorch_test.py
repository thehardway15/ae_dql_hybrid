import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
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
MODEL_PATH = "dqn_cartpole_pytorch.pth"

# Set device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# Neural Network Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.SELU(),
            nn.Linear(32, 32),
            nn.SELU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

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
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(dones).to(DEVICE)
        )
    
    def size(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.epsilon = 1.0
        self.q_network = DQN(input_dim, output_dim).to(DEVICE)
        self.target_network = DQN(input_dim, output_dim).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.output_dim = output_dim
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.output_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + GAMMA * next_q_values * (1 - dones)
        
        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss and update
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Training function
def train():
    env = gym.make(ENV_NAME)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = DQNAgent(input_dim, output_dim)
    
    rewards_history = []
    
    # Initialize progress bar
    pbar = tqdm(range(MAX_EPISODES), desc="Training")
    
    for episode in pbar:
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.store((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if agent.replay_buffer.size() > BATCH_SIZE:
                agent.update(BATCH_SIZE)
        
        rewards_history.append(total_reward)
        
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # Update progress bar
        pbar.set_postfix({
            'Reward': f'{total_reward:.1f}',
            'Epsilon': f'{agent.epsilon:.3f}'
        })
        
        if total_reward >= 200:
            print("\nSolved! Training complete.")
            break
    
    # Save the model
    torch.save(agent.q_network.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Plot training progress
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

# Function to play with trained model
def play():
    env = gym.make(ENV_NAME, render_mode="human")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    model = DQN(input_dim, output_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action = model(state_tensor).argmax().item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    # train()
    # Uncomment to watch the trained model play
    play() 