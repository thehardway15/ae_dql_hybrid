import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import namedtuple, deque

Config = namedtuple('Config', ['epsilon_start', 'epsilon_final', 'epsilon_decay', 
                               'target_update_frequency', 'learning_starts', 'batch_size', 
                               'gamma', 'update_frequency', 'replay_buffer_capacity', 'env_name', 'learning_rate' ])

ConfigCartPole = Config(epsilon_start=1.0, epsilon_final=0.02, epsilon_decay=5_000,
                        target_update_frequency=500, learning_starts=1_000, batch_size=128,
                        gamma=0.99, update_frequency=4, replay_buffer_capacity=60_000, env_name='CartPole-v1', learning_rate=0.001)

class ReplayBuffer:
    def __init__(self, capacity: int, device: str):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(dones).to(self.device),
        )
    
    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, config: Config, env, model: nn.Module, device: str, optimizer: optim.Optimizer):
        self.config = config
        self.env = env
        self.model = model.to(device)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity, device)
        self.optimizer = optimizer
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())
        self.device = device
        self.target_model.to(device)
        self.total_frames = 0
        self.epsilon = config.epsilon_start
        self.loss_list = []

    def _compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_value = next_q_values.max(dim=1).values

        expected_q_value = rewards + self.config.gamma * next_q_value * (1. - dones)

        loss = nn.MSELoss()(q_value, expected_q_value.detach())
        return loss

    def _update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path: str):
        torch.save(self.target_model.state_dict(), path)

    def load_model(self, path: str):
        self.target_model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path))

    def _action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.model(state).argmax().item()
            
    def epsilon_update(self):
        self.epsilon = max(self.config.epsilon_final, self.config.epsilon_start - self.total_frames / self.config.epsilon_decay)

    def train_step(self, state):
        self.total_frames += 1
        self.epsilon_update()

        action = self._action(state)
        next_state, reward, _ = self.env.step(action)
        done = self.env.done

        next_state_copy = next_state.copy()

        self.replay_buffer.push(state, action, reward, next_state_copy, done)

        if self.replay_buffer.size() > self.config.learning_starts and self.total_frames % self.config.update_frequency == 0:
            batch = self.replay_buffer.sample(self.config.batch_size)
            loss = self._compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_list.append(loss.item())

        if self.total_frames % self.config.target_update_frequency == 0:
            self._update_target_network()

        return next_state
    
    def play(self):
        self.target_model.eval()
        self.epsilon = 0.0

        state, _ = self.env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = self._action(state)
            next_state, reward, _ = self.env.step(action)
            done = self.env.done
            state = next_state
            episode_reward += reward
            self.env.render()
        
        print(f"Episode Reward: {episode_reward}")
        self.env.close()

