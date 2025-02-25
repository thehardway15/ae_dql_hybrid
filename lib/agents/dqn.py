import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
from lib.metrics import Metrics
from lib.config import Config
from lib.utils import ReplayBuffer


class DQNAgent:
    def __init__(self, config: Config, env, model: nn.Module, device: str, optimizer: optim.Optimizer = None, checkpoints: int = 1000, path: str = None):
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
        self.history = Metrics()
        self.checkpoints = checkpoints
        self.path = path

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
        torch.save(self.target_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def save_history(self, path: str):
        self.history.save(path)
        self.history.summary(path, 
                          plots=['frames_per_episode', 'reward_per_episode', 'time_per_episode', 'loss', 'memory_usage', 'replay_buffer_size'],
                          additional_stats=['frames_last / total_time_last'])

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
            self.history.add('loss', loss.item())

        if self.total_frames % self.config.target_update_frequency == 0:
            self._update_target_network()

        return next_state

    def train(self, epochs):
        start_time = time.time()
        epoch = 0
        progress_bar = tqdm(total=epochs, desc='Training Progress')
        episode = 0

        while epoch < epochs:
            start_time_episode = time.time()
            state, _ = self.env.reset()
            self.train_step(state)

            while not self.env.done:
                epoch += 1
                next_state = self.train_step(state)
                progress_bar.set_postfix({'total_reward': self.env.total_reward, 'epsilon': self.epsilon})
                progress_bar.update(1)

                state = next_state

            episode += 1
            end_time_episode = time.time()
            self.history.add('frames_per_episode', self.env.frame_count)
            self.history.add('reward_per_episode', self.env.total_reward)
            self.history.add('time_per_episode', end_time_episode - start_time_episode)
            self.history.add('memory_usage', self.replay_buffer.memory_usage())
            self.history.add('replay_buffer_size', self.replay_buffer.size())

            if episode % self.checkpoints == 0:
                checkpoint_path = os.path.join(self.path, f'checkpoint_{episode}')
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                self.save_model(checkpoint_path)
                self.save_history(checkpoint_path)
        
        end_time = time.time()
        self.history.add('total_time', end_time - start_time)
        self.history.add('frames', self.total_frames)
        print(f"Memory usage: {self.replay_buffer.memory_usage()} GB")
    