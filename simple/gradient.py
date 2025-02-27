import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

gym.register_envs(ale_py)


class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.)


def deep_Q_learning(env, batch_size=32, M=30_000_000, epsilon_start=1, epsilon_end=0.1, nb_exploration_steps=1_000_000,
                    buffer_size=1_000_000, gamma=0.99, training_start_it=80_000, update_frequency=4, device='cuda', C=10_000):

    rb = ReplayBuffer(buffer_size, env.observation_space, env.action_space, device=device, n_envs=1, optimize_memory_usage=True, handle_timeout_termination=False)

    q_network = DQN(env.action_space.n).to(device)
    target_network = DQN(env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=1.25e-4)

    smoothed_rewards = []
    rewards = []
    max_reward = 0

    epoch = 0
    progress_bar = tqdm(total=M, desc='Training')

    while epoch < M:
        state, _ = env.reset()
        dead = False
        total_reward = 0

        for _ in range(random.randint(1, 30)):
            obs, _, _, _, info = env.step(1)

        while not dead:
            epsilon = max((epsilon_end - epsilon_start) * epoch / nb_exploration_steps + epsilon_start, epsilon_end)

            if np.random.rand() < epsilon:
                action = np.array(env.action_space.sample())
            else:
                with torch.no_grad():
                    q = q_network(torch.tensor(state).unsqueeze(0).to(device))
                    action = np.array(torch.argmax(q, dim=1).item())

            current_life = info['lives']
            obs, reward, terminated, truncated, info = env.step(action)
            done = info['lives'] < current_life
            dead = terminated or truncated
            total_reward += reward
            reward = np.sign(reward)

            next_state = obs.copy()

            rb.add(state, next_state, action, reward, done, info)

            if (epoch > training_start_it) and (epoch % update_frequency == 0):
                batch = rb.sample(batch_size)

                with torch.no_grad():
                    max_q_value_next_state = target_network(batch.next_observations).max(dim=1).values
                    y_j = batch.rewards.squeeze(-1) + gamma * max_q_value_next_state * (1. - batch.dones.squeeze(-1).float())

                current_q_value = q_network(batch.observations).gather(1, batch.actions).squeeze(-1)

                loss = torch.nn.functional.huber_loss(y_j, current_q_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch % 50_000 == 0) and (epoch > 0):
                smoothed_rewards.append(np.mean(rewards))
                rewards = []
                plt.plot(smoothed_rewards)
                plt.title('Average Reward on Breakout')
                plt.xlabel('Training Epoch')
                plt.ylabel('Average Reward per Episode')
                plt.savefig('results/average_rewards_target_dqn.png')
                plt.close()

            epoch += 1

            if epoch % C == 0:
                target_network.load_state_dict(q_network.state_dict())
            
            state = obs
            progress_bar.update(1)
        rewards.append(total_reward)

        if total_reward > max_reward:
            max_reward = total_reward
            torch.save(q_network.cpu(), f"results/models/target_q_network_{epoch}_{max_reward}")
            q_network.to(device)
            

if __name__ == "__main__":
    env = gym.make('BreakoutNoFrameskip-v4')
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = MaxAndSkipEnv(env, skip=4)
    env = Monitor(env, f"results/logs/gradient_dqn")

    deep_Q_learning(env, batch_size=32, M=10_000_000, epsilon_start=1, epsilon_end=0.01, nb_exploration_steps=1_000_000,
                    buffer_size=500_000, gamma=0.99, training_start_it=80_000, update_frequency=4, device='cuda', C=10_000)
            





