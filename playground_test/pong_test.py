import gymnasium as gym
import ale_py
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer

gym.register_envs(ale_py)


class DQN(nn.Module):
    def __init__(self, nb_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, nb_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)


def Deep_Q_Learning(
        env, replay_buffer_size=300_000, nb_epochs=10_000_000, update_frequency=4, 
        batch_size=32, discount_factor=0.99, replay_start_size=80_000, initial_exploration=1, 
        final_exploration=0.01, exploration_steps=1_000_000, device='cuda'
    ):

    rb = ReplayBuffer(replay_buffer_size, env.observation_space, env.action_space, device, optimize_memory_usage=True, handle_timeout_termination=False)

    q_network = DQN(env.action_space.n).to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=1.25e-4) # 0.000125

    epoch = 0
    smoothed_rewards = []
    rewards = []

    progress_bar = tqdm(total=nb_epochs, desc='Training')

    while epoch <= nb_epochs:

        dead = False
        total_reward = 0

        obs, _ = env.reset()

        for _ in range(random.randint(1, 30)):
            obs, _, _, _, info = env.step(env.action_space.sample())

        while not dead:
            current_life = info['lives']

            epsilon = max((final_exploration - initial_exploration) / exploration_steps * epoch + initial_exploration, final_exploration)

            if random.random() < epsilon:
                action = np.array(env.action_space.sample())
            else:
                q_values = q_network(torch.tensor(obs).unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            dead = terminated or truncated

            done = True if (info['lives'] < current_life) else False

            real_next_obs = next_obs.copy()

            total_reward += reward
            reward = np.sign(reward)

            rb.add(obs, real_next_obs, np.array([action]), reward, done, info)

            obs = next_obs

            if epoch > replay_start_size and epoch % update_frequency == 0:
                data = rb.sample(batch_size)
                with torch.no_grad():
                    max_q_value, _ = q_network(data.next_observations).max(dim=1)
                    y = data.rewards.flatten() + discount_factor * max_q_value * (1 - data.dones.flatten())
                current_q_value = q_network(data.observations).gather(1, data.actions).squeeze()

                loss = F.huber_loss(y, current_q_value)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            epoch += 1
            if (epoch % 50_000 == 0) and (epoch > 0):
                smoothed_rewards.append(np.mean(rewards))
                rewards = []
                plt.plot(smoothed_rewards)
                plt.title("Average Reward ond Beakout")
                plt.xlabel("Training Epochs")
                plt.ylabel("Average Reward per Episode")
                plt.savefig("average_reward_on_breakout.png")
                plt.close()

            progress_bar.update(1)
        rewards.append(total_reward)
    
    torch.save(q_network.state_dict(), 'pong_dqn.pth')


def evaluate_model(model_path, num_episodes=5, render_mode='human'):
    """
    Evaluate a saved DQN model with visual rendering
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to evaluate
    """
    # Set up environment
    env = gym.make('BreakoutNoFrameskip-v4', render_mode=render_mode)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = MaxAndSkipEnv(env, skip=4)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DQN(env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        _, _, _, _, info = env.step(1)
        current_life = info['lives']

        while not done:
            # Get action from model
            with torch.no_grad():
                q_values = model(torch.tensor(obs).unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=1).item()

            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if info['lives'] < current_life:
                done = True

        print(f"Episode {episode + 1} Total Reward: {total_reward}")

    env.close()


if __name__ == '__main__':
    # env = gym.make('BreakoutNoFrameskip-v4')
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayscaleObservation(env)
    # env = gym.wrappers.FrameStackObservation(env, 4)
    # env = MaxAndSkipEnv(env, skip=4)
    # Deep_Q_Learning(env, device='cuda')
    # env.close()

    evaluate_model('pong_dqn.pth', num_episodes=10, render_mode='rgb_array')

