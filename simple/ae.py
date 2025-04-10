import copy
import torch.multiprocessing as mp
import csv
import os
import random
import time
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, FireResetEnv, EpisodicLifeEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
import matplotlib.pyplot as plt
from gymnasium.vector import SyncVectorEnv

gym.register_envs(ale_py)

torch.set_num_threads(1)

env_name = 'BreakoutNoFrameskip-v4'
PARENT_PERCENT = 0.2

MAX_SEED = np.iinfo(np.int32).max
MUTATION_RATE = 0.005


class FireAfterLoseLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if lives < self.lives:
            obs, reward, terminated, truncated, info = self.env.step(1)
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


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
        self.apply(initialize_weights)

    def forward(self, x):
        return self.network(x)



def make_env():

    env = gym.make(env_name)
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=30, scale_obs=True)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.ClipReward(env, -1, 1)
    # env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = FireAfterLoseLifeEnv(env)
    # env = MaxAndSkipEnv(env, skip=4)
    # env = gym.wrappers.MaxAndSkipObservation(env, skip=4)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env

def make_env_wrapper():

    def _init():
        env = make_env()
        return env

    return _init


def make_net(action_space, seeds):
    torch.manual_seed(seeds[0])
    net = DQN(action_space)
    net = net.to('cuda')

    for seed in seeds[1:]:
        net = mutate(net, seed)

    return net


def reset_envs(envs):
    return envs.reset()


def step_envs(envs, actions):
    states, rewards, terminateds, truncateds, infos = envs.step(actions)

    # dones = np.array([terminated or truncated for terminated, truncated in zip(terminateds, truncateds)])
    # lives_indices = np.where(infos['lives'] < 5)
    # if len(lives_indices) > 0:
    #     dones[lives_indices] = True

    return states, rewards, terminateds, truncateds, infos


def close_envs(envs):
    envs.close()


def evaluate_batch(envs, nets):
    size = envs.num_envs
    states, _ = reset_envs(envs)
    total_rewards = torch.zeros(size)
    dones = np.zeros(size, dtype=np.bool)
    frames = torch.zeros(size, dtype=torch.int32)

    while not np.all(dones):
        actions = np.zeros(size, dtype=np.int32)
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32, device='cuda')

            not_done_indices = np.where(dones == False)[0]

            for i in not_done_indices:
                q_values = nets[i](states[i].unsqueeze(0))
                actions[i] = torch.argmax(q_values, dim=1).item()

        states, rewards, terminateds, truncateds, infos = step_envs(envs, actions)
        terminateds_indices = np.where(terminateds == True)[0]
        truncateds_indices = np.where(truncateds == True)[0]

        dones[terminateds_indices] = True
        dones[truncateds_indices] = True

        for i in range(size):
            if not dones[i]:
                total_rewards[i] += rewards[i]
                frames[i] += 1
        
    return total_rewards, frames


def evaluate(env, net):
    state, _ = env.reset()
    frames = 0
    total_reward = 0

    for _ in range(4):
        obs, _, _, _, info = env.step(1)

    for _ in range(1000):
        with torch.no_grad():
            q = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to('cuda'))
            action = torch.argmax(q, dim=1).item()

        state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        frames += 1

        if terminated or truncated:
            break
    
    return total_reward, frames


def mutate(net, mut):
    np.random.seed(mut)
    for p in net.parameters():
        noise = np.random.normal(loc=0, scale=MUTATION_RATE, size=p.data.size())
        noise_t = torch.tensor(noise, dtype=torch.float32).to('cuda')
        p.data.add_(noise_t)

    return net


def evaluate_individual(individual):
    env = make_env()
    net = make_net(env.action_space.n, individual)
    reward, frames = evaluate(env, net)
    env.close()
    return individual, reward, frames


def reward_based_selection(generation, parent_count):
    sorted_generation = sorted(generation, key=lambda x: x[1], reverse=True)
    return sorted_generation[:parent_count]


def mutate_seeds(individual):
    new_seed = np.random.randint(MAX_SEED)
    individual = list(individual) + [new_seed]
    
    return tuple(individual)


def create_new_population(elite_population, population_size):
    new_population = list(elite_population)

    while len(new_population) < population_size:
        parent_idx = np.random.randint(len(elite_population))
        parent = elite_population[parent_idx]
        child = mutate_seeds(parent)
        new_population.append(child)

    return new_population


global_envs = None
def evaluate_batch_worker(args):
    global global_envs

    batch_population, action_space = args
    global_envs.reset()

    nets = [make_net(action_space, ind) for ind in batch_population]
    rewards, frames = evaluate_batch(global_envs, nets)
    return rewards, frames

def init_global_envs(num_envs):
    global global_envs
    global_envs = [make_env() for _ in range(num_envs)]

    for env in global_envs:
        env.reset()

def init_global_envs_vec(num_envs):
    global global_envs
    global_envs = SyncVectorEnv([make_env_wrapper() for _ in range(num_envs)])


def ae_learning(population_size=1000, M=25_000_000):
    num_batches = 20
    batch_population_size = population_size // num_batches
    action_space = gym.make(env_name).action_space.n

    epoch = 0
    max_reward = 0
    rewards_max, rewards_avg, rewards_std = [], [], []
    frames_max, frames_avg, frames_std = [], [], []
    mutation_count_per_generation = []

    print("Make population")
    parent_count = int(population_size * PARENT_PERCENT)
    population = [(np.random.randint(MAX_SEED),) for _ in range(population_size)]

    pool = mp.Pool(processes=num_batches, initializer=init_global_envs_vec, initargs=(batch_population_size,))
    
    print("Start training")
    start_time = time.time()
    while epoch < M:
        batch_rewards, batch_frames = [], []

        batch_args = []
        for batch_idx in range(num_batches):
            batch_population = population[batch_idx * batch_population_size:(batch_idx + 1) * batch_population_size]
            batch_args.append((batch_population, action_space))
        
        results = pool.map(evaluate_batch_worker, batch_args)

        for rewards, frames in results:
            batch_rewards.extend(rewards)
            batch_frames.extend(frames)

        epoch += np.sum(batch_frames)
                    
        rewards_max.append(np.max(batch_rewards))
        rewards_avg.append(np.mean(batch_rewards))
        rewards_std.append(np.std(batch_rewards))

        frames_max.append(np.max(batch_frames))
        frames_avg.append(np.mean(batch_frames))
        frames_std.append(np.std(batch_frames))

        generation = [(copy.deepcopy(population[i]), batch_rewards[i], batch_frames[i]) for i in range(len(batch_rewards))]
        generation = reward_based_selection(generation, parent_count)

        elite = generation[0]

        if elite[1] > max_reward:
            max_reward = elite[1]
            if max_reward > 100:
                os.makedirs(f"results/ae/models/{epoch}", exist_ok=True)
                net = make_net(action_space, elite[0])
                torch.save(net.state_dict(), f"results/ae/models/{epoch}/target_q_network_{max_reward}.pt")

        # New population
        elite_population = [p[0] for p in generation]
        population = create_new_population(elite_population, population_size)
        
        mutation_count_per_generation.append(np.sum(np.array([len(x) for x in population]) > 1))

        iteration_per_second = epoch / (time.time() - start_time)
        estimated_time = (M - epoch) / iteration_per_second
        estimated_time = int_to_time(estimated_time)
        percentage = str(round(epoch/M*100, 2))
        print(f"M: {epoch}/{M} | R: max {max_reward} avg {rewards_avg[-1]:.4f} ep_max {rewards_max[-1]} | F: avg {frames_avg[-1]:.4f} | P: {percentage}% | ET: {estimated_time} | {iteration_per_second:.2f} it/s")
    
    pool.close()
    pool.join()


    print('Training finished')

    end_time = time.time()
    with open('results/ae/logs/time.txt', 'w') as f:
        f.write(f"Total time: {end_time - start_time} seconds")

    return rewards_max, rewards_avg, rewards_std, frames_max, frames_avg, frames_std, mutation_count_per_generation


def int_to_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:.0f}:{minutes:.0f}:{seconds:.0f}"


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.makedirs('results/ae/models', exist_ok=True)
    os.makedirs('results/ae/logs', exist_ok=True)

    rewards_max, rewards_avg, rewards_std, frames_max, frames_avg, frames_std, mutation_count_per_generation = ae_learning(population_size=1000, M=50_000_000)

    with open('results/ae/logs/summary.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'Max Reward', 'Average Reward', 'Standard Deviation', 'Max Frames', 'Average Frames', 'Standard Deviation', 'Mutation Count'])
        for i in range(len(rewards_max)):
            writer.writerow([i, rewards_max[i], rewards_avg[i], rewards_std[i], frames_max[i], frames_avg[i], frames_std[i], mutation_count_per_generation[i]])
        
    
    # Wykres 1: Reward per Game (Postęp agenta)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_max, label='Max Reward')
    plt.plot(rewards_avg, label='Average Reward')
    plt.plot(rewards_std, label='Standard Deviation')
    plt.legend()
    plt.savefig('results/ae/reward_per_generation.png')

    # Wykres 2: Frames per Game (Czas trwania epizodów)
    plt.figure(figsize=(10, 5))
    plt.plot(frames_max, label='Max Frames')
    plt.plot(frames_avg, label='Average Frames')
    plt.plot(frames_std, label='Standard Deviation')
    plt.legend()
    plt.savefig('results/ae/frames_per_generation.png')

    # Wykres 3: Reward vs Frames (Porównanie długości epizodu i nagrody)
    plt.figure(figsize=(10, 5))
    plt.scatter(frames_max, rewards_max, c='blue', label='Reward vs Frames')
    plt.xlabel('Frames per Generation')
    plt.ylabel('Total Reward')
    plt.title('Reward vs Frames')
    plt.legend()
    plt.savefig('results/ae/reward_vs_frames.png')

    # Wykres 4: Mutation count per generation
    plt.figure(figsize=(10, 5))
    plt.plot(mutation_count_per_generation)
    plt.xlabel('Generation')
    plt.ylabel('Mutation Count')
    plt.title('Mutation Count per Generation')
    plt.savefig('results/ae/mutation_count_per_generation.png')

