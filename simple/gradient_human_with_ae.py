import argparse
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
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

gym.register_envs(ale_py)


class NoiseTable:
    def __init__(self, table_size=10**6, seed=42):
        # Pre-generujemy dużą tablicę losowych liczb z rozkładu Gaussa
        self.table_size = table_size
        self.rng = np.random.RandomState(seed)
        self.noise = self.rng.randn(table_size).astype(np.float32)
    
    def get_noise(self, shape, offset=None):
        """
        Zwraca tensor z szumem o podanym kształcie.
        Jeśli offset nie jest podany, losowo wybiera początek.
        """
        num_elements = np.prod(shape)
        if offset is None:
            # Upewnij się, że offset + num_elements nie przekracza rozmiaru tablicy
            offset = self.rng.randint(0, self.table_size - num_elements)
        # Pobieramy fragment tablicy i przekształcamy go do zadanego kształtu
        noise_slice = self.noise[offset: offset + num_elements]
        noise_tensor = torch.tensor(noise_slice.reshape(shape))
        return noise_tensor, offset  # zwracamy też offset, by ewentualnie móc później go reużyć

noise_table = NoiseTable(table_size=10**7, seed=42)

class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.)


# Funkcja dodająca szum Gaussa do wag modelu (mutacja)
def mutate_agent(agent, sigma):
    for param in agent.parameters():
        noise, _ = noise_table.get_noise(param.data.shape)
        noise = noise.to('cuda')
        param.data.add_(noise * sigma)
    return agent


def evaluate_agent(agent, env, device, num_episodes=1):
    agent.eval()
    total_rewards = []
    trajectory = []  # lista przechowująca przejścia: (state, next_state, action, reward, done, info)
    for _ in range(num_episodes):
        state, info = env.reset()
        dead = False
        ep_reward = 0

        for _ in range(random.randint(1, 30)):
            obs, _, _, _, info = env.step(1)

        while not dead:
            # Działanie według polityki deterministycznej
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = agent(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
            current_life = info['lives']
            obs, reward, terminated, truncated, info = env.step(action)
            done = 'lives' in info and info['lives'] < current_life
            dead = terminated or truncated
            ep_reward += reward
            next_state = obs.copy()
            trajectory.append((state, next_state, action, reward, done, info))

            if done and not dead:
                env.step(1)

            state = obs
        total_rewards.append(ep_reward)
    return np.mean(total_rewards), trajectory


# Faza ewolucyjna – generujemy populację, ewoluujemy przez określoną liczbę generacji
def evolutionary_phase(env, population_size, generations, sigma, device, base_agent, num_episodes=1):
    # Generujemy początkową populację: kopiujemy bazowy model i stosujemy mutację
    population = []
    new_population = []
    for _ in range(population_size):
        agent = DQN(env.action_space.n).to(device)
        agent.load_state_dict(base_agent.state_dict())
        agent = mutate_agent(agent, sigma)
        population.append(agent)
        
    trajectories = []
    # pb = tqdm(total=generations, desc='Evolutionary Phase')
    for gen in range(generations):
        rewards = []
        if len(new_population) > 0:
            population = new_population
        # Ewaluacja wszystkich agentów – jeden epizod na agenta
        for agent in population:
            avg_reward, traj = evaluate_agent(agent, env, device, num_episodes=num_episodes)
            rewards.append(avg_reward)
            trajectories.extend(traj)
        best_reward = max(rewards)
        # Selekcja: wybieramy top 20% agentów
        num_selected = max(1, int(0.2 * population_size))
        sorted_indices = np.argsort(rewards)[::-1]
        selected_agents = [population[i] for i in sorted_indices[:num_selected]]

        # Tworzymy nową populację: rodzice przechodzą bez zmian, reszta generowana jest przez mutację
        new_population = []
        new_population.extend(selected_agents)  # zachowujemy rodziców
        while len(new_population) < population_size:
            parent = random.choice(selected_agents)
            child = DQN(env.action_space.n).to(device)
            child.load_state_dict(parent.state_dict())
            child = mutate_agent(child, sigma)
            new_population.append(child)
        # pb.set_postfix(best_reward=best_reward, mean_reward=np.mean(rewards))
        # pb.update(1)
    # pb.close()
    # Po zakończeniu ewolucji, ewaluujemy populację jeszcze raz, aby zebrać doświadczenia
    final_rewards = []
    final_trajectories = []
    for agent in population:
        avg_reward, traj = evaluate_agent(agent, env, device, num_episodes=num_episodes)
        final_rewards.append(avg_reward)
        final_trajectories.append(traj)
    sorted_indices = np.argsort(final_rewards)[::-1]
    num_best = max(1, int(0.2 * population_size))
    best_agents = [population[i] for i in sorted_indices[:num_best]]
    combined_experiences = trajectories
    for i in sorted_indices[:num_best]:
        combined_experiences.extend(final_trajectories[i])

    mean_reward = np.mean(final_rewards)
    best_reward = max(final_rewards)
    return best_agents, combined_experiences, mean_reward, best_reward



def deep_Q_learning(env, folder, batch_size=32, M=30_000_000, epsilon_start=1, epsilon_end=0.1, nb_exploration_steps=1_000_000,
                    buffer_size=1_000_000, gamma=0.99, training_start_it=80_000, update_frequency=4, device='cuda', C=10_000, population_size=5, generations=3, sigma=0.01, ae_start_after=0, best_to_new_model=False):
    
    rb = ReplayBuffer(buffer_size, env.observation_space, env.action_space, device=device, n_envs=1, optimize_memory_usage=True, handle_timeout_termination=False)

    q_network = DQN(env.action_space.n).to(device)
    target_network = DQN(env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=1.25e-4)

    smoothed_rewards = []
    smoothed_rewards_50_000 = []
    mean_reward_generations = []
    best_reward_generations = []
    experiance_from_evo = []

    rewards = []
    max_reward = 0
    all_rewards = []
    rewards_50_000 = []

    epoch = 0
    progress_bar = tqdm(total=M, desc='Training')

    ae_explore = False

    sigma = torch.tensor(sigma, dtype=torch.float32).to(device)

    while epoch < M:
        state, _ = env.reset()
        dead = False
        total_reward = 0


        if ae_explore and population_size > 1 and epoch > ae_start_after:
            best_agents, evo_experiences, mean_reward_generation, best_reward_generation = evolutionary_phase(env, population_size, generations, sigma, device, q_network, num_episodes=1)
            mean_reward_generations.append(mean_reward_generation)
            best_reward_generations.append(best_reward_generation)
            experiance_from_evo.append(len(evo_experiences))
            for transition in evo_experiences:
                state, next_state, action, reward, done, info = transition
                rb.add(state, next_state, np.array(action), np.sign(reward), done, info)
            ae_explore = False
            state, _ = env.reset()

            if best_to_new_model:
                q_network.load_state_dict(best_agents[0].state_dict())


        for _ in range(random.randint(1, 30)):
            obs, _, _, _, info = env.step(1)

        while not dead:
            epsilon = max((epsilon_end - epsilon_start) * epoch / nb_exploration_steps + epsilon_start, epsilon_end)

            if np.random.rand() < epsilon:
                action = np.array(env.action_space.sample())
            else:
                with torch.no_grad():
                    q = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                    action = np.array(torch.argmax(q, dim=1).item())

            current_life = info['lives']
            obs, reward, terminated, truncated, info = env.step(action)
            done = 'lives' in info and info['lives'] < current_life
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

            if (epoch % 5_000 == 0) and (epoch > 0):
                smoothed_rewards.append(np.mean(rewards))
                rewards = []
                plt.clf()
                plt.plot(smoothed_rewards)
                plt.title('Average Reward on Breakout Mean 5 000 steps')
                plt.xlabel('Training Epoch')
                plt.ylabel('Average Reward per Episode')
                plt.savefig(f'results/{folder}/average_rewards_target_dqn_5_000.png')
                plt.close()
            
            if (epoch % 50_000 == 0) and (epoch > 0):
                smoothed_rewards_50_000.append(np.mean(rewards_50_000))
                rewards_50_000 = []
                plt.clf()
                plt.plot(smoothed_rewards_50_000)
                plt.title('Average Reward on Breakout Mean 50 000 steps')
                plt.xlabel('Training Epoch')
                plt.ylabel('Average Reward per Episode')
                plt.savefig(f'results/{folder}/average_rewards_target_dqn_50_000.png')
                plt.close()


            epoch += 1

            if epoch % C == 0:
                target_network.load_state_dict(q_network.state_dict())
                ae_explore = True
            
            state = obs
            progress_bar.update(1)
        rewards.append(total_reward)
        all_rewards.append(total_reward)
        rewards_50_000.append(total_reward)

        if total_reward > max_reward:
            max_reward = total_reward
            os.makedirs(f"results/{folder}/models/{epoch}", exist_ok=True)
            torch.save(q_network.state_dict(), f"results/{folder}/models/{epoch}/target_q_network_{max_reward}.pt")
    
    return q_network, mean_reward_generations, best_reward_generations, experiance_from_evo, all_rewards


def experiment(population_size, generation, noise_sigma):

        folder = f'gradient_ae_experiments/ps{population_size}_g{generation}_ns{noise_sigma}'
        os.makedirs(f'results/{folder}/models', exist_ok=True)
        os.makedirs(f'results/{folder}/logs', exist_ok=True)

        env = gym.make('BreakoutNoFrameskip-v4')
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env = MaxAndSkipEnv(env, skip=4)
        env = Monitor(env, f"results/{folder}/logs/monitor.csv")

        start_time = time.time()
        q_network, mean_reward_generations, best_reward_generations, experiance_from_evo, all_rewards = deep_Q_learning(
            env, folder, batch_size=32, M=1_000_000, epsilon_start=1, epsilon_end=0.01, nb_exploration_steps=1_000_000,
            buffer_size=1_000_000, gamma=0.99, training_start_it=80_000, update_frequency=4, device='cuda', C=10_000,
            population_size=population_size, generations=generation, sigma=noise_sigma
        )
                    
        end_time = time.time()

        total_reward, _ = evaluate_agent(q_network, env, 'cuda', num_episodes=1)
        torch.save(q_network.state_dict(), f"results/{folder}/final_trained_model_{total_reward}.pt")

        with open(f'results/{folder}/logs/time.txt', 'w') as f:
            f.write(f"Total time: {end_time - start_time} seconds")
        
        with open(f'results/{folder}/logs/generations_report.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'Mean Reward', 'Best Reward', 'Experience from Evo'])
            for i, (mean, best, exp) in enumerate(zip(mean_reward_generations, best_reward_generations, experiance_from_evo)):
                writer.writerow([i, mean, best, exp])
        
        with open(f'results/{folder}/logs/all_rewards.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Reward'])
            for reward in all_rewards:
                writer.writerow([reward])

        
        # Wczytanie danych z pliku CSV
        file_path = f"results/{folder}/logs/monitor.csv"
        df = pd.read_csv(file_path, skiprows=1)  # Pomija pierwszy wiersz z metadanymi

        # Ustawienie nazw kolumn
        df.columns = ["reward", "frames", "time"]
        # grupowanie danych po 1000 epizodach
        df = df.groupby(df.index // 100).mean()

        # Wykres 1: Reward per Game (Postęp agenta)
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["reward"], marker="o", linestyle="-", label="Reward per Game")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/reward_per_episode.png')

        # Wykres 2: Frames per Game (Czas trwania epizodów)
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["frames"], marker="o", linestyle="-", color="red", label="Frames per Game")
        plt.xlabel("Episode")
        plt.ylabel("Frames")
        plt.title("Frames per Episode")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/frames_per_episode.png')

        # Wykres 3: Reward vs Frames (Porównanie długości epizodu i nagrody)
        plt.figure(figsize=(10, 5))
        plt.scatter(df["frames"], df["reward"], c="blue", label="Reward vs Frames")
        plt.xlabel("Frames per Game")
        plt.ylabel("Total Reward")
        plt.title("Reward vs Frames")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/reward_vs_frames.png')

        plt.figure(figsize=(10, 5))
        plt.plot(all_rewards)
        plt.title("All Rewards without AE")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(f'results/{folder}/all_rewards.png')

        # Revards mean 50 000
        all_rewards_mean_100_episodes = [np.mean(all_rewards[i:i+100]) for i in range(0, len(all_rewards), 100)]
        plt.figure(figsize=(10, 5))
        plt.plot(all_rewards_mean_100_episodes)
        plt.title("All Rewards mean 100 episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(f'results/{folder}/all_rewards_mean_100_episodes.png')


def experiment2(ae_start_after, population_size):

        folder = f'gradient_ae_experiments2_attempt2/ps{population_size}_ae{ae_start_after}'
        os.makedirs(f'results/{folder}/models', exist_ok=True)
        os.makedirs(f'results/{folder}/logs', exist_ok=True)

        env = gym.make('BreakoutNoFrameskip-v4')
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env = MaxAndSkipEnv(env, skip=4)
        env = Monitor(env, f"results/{folder}/logs/monitor.csv")

        start_time = time.time()
        q_network, mean_reward_generations, best_reward_generations, experiance_from_evo, all_rewards = deep_Q_learning(
            env, folder, batch_size=32, M=2_000_000, epsilon_start=1, epsilon_end=0.01, nb_exploration_steps=1_000_000,
            buffer_size=1_000_000, gamma=0.99, training_start_it=80_000, update_frequency=4, device='cuda', C=10_000,
            population_size=population_size, generations=3, sigma=0.0005, ae_start_after=ae_start_after, best_to_new_model=False
        )
                    
        end_time = time.time()

        total_reward, _ = evaluate_agent(q_network, env, 'cuda', num_episodes=1)
        torch.save(q_network.state_dict(), f"results/{folder}/final_trained_model_{total_reward}.pt")

        with open(f'results/{folder}/logs/time.txt', 'w') as f:
            f.write(f"Total time: {end_time - start_time} seconds")
        
        with open(f'results/{folder}/logs/generations_report.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'Mean Reward', 'Best Reward', 'Experience from Evo'])
            for i, (mean, best, exp) in enumerate(zip(mean_reward_generations, best_reward_generations, experiance_from_evo)):
                writer.writerow([i, mean, best, exp])
        
        with open(f'results/{folder}/logs/all_rewards.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Reward'])
            for reward in all_rewards:
                writer.writerow([reward])

        
        # Wczytanie danych z pliku CSV
        file_path = f"results/{folder}/logs/monitor.csv"
        df = pd.read_csv(file_path, skiprows=1)  # Pomija pierwszy wiersz z metadanymi

        # Ustawienie nazw kolumn
        df.columns = ["reward", "frames", "time"]
        # grupowanie danych po 1000 epizodach
        df = df.groupby(df.index // 100).mean()

        # Wykres 1: Reward per Game (Postęp agenta)
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["reward"], marker="o", linestyle="-", label="Reward per Game")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/reward_per_episode.png')

        # Wykres 2: Frames per Game (Czas trwania epizodów)
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["frames"], marker="o", linestyle="-", color="red", label="Frames per Game")
        plt.xlabel("Episode")
        plt.ylabel("Frames")
        plt.title("Frames per Episode")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/frames_per_episode.png')

        # Wykres 3: Reward vs Frames (Porównanie długości epizodu i nagrody)
        plt.figure(figsize=(10, 5))
        plt.scatter(df["frames"], df["reward"], c="blue", label="Reward vs Frames")
        plt.xlabel("Frames per Game")
        plt.ylabel("Total Reward")
        plt.title("Reward vs Frames")
        plt.legend()
        plt.grid()
        plt.savefig(f'results/{folder}/reward_vs_frames.png')

        plt.figure(figsize=(10, 5))
        plt.plot(all_rewards)
        plt.title("All Rewards without AE")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(f'results/{folder}/all_rewards.png')

        # Revards mean 50 000
        all_rewards_mean_100_episodes = [np.mean(all_rewards[i:i+100]) for i in range(0, len(all_rewards), 100)]
        plt.figure(figsize=(10, 5))
        plt.plot(all_rewards_mean_100_episodes)
        plt.title("All Rewards mean 100 episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(f'results/{folder}/all_rewards_mean_100_episodes.png')




if __name__ == "__main__":

    population_sizes = [4, 8, 16]
    generations = [1, 3, 5]
    noise_sigmas = [0.0001, 0.0005, 0.001]

    # get sigmas from args    
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_size", type=int, default=4)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--noise_sigma", type=float, default=0.0001)
    parser.add_argument("--ae_start_after", type=int, default=0)
     
    args = parser.parse_args()

    population_size = args.population_size
    generation = args.generation
    noise_sigma = args.noise_sigma
    ae_start_after = args.ae_start_after

    # experiment(population_size, generation, noise_sigma)
    experiment2(ae_start_after, population_size)
    # experiment(0, 0, 0)

    # print("Starting experiment with AE")
    # for generation in generations:
    #     for noise_sigma in noise_sigmas:
    #         print(f"Starting experiment with population size {population_size}, generation {generation}, noise sigma {noise_sigma}")
    #         experiment(population_size, generation, noise_sigma)

    
    # env = gym.make('BreakoutNoFrameskip-v4')
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayscaleObservation(env)
    # env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    # env = MaxAndSkipEnv(env, skip=4)

    # net = DQN(env.action_space.n).to('cuda')
    # net.load_state_dict(torch.load(f"results/gradient_human/models/42602170/target_q_network_428.0.pt"))

    # mean_reward, _ = evaluate_agent(net, env, 'cuda', num_episodes=1)
    # print(f"Mean reward: {mean_reward}")

    # _, _, mean_reward_generation, best_reward_generation = evolutionary_phase(env, 4, 1, 0.0005, 'cuda', net, num_episodes=1)
    # print(f"Mean reward generation: {mean_reward_generation}")
    # print(f"Best reward generation: {best_reward_generation}")
