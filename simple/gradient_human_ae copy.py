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
        noise = torch.randn_like(param) * sigma
        param.data.add_(noise)
    return agent


def evaluate_agent(agent, env, device, num_episodes=1):
    agent.eval()
    total_rewards = []
    trajectory = []  # lista przechowująca przejścia: (state, next_state, action, reward, done, info)
    for _ in range(num_episodes):
        state, info = env.reset()
        dead = False
        ep_reward = 0

        # for _ in range(random.randint(1, 30)):
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
    agent.train()
    return np.mean(total_rewards), trajectory


# Faza ewolucyjna – generujemy populację, ewoluujemy przez określoną liczbę generacji
def evolutionary_phase(env, population_size, generations, sigma, device, base_agent, num_episodes=1):
    # Generujemy początkową populację: kopiujemy bazowy model i stosujemy mutację
    population = []
    for _ in range(population_size):
        agent = DQN(env.action_space.n).to(device)
        agent.load_state_dict(base_agent.state_dict())
        agent = mutate_agent(agent, sigma)
        population.append(agent)
        
    trajectories_best = []
    pb = tqdm(total=generations, desc='Evolutionary Phase')
    for gen in range(generations):
        rewards = []
        trajectories = []
        # Ewaluacja wszystkich agentów – jeden epizod na agenta
        for agent in population:
            avg_reward, traj = evaluate_agent(agent, env, device, num_episodes=num_episodes)
            rewards.append(avg_reward)
            trajectories.append(traj)
        best_reward = max(rewards)
        # Selekcja: wybieramy top 20% agentów
        num_selected = max(1, int(0.2 * population_size))
        sorted_indices = np.argsort(rewards)[::-1]
        selected_agents = [population[i] for i in sorted_indices[:num_selected]]

        for i in sorted_indices[:num_selected]:
            trajectories_best.extend(trajectories[i])
        # Tworzymy nową populację: rodzice przechodzą bez zmian, reszta generowana jest przez mutację
        new_population = []
        new_population.extend(selected_agents)  # zachowujemy rodziców
        while len(new_population) < population_size:
            parent = random.choice(selected_agents)
            child = DQN(env.action_space.n).to(device)
            child.load_state_dict(parent.state_dict())
            child = mutate_agent(child, sigma)
            new_population.append(child)
        population = new_population
        pb.set_postfix(best_reward=best_reward, mean_reward=np.mean(rewards))
        pb.update(1)
    pb.close()
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
    combined_experiences = trajectories_best
    for i in sorted_indices[:num_best]:
        combined_experiences.extend(final_trajectories[i])

    mean_reward = np.mean(final_rewards)
    best_reward = max(final_rewards)
    return best_agents, combined_experiences, mean_reward, best_reward


# Faza gradientowa – fine-tuning modelu przy wykorzystaniu zebranych doświadczeń
def gradient_training_phase(env, replay_buffer, device, total_steps, folder, base_agent, smoothed_rewards, batch_size=32, gamma=0.99, training_start_it=80_000, update_frequency=4, target_update_freq=10_000, epsilon_start=1, epsilon_end=0.1, nb_exploration_steps=1_000_000):
    q_network = base_agent
    target_network = DQN(env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=1.25e-4)
    
    epoch = 0
    progress_bar = tqdm(total=total_steps, desc='Gradient Fine-Tuning')
    rewards = []
    max_reward = -float('inf')
    
    while epoch < total_steps:
        state, info = env.reset()
        dead = False
        total_reward = 0

        for _ in range(random.randint(1, 30)):
            obs, _, _, _, info = env.step(1)
        
        while not dead:
            # Używamy epsilon-greedy (można dostroić epsilon według schematu)
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
            next_state = obs.copy()
            # Dodajemy do replay buffera – uzupełniony wcześniej o doświadczenia z ewolucji
            replay_buffer.add(state, next_state, action, np.sign(reward), done, info)

            if done and not dead:
                env.step(1)
            
            if (epoch > training_start_it) and (epoch % update_frequency == 0):
                batch = replay_buffer.sample(batch_size)
                with torch.no_grad():
                    max_q_value_next_state = target_network(batch.next_observations).max(dim=1).values
                    y_j = batch.rewards.squeeze(-1) + gamma * max_q_value_next_state * (1. - batch.dones.squeeze(-1).float())
                current_q_value = q_network(batch.observations).gather(1, batch.actions).squeeze(-1)
                loss = torch.nn.functional.huber_loss(y_j, current_q_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % target_update_freq == 0 and epoch > 0:
                target_network.load_state_dict(q_network.state_dict())

            if (epoch % 50_000 == 0) and (epoch > 0):
                smoothed_rewards.append(np.mean(rewards))
                rewards = []
                plt.plot(smoothed_rewards)
                plt.title('Average Reward on Breakout')
                plt.xlabel('Training Epoch')
                plt.ylabel('Average Reward per Episode')
                plt.savefig(f"results/{folder}/average_rewards_target_dqn.png")
                plt.close()
                
            state = obs
            epoch += 1
            progress_bar.update(1)
        rewards.append(total_reward)

        if total_reward > max_reward:
            max_reward = total_reward
            os.makedirs(f"results/{folder}/models/{epoch}", exist_ok=True)
            torch.save(q_network.state_dict(), f"results/{folder}/models/{epoch}/target_q_network_{max_reward}.pt")
    progress_bar.close()
    return q_network, total_reward


def gradient_phase_witout_evaluation(env, replay_buffer, device, total_steps, base_agent, batch_size=32, gamma=0.99):
    base_agent.train()
    target_network = DQN(env.action_space.n).to(device)
    target_network.load_state_dict(base_agent.state_dict())
    optimizer = torch.optim.Adam(base_agent.parameters(), lr=1.25e-4)

    epoch = 0
    progress_bar = tqdm(total=total_steps, desc='Gradient Fine-Tuning')

    while epoch < total_steps:
        batch = replay_buffer.sample(batch_size)
        with torch.no_grad():
            max_q_value_next_state = target_network(batch.next_observations).max(dim=1).values
            y_j = batch.rewards.squeeze(-1) + gamma * max_q_value_next_state * (1. - batch.dones.squeeze(-1).float())
        current_q_value = base_agent(batch.observations).gather(1, batch.actions).squeeze(-1)
        loss = torch.nn.functional.huber_loss(y_j, current_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        progress_bar.update(1)
    progress_bar.close()
    return base_agent


if __name__ == "__main__":

    population_sizes = [10, 20, 50, 100]
    generations = [10, 20, 50, 100]
    noise_sigmas = [0.01, 0.05, 0.1, 0.2]

    experiance_from_evo = []

    train_iterations = 25
    fine_tuning_steps = 10_000
    num_episodes_per_agent = 1

    for population_size in population_sizes:
        for generation in generations:
            for noise_sigma in noise_sigmas:

                smoothed_rewards = []
                mean_reward_generations = []
                best_reward_generations = []

                max_reward = -float('inf')

                folder = f'evolutionary_exp1/{population_size}_g{generation}_ns{noise_sigma}'
                os.makedirs(f'results/{folder}/models', exist_ok=True)
                os.makedirs(f'results/{folder}/logs', exist_ok=True)

                env = gym.make('BreakoutNoFrameskip-v4')
                env = gym.wrappers.ResizeObservation(env, (84, 84))
                env = gym.wrappers.GrayscaleObservation(env)
                env = gym.wrappers.FrameStackObservation(env, stack_size=4)
                env = MaxAndSkipEnv(env, skip=4)
                env = Monitor(env, f"results/{folder}/logs/monitor.csv", allow_early_resets=True)

                start_time = time.time()
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # Faza ewolucyjna: generujemy populację, ewoluujemy i zbieramy doświadczenia
                    
                base_agent = DQN(env.action_space.n).to(device)

                replay_buffer = ReplayBuffer(
                    1_000_000, 
                    observation_space=env.observation_space, 
                    action_space=env.action_space, 
                    device=device, 
                    n_envs=1, 
                    optimize_memory_usage=True, 
                    handle_timeout_termination=False
                )

                pb = tqdm(total=train_iterations, desc='Training')

                for i in range(train_iterations):

                    print("Starting Evolutionary Phase...")
                    best_agents, evo_experiences, mean_reward_generation, best_reward_generation = evolutionary_phase(env, population_size, generation, noise_sigma, device, base_agent, num_episodes=num_episodes_per_agent)
                    print(f"Evolutionary phase complete. Collected {len(evo_experiences)} transitions from best agents.")
                    experiance_from_evo.append(len(evo_experiences))
                    mean_reward_generations.append(mean_reward_generation)
                    best_reward_generations.append(best_reward_generation)

                    # get best agent from evolutionary phase
                    base_agent = best_agents[0]

                    # Dodajemy przejścia do buffera
                    for transition in evo_experiences:
                        state, next_state, action, reward, done, info = transition
                        replay_buffer.add(state, next_state, np.array(action), np.sign(reward), done, info)
                    
                    # Evaluate best agent from evolutionary phase
                    reward, _ = evaluate_agent(base_agent, env, device, num_episodes=1)
                    print(f"Best agent from evolutionary phase reward: {reward}")
                    
                    # Faza gradientowa – fine-tuning modelu na podstawie Replay Buffera
                    print("Starting Gradient Fine-Tuning Phase...")
                    base_agent = gradient_phase_witout_evaluation(env, replay_buffer, device, fine_tuning_steps, base_agent, batch_size=32, gamma=0.99)

                    # evaluate iteration
                    reward, _ = evaluate_agent(base_agent, env, device, num_episodes=1)
                    print(f"Gradient Fine-Tuning Phase complete. Reward: {reward}")
                    smoothed_rewards.append(reward)

                    if reward > max_reward:
                        max_reward = reward
                        os.makedirs(f"results/{folder}/models/{(i+1)*fine_tuning_steps*4}", exist_ok=True)
                        torch.save(base_agent.state_dict(), f"results/{folder}/models/{(i+1)*fine_tuning_steps*4}/model_{max_reward}.pt")


                    plt.clf()
                    plt.plot(smoothed_rewards)
                    plt.title('Average Reward on Breakout')
                    plt.xlabel('Training Epoch')
                    plt.ylabel('Average Reward per Episode')
                    plt.savefig(f"results/{folder}/average_rewards_target_dqn.png")
                    plt.close()

                    pb.update(1)
                pb.close()
                # final evaluation
                total_reward, _ = evaluate_agent(base_agent, env, device, num_episodes=1)

                # Zapisywanie wyników i modeli
                torch.save(base_agent.state_dict(), f"results/{folder}/final_trained_model_{total_reward}.pt")
                
                end_time = time.time()
                env.close()
                with open(f'results/{folder}/logs/time.txt', 'w') as f:
                    f.write(f"Total time: {end_time - start_time} seconds")
                
                with open(f'results/{folder}/logs/generations_report.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Generation', 'Mean Reward', 'Best Reward', 'Experience from Evo'])
                    for i, (mean, best, exp) in enumerate(zip(mean_reward_generations, best_reward_generations, experiance_from_evo)):
                        writer.writerow([i, mean, best, exp])
                
                # Wczytanie danych z pliku CSV
                file_path = f"results/{folder}/logs/monitor.csv"
                df = pd.read_csv(file_path, skiprows=1)  # Pomija pierwszy wiersz z metadanymi

                # Ustawienie nazw kolumn
                df.columns = ["reward", "frames", "time"]
                # grupowanie danych po 1000 epizodach
                df = df.groupby(df.index // 200).mean()

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
                    




