import argparse
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm

gym.register_envs(ale_py)
# ====================================================
# Definicja sieci DQN (ta sama architektura co wcześniej)
# ====================================================
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy_input)
            conv_out_size = int(conv_out.view(1, -1).size(1))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0  # normalizacja pikseli
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size(0), -1)
        return self.fc(conv_out)

# ====================================================
# Funkcje pomocnicze do pobierania i ustawiania parametrów
# ====================================================
def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().ravel())
    flat_params = np.concatenate(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        new_vals = flat_params[offset:offset + numel].reshape(param.shape)
        param.data.copy_(torch.tensor(new_vals, dtype=param.data.dtype, device=param.data.device))
        offset += numel

# ====================================================
# Funkcja ewaluacji – wykonuje jeden (lub więcej) epizodów i zwraca średnią sumę nagród
# ====================================================
def evaluate(model, env, device, episodes=1, render=False):
    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # Przygotowanie stanu (przekształcenie do tensora)
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = q_values.argmax(1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if render:
                env.render()
    return total_reward / episodes

# ====================================================
# Główna funkcja treningowa przy użyciu algorytmu genetycznego
# ====================================================
def train_ga(args):
    # Tworzymy środowisko i ustawiamy wrappery Atari
    env = gym.make(args.env_name)
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4, terminal_on_life_loss=True)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Upewnij się, że kształt obserwacji jest krotką liczb całkowitych
    input_shape = tuple(map(int, env.observation_space.shape))
    if len(input_shape) == 3 and input_shape[2] == 4:
        input_shape = (4, input_shape[0], input_shape[1])
    num_actions = int(env.action_space.n)

    # Inicjalizacja bazowego modelu i pobranie spłaszczonego wektora parametrów
    model = DQN(input_shape, num_actions).to(device)
    flat_params = get_flat_params_from(model)
    param_dim = flat_params.shape[0]

    # Ustawienia algorytmu genetycznego
    population_size = args.population_size
    num_generations = args.num_generations
    elite_frac = args.elite_frac   # np. 0.2 oznacza, że 20% najlepszych przejdzie do kolejnej generacji
    num_elite = max(1, int(population_size * elite_frac))
    mutation_std = args.mutation_std  # np. 0.02

    # Inicjalizacja populacji – każdy osobnik to wektor parametrów
    population = [flat_params + np.random.randn(param_dim) * 0.1 for _ in range(population_size)]

    best_reward = -np.inf
    best_params = None

    progress_bar = tqdm(total=num_generations, desc='Training Progress')
    for gen in range(num_generations):
        rewards = []
        for individual in population:
            set_flat_params_to(model, individual)
            # Ewaluujemy pojedynczy epizod; można zwiększyć liczbę epizodów dla stabilniejszego oszacowania fitness
            reward = evaluate(model, env, device, episodes=1)
            rewards.append(reward)

        # Sortujemy populację według fitness (malejąco)
        sorted_indices = np.argsort(rewards)[::-1]
        population = [population[i] for i in sorted_indices]
        rewards = [rewards[i] for i in sorted_indices]

        # Zapamiętujemy najlepszego osobnika
        if rewards[0] > best_reward:
            best_reward = rewards[0]
            best_params = population[0].copy()

        progress_bar.set_postfix({'generation': gen, 'best_reward': f'{rewards[0]:.2f}', 'mean_reward': f'{np.mean(rewards):.2f}'})

        # Tworzymy nową populację – elitizm oraz mutacja
        elites = population[:num_elite]
        new_population = elites.copy()
        while len(new_population) < population_size:
            parent = random.choice(elites)
            child = parent + np.random.randn(param_dim) * mutation_std
            new_population.append(child)
        population = new_population
        progress_bar.update(1)
    progress_bar.close()

    # Po zakończeniu treningu ustawiamy najlepsze parametry do modelu
    set_flat_params_to(model, best_params)

    # Zapisujemy model (można zapisać state_dict lub np. skompilowaną wersję)
    torch.save(model.state_dict(), args.save_path)
    print(f"Najlepszy model zapisany do {args.save_path} z nagrodą {best_reward:.2f}")

    env.close()

# ====================================================
# Funkcja renderingu – uruchomienie wytrenowanego modelu
# ====================================================
def render_model(args):
    env = gym.make(args.env_name, render_mode="human")
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4)
    env = gym.wrappers.FrameStack(env, num_stack=4)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    input_shape = tuple(map(int, env.observation_space.shape))
    if len(input_shape) == 3 and input_shape[2] == 4:
        input_shape = (4, input_shape[0], input_shape[1])
    num_actions = int(env.action_space.n)

    model = DQN(input_shape, num_actions).to(device)
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    model.eval()

    num_episodes = args.num_episodes
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = q_values.argmax(1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            env.render()
        print(f"Epizod {ep}: Nagroda = {episode_reward}")
    env.close()

# ====================================================
# Główna część – parsowanie argumentów i wybór trybu
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="Deep Neuroevolution dla DQN (GA)")
    parser.add_argument("--mode", choices=["train", "render"], default="train", help="Tryb: train lub render")
    parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4", help="Nazwa środowiska Atari")
    parser.add_argument("--save_path", type=str, default="ga_dqn_model.pth", help="Ścieżka zapisu/ładowania modelu")
    parser.add_argument("--population_size", type=int, default=50, help="Wielkość populacji")
    parser.add_argument("--num_generations", type=int, default=100, help="Liczba generacji")
    parser.add_argument("--elite_frac", type=float, default=0.2, help="Frakcja elit (np. 0.2 = 20%)")
    parser.add_argument("--mutation_std", type=float, default=0.02, help="Odchylenie standardowe mutacji")
    parser.add_argument("--num_episodes", type=int, default=10, help="Liczba epizodów przy renderingu")

    args = parser.parse_args()

    if args.mode == "train":
        train_ga(args)
    elif args.mode == "render":
        render_model(args)

if __name__ == "__main__":
    main()
