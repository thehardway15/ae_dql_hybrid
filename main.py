import argparse
import gymnasium as gym
import ale_py
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

gym.register_envs(ale_py)

# ===========================================
# Definicja bufora powtórkowego (Replay Buffer)
# ===========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), actions, rewards, np.stack(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# ===========================================
# Definicja sieci DQN
# ===========================================
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
        x = x / 255.0  # Normalizacja pikseli
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0], -1)
        return self.fc(conv_out)

# ===========================================
# Funkcja wyboru akcji – epsilon-greedy
# ===========================================
def select_action(state, current_model, epsilon, num_actions, device):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = current_model(state)
        return q_values.argmax(1).item()

# ===========================================
# Funkcja obliczania straty (loss)
# ===========================================
def compute_loss(batch, current_model, target_model, device, gamma):
    states, actions, rewards, next_states, dones = batch
    states      = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions     = torch.LongTensor(actions).to(device)
    rewards     = torch.FloatTensor(rewards).to(device)
    dones       = torch.FloatTensor(dones).to(device)
    
    q_values = current_model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    next_q_values = target_model(next_states)
    next_q_value  = next_q_values.max(1)[0]
    
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)
    
    loss = nn.MSELoss()(q_value, expected_q_value.detach())
    return loss

# ===========================================
# Funkcja treningowa
# ===========================================
def train(args):
    env_name = args.env_name
    env = gym.make(env_name)
    
    # Wrappery Atari:
    # - AtariPreprocessing: konwertuje obraz do skali szarości, resize do 84x84, stosuje frame skip
    #   oraz no-op reset (domyślnie noop_max=30)
    # - FrameStack: stackuje kolejne 4 klatki
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # Upewnij się, że kształt obserwacji to krotka zwykłych int
    input_shape = tuple(map(int, env.observation_space.shape))
    if len(input_shape) == 3 and input_shape[2] == 4:
        input_shape = (4, input_shape[0], input_shape[1])
    num_actions = int(env.action_space.n)
    replay_buffer_capacity = args.replay_buffer_capacity
    
    # Inicjalizacja sieci
    current_model = DQN(input_shape, num_actions).to(device)
    target_model  = DQN(input_shape, num_actions).to(device)
    target_model.load_state_dict(current_model.state_dict())
    
    # Kompilacja modeli przy użyciu TorchScript
    current_model = torch.jit.script(current_model)
    target_model  = torch.jit.script(target_model)
    
    # Optymalizator – jeśli chcesz użyć RMSProp (zgodnie z artykułem), odkomentuj odpowiednią linię
    # optimizer = optim.RMSprop(current_model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
    optimizer = optim.Adam(current_model.parameters(), lr=0.00025)
    
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
    
    # Parametry treningu
    num_frames = args.num_frames
    batch_size = args.batch_size
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = num_frames  # liniowa dekrementacja epsilon przez cały trening
    target_update_frequency = 10000
    learning_starts = 50000
    update_frequency = 4
    clip_rewards = True  # opcjonalny reward clipping do -1, 0, 1
    
    # Do zbierania wyników
    all_rewards = []
    losses = []
    
    # UWAGA: W gymnasium reset() zwraca krotkę (observation, info)
    state, _ = env.reset()
    episode_reward = 0
    frame_idx = 0
    episode = 0

    progress_bar = tqdm(total=num_frames, desc='Training Progress')
    while frame_idx < num_frames:
        frame_idx += 1
        epsilon = max(epsilon_final, epsilon_start - frame_idx / epsilon_decay)
        
        action = select_action(state, current_model, epsilon, num_actions, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if clip_rewards:
            reward = np.sign(reward)
        
        replay_buffer.push(np.array(state), action, reward, np.array(next_state), done)
        state = next_state
        episode_reward += reward
        
        if done:
            state, _ = env.reset()
            # all_rewards.append(episode_reward)
            progress_bar.set_postfix({'episode': episode, 'reward': f'{episode_reward:.2f}', 'epsilon': f'{epsilon:.3f}'})
            episode_reward = 0
            episode += 1
        
        if len(replay_buffer) > learning_starts and frame_idx % update_frequency == 0:
            batch = replay_buffer.sample(batch_size)
            loss = compute_loss(batch, current_model, target_model, device, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # losses.append(loss.item())
        
        if frame_idx % target_update_frequency == 0:
            target_model.load_state_dict(current_model.state_dict())
        progress_bar.update(1)
    
    progress_bar.close()
    env.close()
    
    # Wyświetlenie statystyk – średnia nagroda z ostatnich 10 epizodów
    # if all_rewards:
    #     mean_reward = np.mean(all_rewards[-10:])
    #     print(f"Średnia nagroda z ostatnich 10 epizodów: {mean_reward:.2f}")
    
    # Zapis modelu
    torch.jit.save(current_model, args.save_path)
    print(f"Model zapisany do: {args.save_path}")
    
    # Zapisujemy wyniki do plików (opcjonalnie)
    # np.save("episode_rewards.npy", np.array(all_rewards))
    # np.save("losses.npy", np.array(losses))
    # print("Wyniki treningu zapisane do plików: episode_rewards.npy oraz losses.npy")
    
    return all_rewards, losses

# ===========================================
# Funkcja ewaluacji (renderingu) wytrenowanego modelu
# ===========================================
def render(args):
    env_name = args.env_name
    # Tworzymy środowisko z render_mode ustawionym na "human"
    env = gym.make(env_name, render_mode="human")
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # Ładujemy zapisany model
    model = torch.jit.load(args.save_path, map_location=device)
    model.eval()
    
    num_episodes = args.num_episodes
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # W trybie ewaluacji stosujemy greedy policy (epsilon = 0)
            action = select_action(state, model, epsilon=0.0, num_actions=int(env.action_space.n), device=device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            # W niektórych wersjach gymnasium render jest wywoływany automatycznie, ale możemy też jawnie wywołać:
            env.render()
        print(f"Episode {ep} Reward: {episode_reward}")
    env.close()

# ===========================================
# Główna część – parsowanie argumentów i wybór trybu
# ===========================================
def main():
    parser = argparse.ArgumentParser(description="DQN Atari - Trening i Rendering")
    parser.add_argument("--mode", choices=["train", "render"], default="train", help="Wybierz tryb: train lub render")
    parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4", help="Nazwa środowiska Atari")
    parser.add_argument("--num_frames", type=int, default=1000000, help="Liczba klatek treningowych")
    parser.add_argument("--replay_buffer_capacity", type=int, default=100000, help="Liczba klatek treningowych")
    parser.add_argument("--batch_size", type=int, default=32, help="Rozmiar partii")
    parser.add_argument("--save_path", type=str, default="dqn_model.pt", help="Ścieżka zapisu/ładowania modelu")
    parser.add_argument("--num_episodes", type=int, default=10, help="Liczba epizodów do renderingu")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "render":
        render(args)

if __name__ == "__main__":
    main()
