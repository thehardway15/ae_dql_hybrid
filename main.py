import sys
import torch
import argparse
from lib.agents.ae import AEAgent
from lib.agents.hybrid import HybridAgent
from lib.environ import Environment
from lib.agents import DQNAgent
import lib.config as config
import lib.model as models
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

def train_gradient(args):
    model_class = getattr(models, args.model_name)
    config_class = getattr(config, args.config)
    
    env = Environment(config_class.env_name)
    model = model_class(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=config_class.learning_rate)
    agent = DQNAgent(config_class, env, model, device, optimizer)
    epochs = args.epochs

    agent.train(epochs)
    agent.save_model(args.model_path)
    agent.history.save(args.model_path.replace('.pt', ''))
    agent.history.summary(args.model_path.replace('.pt', ''), 
                          plots=['frames_per_episode', 'reward_per_episode', 'time_per_episode', 'loss', 'memory_usage', 'replay_buffer_size'],
                          additional_stats=['frames_last / total_time_last'])

                        
def train_ae(args):
    model_class = getattr(models, args.model_name)
    config_class = getattr(config, args.config)
    epochs = args.epochs

    agent = AEAgent(config_class, model_class, device)
    net = agent.train(epochs)
    agent.save_model(net, args.model_path)
    agent.history.save(args.model_path.replace('.pt', ''))
    agent.history.summary(args.model_path.replace('.pt', ''), 
                          plots=['frames_per_epoch', 'reward_avg', 'reward_max', 'reward_std', 'speed'],
                          additional_stats=['frames_last / total_time_last'])
                        
                        
def train_hybrid(args):
    model_class = getattr(models, args.model_name)
    config_class = getattr(config, args.config)
    epochs = args.epochs

    agent = HybridAgent(config_class, model_class, device)
    net = agent.train(epochs)
    agent.save_model(net, args.model_path)
    agent.history.save(args.model_path.replace('.pt', ''))
    agent.history.summary(args.model_path.replace('.pt', ''), 
                          plots=['frames_per_epoch', 'reward_avg', 'reward_max', 'reward_std', 'speed', 'memory_usage', 'replay_buffer_size'],
                          additional_stats=['frames_last / total_time_last'])

def train(args):
    version = args.version

    if version == "gradient":
        train_gradient(args)
    elif version == "ae":
        train_ae(args)
    elif version == "hybrid":
        train_hybrid(args)
    else:
        raise ValueError(f"Nieznana wersja: {version}")


def render(args):
    model_class = getattr(models, args.model_name)
    config_class = getattr(config, args.config)
    env = Environment(config_class.env_name, render='human')
    model = model_class(env.observation_space.shape, env.action_space.n)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = model(state).argmax().item()

        next_state, reward, _ = env.step(action)
        done = env.done
        state = next_state
        episode_reward += reward
        env.render()
    
    print(f"Episode Reward: {episode_reward}")
    env.close()
    

def main():
    parser = argparse.ArgumentParser(description="DQN Atari - Trening i Rendering")
    parser.add_argument("--mode", choices=["train", "render"], default="train", help="Wybierz tryb: train lub render")
    parser.add_argument("--model_path", type=str, default="dqn_model.pt", help="Ścieżka zapisu/ładowania modelu")
    parser.add_argument("--version", type=str, default="gradient", help="Wersja treningu")
    parser.add_argument("--epochs", type=int, default=100_000, help="Liczba epok treningu")
    parser.add_argument("--model_name", type=str, default="DQNCartPole", help="Nazwa modelu")
    parser.add_argument("--config", type=str, default="ConfigCartPole", help="Nazwa konfiguracji")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "render":
        render(args)
    
if __name__ == "__main__":
    main()
