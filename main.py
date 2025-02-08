import torch
import argparse
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
                          plots=['frames_per_episode', 'reward_per_episode', 'time_per_episode', 'loss'],
                          additional_stats=['frames_last / total_time_last'])


def train(args):
    version = args.version

    if version == "gradient":
        train_gradient(args)
    elif version == "ae":
        pass
    else:
        raise ValueError(f"Nieznana wersja: {version}")


def render(args):
    model_class = getattr(models, args.model_name)
    config_class = getattr(config, args.config)

    env = Environment(config_class.env_name, render='human')
    model = model_class(env.observation_space.shape, env.action_space.n)
    agent = DQNAgent(config_class, env, model, device)
    agent.load_model(args.model_path)
    agent.play()


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
