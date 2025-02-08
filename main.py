import torch
import argparse
from lib.environ import Environment
from lib.agents import DQNAgent
from lib.config import ConfigCartPole
from lib.model import DQNCartPole
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

def train(args):
    env = Environment(ConfigCartPole.env_name)
    model = DQNCartPole(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=ConfigCartPole.learning_rate)
    agent = DQNAgent(ConfigCartPole, env, model, device, optimizer)
    epochs = 100_000

    agent.train(epochs)
    agent.save_model(args.model_path)
    agent.history.save(args.model_path.replace('.pt', ''))
    agent.history.summary(args.model_path.replace('.pt', ''), 
                          plots=['frames_per_episode', 'reward_per_episode', 'time_per_episode', 'loss'],
                          additional_stats=['frames_last / total_time_last'])

def render(args):
    env = Environment(ConfigCartPole.env_name, render='human')
    model = DQNCartPole(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=ConfigCartPole.learning_rate)
    agent = DQNAgent(ConfigCartPole, env, model, device, optimizer)
    agent.load_model(args.model_path)
    agent.play()


def main():
    parser = argparse.ArgumentParser(description="DQN Atari - Trening i Rendering")
    parser.add_argument("--mode", choices=["train", "render"], default="train", help="Wybierz tryb: train lub render")
    parser.add_argument("--model_path", type=str, default="dqn_model.pt", help="Ścieżka zapisu/ładowania modelu")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "render":
        render(args)
    
if __name__ == "__main__":
    main()
