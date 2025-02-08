import torch
import argparse
from lib.environ import Environment
from lib.agents import DQNAgent, ConfigCartPole
from lib.model import DQNCartPole
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

def train(args):
    env = Environment(ConfigCartPole.env_name)
    model = DQNCartPole(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=ConfigCartPole.learning_rate)
    agent = DQNAgent(ConfigCartPole, env, model, device, optimizer)

    epoch = 0
    rewards = []
    smoothed_rewards = []
    smoothed_losses = []

    epochs = 100_000

    progress_bar = tqdm(total=epochs, desc='Training Progress')

    while epoch < epochs:
        state, _ = env.reset()
        agent.train_step(state)

        while not agent.env.done:
            epoch += 1
            next_state = agent.train_step(state)
            progress_bar.set_postfix({'total_reward': env.total_reward, 'epsilon': agent.epsilon})
            progress_bar.update(1)

            state = next_state

            if (epoch % 1_000 == 0) and (epoch > 0): 
                smoothed_rewards.append(np.mean(rewards))
                if len(agent.loss_list) > 0:
                    smoothed_losses.append(np.mean(agent.loss_list))
                rewards = []
                agent.loss_list = []
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(smoothed_rewards)
                ax[0].set_title("Average Reward on CartPole")
                ax[0].set_xlabel("Training Epochs")
                ax[0].set_ylabel("Average Reward per Episode")
                ax[1].plot(smoothed_losses)
                ax[1].set_title("Average Loss on CartPole")
                ax[1].set_xlabel("Training Epochs")
                ax[1].set_ylabel("Average Loss per Episode")
                plt.savefig("average_reward_and_loss_on_cartpole.png")
                plt.close()
                        
        rewards.append(env.total_reward)


    agent.save_model(args.model_path)

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
