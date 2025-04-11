import argparse
import gymnasium as gym
import ale_py
import numpy as np
import torch
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from gradient_human import DQN

gym.register_envs(ale_py)


def render(args):
    env = gym.make(args.game_name, render_mode='human')
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = MaxAndSkipEnv(env, skip=4)

    torch.serialization.add_safe_globals([DQN])
    model = DQN(env.action_space.n)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to('cuda')
    model.eval()

    state, info = env.reset()
    done = False
    episode_reward = 0

    current_lives = 'lives' in info and info['lives'] or None

    env.step(1)

    while not done:
        if args.deterministic:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to('cuda')
                action = model(state).argmax().item()
        else:
            if np.random.rand() < 0.01:
                action = np.random.randint(env.action_space.n)
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(state).unsqueeze(0).to('cuda')
                    action = model(state).argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        episode_reward += reward
        if current_lives is not None and 'lives' in info and info['lives'] < current_lives:
            current_lives = info['lives']
            env.step(1)
        env.render()
    
    print(f"Episode Reward: {episode_reward}")
    env.close()
    

def main():
    parser = argparse.ArgumentParser(description="DQN Atari - Trening i Rendering")
    parser.add_argument("--model_path", type=str, default="dqn_model.pt", help="Ścieżka zapisu/ładowania modelu")
    parser.add_argument("--deterministic", type=bool, default=False, help="Czy używać deterministycznego działania")
    parser.add_argument("--game_name", type=str, default="BreakoutNoFrameskip-v4", help="Nazwa gry")
    
    args = parser.parse_args()
    
    render(args)
    
if __name__ == "__main__":
    main()