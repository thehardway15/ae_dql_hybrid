import argparse
import gymnasium as gym
import ale_py
import torch
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from gradient import DQN

gym.register_envs(ale_py)


def render(args):
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = MaxAndSkipEnv(env, skip=4)

    torch.serialization.add_safe_globals([DQN])
    model = torch.load(args.model_path, weights_only=False)
    model = model.to('cuda')
    model.eval()

    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to('cuda')
            action = model(state).argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        episode_reward += reward
        env.render()
    
    print(f"Episode Reward: {episode_reward}")
    env.close()
    

def main():
    parser = argparse.ArgumentParser(description="DQN Atari - Trening i Rendering")
    parser.add_argument("--model_path", type=str, default="dqn_model.pt", help="Ścieżka zapisu/ładowania modelu")
    parser.add_argument("--env_name", type=str, default="BreakoutNoFrameskip-v4", help="Nazwa środowiska")
    
    args = parser.parse_args()
    
    render(args)
    
if __name__ == "__main__":
    main()