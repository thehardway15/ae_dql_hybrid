import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

def make_env(config, render='rgb_array'):
    env_name = config.env_name
    atari_preprocessing = config.atari_preprocessing
    frame_stack = config.frame_stack
    clip_rewards = config.clip_rewards
    terminal_on_life_loss = config.terminal_on_life_loss

    return Environment(env_name, atari_preprocessing, frame_stack, clip_rewards, terminal_on_life_loss, render)
    

class Environment:
    def __init__(self, env_name, atari_preprocessing=False, frame_stack=False, 
                 clip_rewards=True, terminal_on_life_loss=False, render='rgb_array'):
        self.env = gym.make(env_name, render_mode=render)
        if atari_preprocessing:
            self.env = gym.wrappers.AtariPreprocessing(self.env, grayscale_obs=True, scale_obs=False, frame_skip=4)
        if frame_stack:
            self.env = gym.wrappers.FrameStackObservation(self.env, stack_size=4)
        
        self.total_reward = 0
        self.done = False
        self.frame_count = 0
        self.clip_rewards = clip_rewards
        self.terminal_on_life_loss = terminal_on_life_loss
        self.lives = 0

    def reset(self):
        self.total_reward = 0
        self.done = False
        self.frame_count = 0
        state, info = self.env.reset()
        if self.terminal_on_life_loss:
            self.lives = info['lives']
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        self.done = terminated or truncated

        if self.terminal_on_life_loss:
            if info['lives'] < self.lives:
                self.done = True

        self.frame_count += 1

        if self.clip_rewards:
            reward = np.clip(reward, -1, 1)

        return state, reward, info

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space
