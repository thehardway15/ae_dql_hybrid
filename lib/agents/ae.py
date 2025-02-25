import copy
import os
import random
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from lib.config import Config
from lib.environ import Environment, make_env
from lib.metrics import Metrics
from collections import namedtuple
import torch.multiprocessing as mp


MAX_SEED = np.iinfo(np.int32).max

OutputItem = namedtuple('OutputItem', ['seeds', 'fitness', 'frames'])


class AEAgent:
    def __init__(self, config: Config, model_class,  device: str, checkpoints: int = 1000, path: str = None):
        self.config = config
        self.model_class = model_class
        self.total_frames = 0
        self.history = Metrics()
        self.device = device
        self.checkpoints = checkpoints
        self.path = path

    def save_model(self, net: nn.Module, path: str):
        torch.save(net.state_dict(), os.path.join(path, 'model.pt'))

    def save_history(self, path: str):
        self.history.save(path)
        self.history.summary(path, 
                          plots=['frames_per_episode', 'reward_avg', 'reward_max', 'reward_std', 'speed'],
                          additional_stats=['frames_last / total_time_last'])       

    def _make_net(self, seeds: list[int], env: Environment = None):
        if env is None:
            env = make_env(self.config)

        torch.manual_seed(seeds[0])
        net = self.model_class(env.observation_space.shape, env.action_space.n)
        net = net.to(self.device)

        for seed in seeds[1:]:
            net = self._mutate(net, seed, copy_net=True)
                
        return net
    
    def _mutate(self, net, seed, copy_net=False):
        new_net = copy.deepcopy(net) if copy_net else net
        noise_std = torch.tensor(self.config.noise_std).to(self.device)
        np.random.seed(seed)
        for p in new_net.parameters():
            noise = np.random.normal(size=p.data.size())
            noise_t = torch.FloatTensor(noise).to(self.device)
            p.data += noise_std * noise_t
        return new_net
        
    def _evaluate(self, net: nn.Module, env: Environment):
        state, _ = env.reset()
        episode_reward = 0
        frames = 0
        net = net.to(self.device)

        while not env.done:
            action = net(torch.FloatTensor(state).to(self.device)).argmax().item()
            next_state, reward, _ = env.step(action)
            state = next_state
            episode_reward += reward
            frames += 1
        
        env.close()
        return episode_reward, frames

    
    def _evaluate_worker(self, input_queue, output_queue):
        env = make_env(self.config)
        cache = {}

        while True:
            parents = input_queue.get()
            if parents is None:
                break

            new_cache = {}
            for net_seeds in parents:
                if len(net_seeds) > 1:
                    net = cache.get(net_seeds[:-1])
                    if net is not None:
                        net = self._mutate(net, net_seeds[-1], copy_net=True)
                    else:
                        net = self._make_net(net_seeds, env)
                else:
                    net = self._make_net(net_seeds, env)
                new_cache[net_seeds] = net
                reward, frames = self._evaluate(net, env)
                output_queue.put(OutputItem(net_seeds, reward, frames))
            
            cache = new_cache

    def train(self, epochs: int):
        mp.set_start_method('forkserver')
        progress_bar = tqdm(range(epochs), desc="Training")

        SEEDS_PER_WORKER = self.config.population_size // self.config.worker_count

        input_queues= []
        workers = []

        episode = 0

        output_queues = mp.Queue(maxsize=self.config.worker_count)

        for _ in range(self.config.worker_count):
            input_queue = mp.Queue(maxsize=1)
            input_queues.append(input_queue)
            worker = mp.Process(target=self._evaluate_worker, args=(input_queue, output_queues))
            workers.append(worker)
            worker.start()
            seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
            input_queue.put(seeds)
        
        elite = None

        t_start = time.time()
        while self.total_frames < epochs:
            epoch_start = time.time()
            batch_step = 0
            population = []

            while (len(population) < SEEDS_PER_WORKER * self.config.worker_count):
                output_item = output_queues.get()
                population.append((output_item.seeds, output_item.fitness, output_item.frames))
                batch_step += output_item.frames
            
            if elite is not None:
                population.append(elite)

            population.sort(key=lambda x: x[1], reverse=True)
            rewards = [x[1] for x in population[:self.config.parent_count]]
            reward_max = np.max(rewards)
            reward_avg = np.mean(rewards)
            reward_std = np.std(rewards)

            speed = batch_step / (time.time() - epoch_start)

            progress_bar.set_postfix({'AVG fitness': reward_avg, 'Max fitness': reward_max, 'Reward std': reward_std, 'Speed': speed})
            self.history.add('reward_avg', reward_avg)
            self.history.add('reward_max', reward_max)
            self.history.add('reward_std', reward_std)
            self.history.add('speed', speed)
            self.history.add('frames_per_episode', batch_step)
            self.history.add('time_per_episode', time.time() - epoch_start)

            elite = population[0]

            if self.checkpoints is not None and episode > 0 and episode % self.checkpoints == 0:
                checkpoint_path = os.path.join(self.path, f'checkpoint_{episode}')
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                self.save_model(self._make_net(elite[0]), checkpoint_path)
                self.save_history(checkpoint_path)

            self.total_frames += batch_step

            for i, worker_queue in enumerate(input_queues):
                seeds = []
                if i == 0:
                    seeds.append(elite[0])

                while len(seeds) < SEEDS_PER_WORKER:
                    parent = np.random.randint(self.config.parent_count)
                    s = list(population[parent][0])
                    if random.random() < self.config.mutation_rate:
                        next_seed = np.random.randint(MAX_SEED)
                        s.append(next_seed)
                    seeds.append(tuple(s))
                worker_queue.put(seeds)
            progress_bar.update(batch_step)
            episode += 1

        for worker in workers:
            worker.terminate()
            worker.join()

        training_time = time.time() - t_start
        self.history.add('total_time', training_time)
        self.history.add('frames', self.total_frames)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Training finished after {episode} episodes")
        print(f"Total frames: {self.total_frames}")
        return self._make_net(elite[0])
