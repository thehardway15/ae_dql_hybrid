import copy
import gc
import os
import random
import time
import torch
import psutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from lib.config import Config
from lib.environ import Environment, make_env
from lib.metrics import Metrics
from collections import namedtuple
import torch.multiprocessing as mp
from lib.utils import ReplayBuffer


MAX_SEED = np.iinfo(np.int32).max

def log_memory():
    print(f"RAM użycie: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
    print(f"VRAM użycie: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")


class Individual:
    def __init__(self, seeds: list[int], parameters: dict):
        self.seeds = seeds
        self.parameters = parameters

    def copy(self):
        return Individual(self.seeds, copy.deepcopy(self.parameters))

OutputItem = namedtuple('OutputItem', ['individual', 'fitness', 'frames', 'rb'])


class HybridAgent:
    def __init__(self, config: Config, model_class,  device: str, checkpoints: int = None, path: str = None):
        self.config = config
        self.model_class = model_class
        self.total_frames = 0
        self.history = Metrics()
        self.device = device
        self.replay_buffer = ReplayBuffer(config.hybrid_replay_buffer_capacity, device)
        self.checkpoints = checkpoints
        self.path = path


    def save_model(self, net: nn.Module, path: str):
        torch.save(net.state_dict(), os.path.join(path, 'model.pt'))

    def save_history(self, path: str):
        self.history.save(path)
        self.history.summary(path, 
                            plots=['frames_per_episode', 'reward_avg', 'reward_max', 'reward_std', 'speed', 'memory_usage', 'replay_buffer_size', 'gradient_count'],
                            additional_stats=['frames_last / total_time_last'], plot_compress=1000)

    def _compute_loss(self, batch, model, target_model):
        states, actions, rewards, next_states, dones = batch
        
        q_values = model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = target_model(next_states)
            next_q_value = next_q_values.max(dim=1).values

        expected_q_value = rewards + self.config.gamma * next_q_value * (1. - dones)

        loss = nn.MSELoss()(q_value, expected_q_value.detach())
        return loss
    
    def _train_gradient(self, individual):
        env = make_env(self.config)
        target_model = self._make_net(individual, env)
        model = copy.deepcopy(target_model)

        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        for _ in range(self.config.hybrid_epochs):
            batch = self.replay_buffer.sample(self.config.batch_size)
            loss = self._compute_loss(batch, model, target_model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # TODO: delta_min, delta_step per layer
        compressed_wights = {}
        for name, param in target_model.named_parameters():
            tuned_param = dict(model.named_parameters())[name]
            delta_W = tuned_param.data - param.data
            compressed_wights[name], delta_min, delta_step = self._sparsify_and_quantize(delta_W)

        env.close()
        del env

        individual.parameters.append((compressed_wights, delta_min.to('cpu').numpy(), delta_step.to('cpu').numpy()))

    # TODO: zweryfikować obliczenia
    def _sparsify_and_quantize(self, weights):
        delta_min, delta_max = weights.min(), weights.max()
        delta_step = (delta_max - delta_min) / self.config.hybrid_quantize_count

        sparse_weights = weights.clone()
        sparse_weights[torch.abs(weights) < self.config.hybrid_epsilon] = 0

        quantized_weight = torch.round((sparse_weights - delta_min) / delta_step) + 1

        nonzero_indices = (sparse_weights != 0).to('cpu').nonzero(as_tuple=True)
        values = quantized_weight[nonzero_indices]

        nonzero_indices = [x.numpy() for x in nonzero_indices]
        # delta_min = delta_min.to('cpu').numpy()
        # delta_step = delta_step.to('cpu').numpy()

        return (nonzero_indices, values.tolist()), delta_min, delta_step

    def _make_net(self, individual: Individual, env: Environment = None):
        if env is None:
            env = make_env(self.config)

        seeds = individual.seeds
        torch.manual_seed(seeds[0])
        net = self.model_class(env.observation_space.shape, env.action_space.n)
        net = net.to(self.device)

        for seed in seeds[1:]:
            self._apply_mutations(net, seed)
        
        for parameter in individual.parameters:
            self._apply_parameters(net, parameter)
                
        return net

    def _apply_parameters(self, net, parameter):
        for name, param in net.named_parameters():
            compressed_weights, delta_min, delta_step = parameter
            reconstructed_weights = torch.zeros_like(param.data)

            delta_min = torch.tensor(delta_min).to(self.device)
            delta_step = torch.tensor(delta_step).to(self.device)
            reconstructed_weights = reconstructed_weights.to(self.device)

            indexes, values = compressed_weights[name]
            values = torch.tensor(values).to(self.device)

            indexes = [torch.tensor(x).to(self.device) for x in indexes]

            if isinstance(indexes, torch.Tensor):
                reconstructed_weights[indexes] = delta_min + (values - 1) * delta_step
            else:
                for i, v in enumerate(values):
                    indicies = []
                    for j in range(len(indexes)):
                        indicies.append(indexes[j][i])
                    reconstructed_weights[indicies] = delta_min + (v - 1) * delta_step

            param.data = net.state_dict()[name] + (reconstructed_weights * self.config.hybrid_scale)
    
    def _apply_mutations(self, net, seed):
        noise_std = torch.tensor(self.config.noise_std).to(self.device)
        np.random.seed(seed)
        for p in net.parameters():
            noise = np.random.normal(size=p.data.size())
            noise_t = torch.FloatTensor(noise).to(self.device)
            p.data += noise_std * noise_t
        
    def _evaluate(self, net: nn.Module, env: Environment):
        state, _ = env.reset()
        episode_reward = 0
        frames = 0
        net = net.to(self.device)
        rb = ReplayBuffer(self.config.hybrid_replay_buffer_capacity, self.device)

        while not env.done:
            action = net(torch.FloatTensor(state).unsqueeze(0).to(self.device)).argmax().item()
            next_state, reward, _ = env.step(action)
            state = next_state
            episode_reward += reward
            frames += 1
            rb.push(state, action, reward, next_state.copy(), env.done)
        
        env.close()

        buffer = rb.buffer
        rb.buffer = []
        del rb
        return episode_reward, frames, buffer
    
    def _evaluate_worker(self, input_queue, output_queue):
        env = make_env(self.config)

        while True:
            parents = input_queue.get()
            if parents is None:
                env.close()
                print("Worker closed")
                break

            for individual in parents:
                net = self._make_net(individual, env)

                reward, frames, buffer = self._evaluate(net, env)
                output_queue.put(OutputItem(individual, reward, frames, buffer))
            
        del env

    def debug(self):
        env = make_env(self.config)
        individual = Individual(seeds=(np.random.randint(MAX_SEED),), parameters=[])
        net = self._make_net(individual, env)
        for _ in range(100):
            _, _, rb = self._evaluate(net, env)
            self.replay_buffer.merge(rb)
        self._train_gradient(individual)
        self._make_net(individual, env)

    def train(self, epochs: int):
        mp.set_start_method('spawn', force=True)
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
            individuals = [Individual(seeds=(np.random.randint(MAX_SEED),), parameters=[]) for _ in range(SEEDS_PER_WORKER)]
            input_queue.put(individuals)
        
        elite = None

        t_start = time.time()
        while self.total_frames < epochs:
            gc.collect()
            torch.cuda.empty_cache()
            log_memory()
            print(f"Replay buffer memory usage: {self.replay_buffer.memory_usage()} GB")
            print(f"Replay buffer size: {self.replay_buffer.size()}")

            epoch_start = time.time()
            batch_step = 0
            population = []

            while (len(population) < SEEDS_PER_WORKER * self.config.worker_count):
                output_item = output_queues.get()
                population.append((output_item.individual, output_item.fitness, output_item.frames, output_item.rb))
                batch_step += output_item.frames
            
            if elite is not None:
                population.append(elite)

            population.sort(key=lambda x: x[1], reverse=True)
            rewards = [x[1] for x in population[:self.config.parent_count]]
            reward_max = np.max(rewards)
            reward_avg = np.mean(rewards)
            reward_std = np.std(rewards)

            speed = batch_step / (time.time() - epoch_start)

            progress_bar.set_postfix({'AVG fitness': reward_avg, 'Max fitness': reward_max, 'Episode': episode, 'Speed': speed})
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
                
            for individual in population[:self.config.parent_count][::-1]:
                self.replay_buffer.merge(individual[3])
            
            self.history.add('memory_usage', self.replay_buffer.memory_usage())
            self.history.add('replay_buffer_size', self.replay_buffer.size())
                
            if episode > 0 and episode % self.config.hybrid_gradient_frequency == 0:
                self._train_gradient(elite[0])

            self.total_frames += batch_step
            gradient_count = 0

            for i, worker_queue in enumerate(input_queues):
                individuals = []
                if i == 0:
                    individuals.append(elite[0].copy())
                while len(individuals) < SEEDS_PER_WORKER:
                    parent = np.random.randint(self.config.parent_count)
                    s = list(population[parent][0].seeds)
                    if random.random() < self.config.mutation_rate:
                        next_seed = np.random.randint(MAX_SEED)
                        s.append(next_seed)
                    individuals.append(Individual(seeds=tuple(s), parameters=copy.deepcopy(population[parent][0].parameters)))
                gradient_count += len(list(filter(lambda x: len(x.parameters) > 0, individuals)))
                worker_queue.put(individuals)
            progress_bar.update(batch_step)
            episode += 1

            self.history.add('gradient_count', gradient_count)

        for worker_queue in input_queues:
            worker_queue.put(None)

        for queue in input_queues:
            queue.close()
            queue.cancel_join_thread()

        for worker in workers:
            worker.terminate()
            worker.join()

        training_time = time.time() - t_start
        self.history.add('total_time', training_time)
        self.history.add('frames', self.total_frames)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Training finished after {episode} episodes")
        print(f"Total frames: {self.total_frames}")
        print(f"Elite gradient count: {len(elite[0].parameters)}")
        return self._make_net(elite[0])
