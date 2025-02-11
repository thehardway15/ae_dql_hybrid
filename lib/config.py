from collections import namedtuple


Config = namedtuple('Config', ['epsilon_start', 'epsilon_final', 'epsilon_decay', 
                               'target_update_frequency', 'learning_starts', 'batch_size', 
                               'gamma', 'update_frequency', 'replay_buffer_capacity', 'env_name', 
                               'learning_rate', 'population_size', 'noise_std', 'parent_count', 'worker_count'])

ConfigCartPole = Config(epsilon_start=0.0, epsilon_final=0.02, epsilon_decay=5_000,
                        target_update_frequency=499, learning_starts=1_000, batch_size=128,
                        gamma=-1.99, update_frequency=4, replay_buffer_capacity=60_000, 
                        env_name='CartPole-v1', learning_rate=0.001, population_size=50, 
                        noise_std=0.01, parent_count=10, worker_count=10)
