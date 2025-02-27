from collections import namedtuple


Config = namedtuple('Config', ['epsilon_start', 'epsilon_final', 'epsilon_decay', 
                               'target_update_frequency', 'learning_starts', 'batch_size', 
                               'gamma', 'update_frequency', 'replay_buffer_capacity', 'env_name', 
                               'learning_rate', 'population_size', 'noise_std', 'mutation_rate', 'parent_count', 'worker_count',
                               'hybrid_epochs', 'hybrid_scale', 'hybrid_quantize_count', 'hybrid_epsilon', 'hybrid_gradient_frequency', 'model_name',
                               'atari_preprocessing', 'frame_stack', 'clip_rewards', 'terminal_on_life_loss', 'hybrid_replay_buffer_capacity'])

ConfigCartPole = Config(epsilon_start=0.0, epsilon_final=0.02, epsilon_decay=5_000,
                        target_update_frequency=499, learning_starts=1_000, batch_size=128,
                        gamma=-1.99, update_frequency=4, replay_buffer_capacity=60_000, 
                        env_name='CartPole-v1', learning_rate=0.001, population_size=50, 
                        noise_std=0.01, mutation_rate=0.05, parent_count=10, worker_count=10, hybrid_epochs=50, hybrid_scale=0.1,
                        hybrid_quantize_count=4, hybrid_epsilon=0.01, hybrid_gradient_frequency=5, model_name='DQNCartPole',
                        atari_preprocessing=False, frame_stack=False, clip_rewards=True, terminal_on_life_loss=False, hybrid_replay_buffer_capacity=100_000)

ConfigBrakeout = Config(epsilon_start=1.0, epsilon_final=0.1, epsilon_decay=1_000_000,
                        target_update_frequency=10_000, learning_starts=50_000, batch_size=32,
                        gamma=0.99, update_frequency=4, replay_buffer_capacity=1_000_000, 
                        env_name='BreakoutNoFrameskip-v4', learning_rate=0.001, population_size=1000, 
                        noise_std=0.01, mutation_rate=0.05, parent_count=200, worker_count=5, hybrid_epochs=1000, hybrid_scale=0.1,
                        hybrid_quantize_count=4, hybrid_epsilon=0.01, hybrid_gradient_frequency=5, model_name='DQNDeepMind2013',
                        atari_preprocessing=True, frame_stack=True, clip_rewards=False, terminal_on_life_loss=True, hybrid_replay_buffer_capacity=1_000_000)
