python dqn.py --mode train --env_name PongNoFrameskip-v4 --num_frames 1000000 --replay_buffer_capacity 200000 --save_path dqn_model.pt

python dqn.py --mode render --env_name PongNoFrameskip-v4 --save_path dqn_model.pt --num_episodes 10

python ga.py --mode train --env_name PongNoFrameskip-v4 --population_size 50 --num_generations 100 --save_path ga_dqn_model.pth

python ga.py --mode render --env_name PongNoFrameskip-v4 --save_path ga_dqn_model.pth --num_episodes 10


