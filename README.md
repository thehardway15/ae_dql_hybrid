python main.py --mode train --env_name PongNoFrameskip-v4 --num_frames 1000000 --save_path dqn_model.pt

python main.py --mode render --env_name PongNoFrameskip-v4 --save_path dqn_model.pt --num_episodes 10
