# Automated Environment DQN Learning (AE-DQL)

A deep reinforcement learning implementation using Deep Q-Learning (DQL) for automated environment interaction and learning. This project focuses on training agents to perform optimally in various Gymnasium environments, with a particular emphasis on CartPole and Atari games.

## Features

- Deep Q-Network (DQN) implementation with experience replay
- Support for multiple environments (CartPole, Atari games)
- Configurable hyperparameters and network architecture
- Real-time training metrics and visualization
- Model saving and loading capabilities
- Environment rendering for visual inspection

## Requirements

```bash
# Core dependencies
gymnasium[atari,accept-rom-license,classic-control,other]
ale-py
torch
torchvision
pytorch-ignite

# Visualization and utilities
pandas
matplotlib
tqdm
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thehardway15/ae_dql_hybrid.git
cd ae_dql_hybrid
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a new DQN model:

```bash
python main.py --mode train \
               --model_path model_save.pt \
               --version gradient \
               --epochs 100000 \
               --model_name DQNCartPole \
               --config ConfigCartPole
```

Available parameters:
- `--mode`: Choose between "train" or "render"
- `--model_path`: Path to save/load the model (default: "dqn_model.pt")
- `--version`: Training version - "gradient" or "ae" (default: "gradient")
- `--epochs`: Number of training epochs (default: 100000)
- `--model_name`: Name of the model class to use (default: "DQNCartPole")
- `--config`: Name of the configuration class to use (default: "ConfigCartPole")

### Rendering

To visualize a trained agent's performance:

```bash
python main.py --mode render \
               --model_path model_save.pt \
               --model_name DQNCartPole \
               --config ConfigCartPole
```

## Project Structure

```
ae_dql_hybrid/
├── lib/
│   ├── agents/         # Agent implementations (DQN)
│   ├── model/          # Neural network architectures
│   ├── environ/        # Environment wrappers
│   ├── config.py       # Configuration parameters
│   ├── metrics.py      # Performance tracking
│   └── utils.py        # Utility functions
├── main.py             # Main training/rendering script
└── requirements.txt    # Project dependencies
```

## Configuration

The project uses configuration files in `lib/config.py` to manage hyperparameters. Key parameters include:

- Learning rate
- Epsilon decay (exploration vs exploitation)
- Replay buffer size
- Batch size
- Network architecture

## Results

Training results and metrics are saved in the following format:
- Model weights: `.pt` files
- Training metrics: JSON format
- Performance plots: Generated using matplotlib

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) for the environment implementations
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for inspiration on DQN implementation
