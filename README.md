# Automated Environment DQN Learning (AE-DQL)

A deep reinforcement learning implementation that combines Deep Q-Learning (DQL) with Algorithm Evolution (AE) for automated environment interaction and learning. This project focuses on training agents using both gradient-based (DQN) and evolutionary approaches to perform optimally in various Gymnasium environments, with a particular emphasis on CartPole and Atari games.

## Features

- Dual learning approaches:
  - Deep Q-Network (DQN) with experience replay (gradient-based learning)
  - Algorithm Evolution (AE) with population-based optimization
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

# Development tools
ipdb
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

The project supports two training approaches:

1. Gradient-based DQN training:
```bash
python main.py --mode train \
               --version gradient \
               --model_path dqn_model.pt \
               --epochs 100000 \
               --model_name DQNCartPole \
               --config ConfigCartPole
```

2. Algorithm Evolution training:
```bash
python main.py --mode train \
               --version ae \
               --model_path ae_model.pt \
               --epochs 100 \
               --model_name DQNCartPole \
               --config ConfigAE
```

Available parameters:
- `--mode`: Choose between "train" or "render"
- `--version`: Training version - "gradient" (DQN) or "ae" (Algorithm Evolution)
- `--model_path`: Path to save/load the model (default: "dqn_model.pt")
- `--epochs`: Number of training epochs
  - For DQN: typically 100000
  - For AE: typically 100 (generations)
- `--model_name`: Name of the model class to use (default: "DQNCartPole")
- `--config`: Name of the configuration class to use
  - For DQN: "ConfigCartPole"
  - For AE: "ConfigAE"

### Rendering

To visualize a trained agent's performance:

```bash
python main.py --mode render \
               --version ae \  # or gradient
               --model_path model_save.pt \
               --model_name DQNCartPole \
               --config ConfigCartPole  # or ConfigAE
```

## Project Structure

```
ae_dql_hybrid/
├── lib/
│   ├── agents/         # Agent implementations (DQN and AE)
│   ├── model/          # Neural network architectures
│   ├── environ/        # Environment wrappers
│   ├── config.py       # Configuration parameters
│   ├── metrics.py      # Performance tracking
│   └── utils.py        # Utility functions
├── cartpol_result/     # DQN training results
├── cartpol_ae_result/  # Algorithm Evolution results
├── main.py             # Main training/rendering script
└── requirements.txt    # Project dependencies
```

## Configuration

The project uses configuration files in `lib/config.py` to manage hyperparameters:

DQN Configuration:
- Learning rate
- Epsilon decay (exploration vs exploitation)
- Replay buffer size
- Batch size
- Network architecture

Algorithm Evolution Configuration:
- Population size
- Parent count for selection
- Mutation noise standard deviation
- Network architecture

## Results

Training results and metrics are saved in separate directories for each approach:
- DQN results: `cartpol_result/`
- Algorithm Evolution results: `cartpol_ae_result/`

Each directory contains:
- Model weights: `.pt` files
- Training metrics: JSON format
- Performance plots: Generated using matplotlib (`.png` files)

The metrics tracked include:
- Episode rewards
- Training time
- Frame count
- For AE: 
  - Population fitness statistics
  - Best individual performance
  - Generation statistics

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
