# Automated Environment DQN Learning (AE-DQL)

A deep reinforcement learning implementation that combines Deep Q-Learning (DQL) with Algorithm Evolution (AE) for automated environment interaction and learning. This project features three distinct training approaches: gradient-based DQN, evolutionary optimization (AE), and a hybrid approach combining both methods. The implementation is tested on various Gymnasium environments, with a particular emphasis on CartPole and Atari games.

## Features

- Three learning approaches:
  - Deep Q-Network (DQN) with experience replay (gradient-based learning)
  - Algorithm Evolution (AE) with population-based optimization
  - Hybrid approach combining DQN and AE
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

The project supports three training approaches:

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

3. Hybrid DQN-AE training:
```bash
python main.py --mode train \
               --version hybrid \
               --model_path hybrid_model.pt \
               --epochs 100 \
               --model_name DQNCartPole \
               --config ConfigHybrid
```

Available parameters:
- `--mode`: Choose between "train" or "render"
- `--version`: Training version - "gradient" (DQN), "ae" (Algorithm Evolution), or "hybrid"
- `--model_path`: Path to save/load the model (default: "dqn_model.pt")
- `--epochs`: Number of training epochs
  - For DQN: typically 100000
  - For AE/Hybrid: typically 100 (generations)
- `--model_name`: Name of the model class to use (default: "DQNCartPole")
- `--config`: Name of the configuration class to use
  - For DQN: "ConfigCartPole"
  - For AE: "ConfigAE"
  - For Hybrid: "ConfigHybrid"

### Rendering

To visualize a trained agent's performance:

```bash
python main.py --mode render \
               --version gradient \  # or "ae" or "hybrid"
               --model_path model_save.pt \
               --model_name DQNCartPole \
               --config ConfigCartPole  # or appropriate config
```

## Project Structure

```
ae_dql_hybrid/
├── lib/
│   ├── agents/         # Agent implementations (DQN, AE, and Hybrid)
│   │   ├── ae.py      # Algorithm Evolution agent
│   │   ├── hybrid.py  # Hybrid DQN-AE agent
│   │   └── dqn.py     # DQN agent
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

Hybrid Configuration:
- Combines parameters from both DQN and AE
- Additional hybrid-specific parameters for balancing both approaches

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
- For AE and Hybrid: 
  - Population fitness statistics
  - Best individual performance
  - Generation statistics
  - Average/Max/Std rewards per epoch
  - Training speed metrics

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
