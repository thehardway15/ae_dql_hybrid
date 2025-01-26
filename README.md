# DQN Agent for Atari Breakout

This project implements a Deep Q-Network (DQN) agent with prioritized experience replay to play Atari Breakout. The agent uses a convolutional neural network to learn directly from frame pixels.

## Features

- Deep Q-Network with prioritized experience replay
- Frame stacking (4 frames) for temporal information
- TensorBoard integration for monitoring training progress
- Progress bar with real-time metrics
- Model saving and loading capabilities
- Evaluation during training
- Ability to watch the trained agent play

## Requirements

```bash
pip install -r requirements.txt
```

## Training the Agent

To start training the agent, run:

```bash
python main.py
```

The training script will:
- Show a progress bar with current episode, frames processed, and metrics
- Save the model periodically as `dqn_breakout_model.h5`
- Save training results to `training_results.csv`
- Log metrics to TensorBoard

### Monitoring Training with TensorBoard

1. During training, TensorBoard logs are saved to `logs/dqn_breakout/YYYYMMDD-HHMMSS/`
2. To view the training progress, open a new terminal and run:
   ```bash
   tensorboard --logdir logs/dqn_breakout
   ```
3. Open your browser and go to `http://localhost:6006`

You can monitor:
- Episode rewards
- Training loss
- Epsilon value (exploration rate)
- Other training metrics

### Training Parameters

Key hyperparameters (can be modified in `main.py`):
- `MAX_FRAMES`: Total frames to train (default: 500,000)
- `EPSILON_DECAY`: Exploration rate decay (default: 0.999)
- `EPSILON_MIN`: Minimum exploration rate (default: 0.1)
- `LEARNING_RATE`: Learning rate (default: 0.00025)
- `BUFFER_SIZE`: Replay buffer size (default: 100,000)

## Playing with Trained Agent

To watch the trained agent play, uncomment the last line in `main.py` or run:

```python
from main import play_final_agent
play_final_agent("dqn_breakout_model.h5")
```

This will:
- Load the trained model
- Open a window showing the game
- Display the agent playing Breakout
- Print episode rewards

## Model Architecture

The DQN uses a convolutional neural network:
1. Input: 84x84x4 (4 stacked grayscale frames)
2. Conv2D: 32 filters, 8x8 kernel, stride 4, ReLU
3. Conv2D: 64 filters, 4x4 kernel, stride 2, ReLU
4. Conv2D: 64 filters, 3x3 kernel, stride 1, ReLU
5. Dense: 512 units, ReLU
6. Output: 4 units (actions)

## Results

Training results are saved to:
- Model weights: `dqn_breakout_model.h5`
- Training metrics: `training_results.csv`
- TensorBoard logs: `logs/dqn_breakout/YYYYMMDD-HHMMSS/`

A plot of training progress is shown after training completes.
