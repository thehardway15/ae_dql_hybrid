import torch
import torch.nn as nn
import numpy as np

class DQNDeepMind2013(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQNDeepMind2013, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_dim[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)
        

class DQNHuman(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQNHuman, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x


class DQNCartPole(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQNCartPole, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        return self.fc(x)
