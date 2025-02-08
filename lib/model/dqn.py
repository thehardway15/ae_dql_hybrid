import torch
import torch.nn as nn
import numpy as np

class DQNHuman(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQNHuman, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_dim)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


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
