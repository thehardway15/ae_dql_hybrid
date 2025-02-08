from collections import deque
import zlib
import sys
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, device: str, compress: bool = False):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.compress = compress
        self.dtype = None

    def push(self, state, action, reward, next_state, done):
        if self.dtype is None:
            self.dtype = state.dtype
        if self.compress:
            state = zlib.compress(state)
            next_state = zlib.compress(next_state)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        if self.compress:
            states = [np.frombuffer(zlib.decompress(state), dtype=self.dtype) for state in states]
            next_states = [np.frombuffer(zlib.decompress(next_state), dtype=self.dtype) for next_state in next_states]

        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(dones).to(self.device),
        )

    # memory usage in GB    
    def memory_usage(self):
        return sum(sys.getsizeof(state) for state in self.buffer) / 1024 ** 3
    
    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)
