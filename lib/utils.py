from collections import deque
import copy
import zlib
import sys
import random
import numpy as np
import torch
import time

class ReplayBuffer:
    def __init__(self, capacity: int, device: str, compress: bool = False):
        self.buffer = []
        self.device = device
        self.compress = compress
        self.dtype = None
        self.capacity = capacity

    def _get_tensor_memory(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor.nbytes
        elif isinstance(tensor, torch.Tensor):
            return tensor.element_size() * tensor.nelement()
        return 0

    def push(self, state, action, reward, next_state, done):
        if self.dtype is None:
            self.dtype = state.dtype
        if self.compress:
            state_shape = state.shape
            next_state_shape = next_state.shape
            state = (state_shape, zlib.compress(state.tobytes()))
            next_state = (next_state_shape, zlib.compress(next_state.tobytes()))

        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        if self.compress:
            states = [np.frombuffer(zlib.decompress(state[1]), dtype=self.dtype).reshape(state[0]) for state in states]
            next_states = [np.frombuffer(zlib.decompress(next_state[1]), dtype=self.dtype).reshape(next_state[0]) for next_state in next_states]

        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(dones).to(self.device),
        )
    
    def merge(self, buffer):
        self.buffer.extend(copy.deepcopy(buffer))
        self.buffer = self.buffer[-self.capacity:]

    # memory usage in GB    
    def memory_usage(self):
        return sum(self._get_tensor_memory(tensor) for state in self.buffer for tensor in state) / 1024 ** 3
    
    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)
