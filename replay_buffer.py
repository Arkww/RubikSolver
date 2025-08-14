from collections import deque, namedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
import random

class ReplayBuffer:
    """u
    Experience replay buffer - stores past experiences for learning
    """
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        """Store an experience"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch for training"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)