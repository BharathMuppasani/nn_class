import numpy as np
import torch
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        
    def push(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)

        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.stack([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-5
        self.max_priority = 1.0

        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)

        experience = Experience(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_idx] = experience

        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.capacity

    def get_probabilities(self):
        scaled_priorities = self.priorities[:len(self.buffer)] ** self.alpha
        return scaled_priorities / np.sum(scaled_priorities)

    def get_importance_weights(self, probs):
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        weights = (len(self.buffer) * probs) ** (-beta)
        return torch.FloatTensor(weights / weights.max()).to(self.device)

    def sample(self, batch_size):
        probs = self.get_probabilities()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = self.get_importance_weights(probs[indices])

        experiences = [self.buffer[idx] for idx in indices]

        states = torch.FloatTensor(np.stack([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).unsqueeze(1).to(self.device)

        self.frame += 1
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = np.minimum(td_errors, 100.0)
        self.max_priority = max(self.max_priority, np.max(td_errors))

    def __len__(self):
        return len(self.buffer)