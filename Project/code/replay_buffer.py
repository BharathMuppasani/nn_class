import numpy as np
import torch
from collections import namedtuple
import random

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Optimized replay buffer using pre-allocated numpy arrays with uint8 storage."""

    def __init__(self, capacity, device, state_shape=None):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Will be lazily initialized on first push
        self.state_shape = state_shape
        self.states = None
        self.next_states = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self._initialized = False

    def _initialize(self, state):
        """Lazily initialize arrays based on first state."""
        state = np.array(state)
        self.state_shape = state.shape

        # Use uint8 for images (0-255), float32 for low-dim states
        is_image = len(self.state_shape) == 3
        dtype = np.uint8 if is_image else np.float32

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=dtype)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._initialized = True

    def push(self, state, action, reward, next_state, done):
        if not self._initialized:
            self._initialize(state)

        self.states[self.position] = np.array(state)
        self.next_states[self.position] = np.array(next_state)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Direct array indexing - much faster than list comprehension + stack
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer:
    """Optimized PER buffer using pre-allocated numpy arrays with uint8 storage."""

    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_frames=100000, state_shape=None):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-5
        self.max_priority = 1.0

        self.position = 0
        self.size = 0

        # Will be lazily initialized on first push
        self.state_shape = state_shape
        self.states = None
        self.next_states = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self._initialized = False

    def _initialize(self, state):
        """Lazily initialize arrays based on first state."""
        state = np.array(state)
        self.state_shape = state.shape

        # Use uint8 for images (0-255), float32 for low-dim states
        is_image = len(self.state_shape) == 3
        dtype = np.uint8 if is_image else np.float32

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=dtype)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._initialized = True

    def push(self, state, action, reward, next_state, done):
        if not self._initialized:
            self._initialize(state)

        self.states[self.position] = np.array(state)
        self.next_states[self.position] = np.array(next_state)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)
        self.priorities[self.position] = self.max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_probabilities(self):
        scaled_priorities = self.priorities[:self.size] ** self.alpha
        return scaled_priorities / np.sum(scaled_priorities)

    def get_importance_weights(self, probs):
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        weights = (self.size * probs) ** (-beta)
        return torch.FloatTensor(weights / weights.max()).to(self.device)

    def sample(self, batch_size):
        probs = self.get_probabilities()
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        weights = self.get_importance_weights(probs[indices])

        # Direct array indexing - much faster than list comprehension + stack
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).unsqueeze(1).to(self.device)

        self.frame += 1
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = np.minimum(td_errors, 100.0)
        self.max_priority = max(self.max_priority, np.max(td_errors))

    def __len__(self):
        return self.size
