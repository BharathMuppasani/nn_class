import torch
import torch.nn.functional as F
import numpy as np
from dqn import DQN, DuelingDQN, CNNDQN, DuelingCNNDQN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    def __init__(self, 
                 state_dim,
                 action_dim,
                 device,
                 learning_rate=3e-4,
                 gamma=0.99,
                 buffer_size=100000,
                 batch_size=128,
                 target_update=10,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 use_double=False,
                 use_dueling=False,
                 use_priority=False):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.use_priority = use_priority

        # Initialize networks
        is_image = isinstance(state_dim, (tuple, list, torch.Size)) and len(state_dim) == 3

        if is_image:
            network = DuelingCNNDQN if use_dueling else CNNDQN
            self.policy_net = network(state_dim, action_dim).to(device)
            self.target_net = network(state_dim, action_dim).to(device)
        else:
            network = DuelingDQN if use_dueling else DQN
            self.policy_net = network(state_dim, action_dim).to(device)
            self.target_net = network(state_dim, action_dim).to(device)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, eps=1e-5)
        
        # Initialize replay buffer
        if use_priority:
            self.memory = PrioritizedReplayBuffer(buffer_size, device)
        else:
            self.memory = ReplayBuffer(buffer_size, device)
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.training_step = 0
        self.tau = 0.005  # Soft update parameter
    
    def select_action(self, state):
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return np.random.randint(self.action_dim)
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from replay buffer
        if self.use_priority:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None

        weights = weights.unsqueeze(1)  # [B] → [B, 1] to match elementwise_loss shape

        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            if self.use_double:
                # Double Q-learning: action selection from policy net
                next_q_values = self.policy_net(next_states)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                # Value estimation from target net
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Regular Q-learning
                next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute TD errors for priority update
        td_errors = torch.abs(target_q_values - current_q_values)
        
        # Compute loss
        elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values.detach(), reduction='none')
        loss = (weights * elementwise_loss).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities in PER
        if self.use_priority and indices is not None:
            self.memory.update_priorities(indices, td_errors.squeeze(1).detach().cpu().numpy())
        
        # Soft update target network
        if self.training_step % self.target_update == 0:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1 - self.tau) * target_param.data
                )


        
        self.training_step += 1
        
        return loss.item(), target_q_values, current_q_values, next_q_values, dones, td_errors

    def decay_epsilon(self):
        """Decay epsilon once per episode (called from training loop)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'use_double': self.use_double,
            'use_dueling': self.use_dueling,
            'use_priority': self.use_priority
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']