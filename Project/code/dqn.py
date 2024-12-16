import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Value stream
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        # Advantage stream
        self.fc_adv = nn.Linear(hidden_dim, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, output_dim)

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = F.relu(self.fc_value(x))
        value = self.value(value)

        adv = F.relu(self.fc_adv(x))
        adv = self.advantage(adv)

        # Advantage mean subtraction for stability
        qvals = value + adv - adv.mean(dim=1, keepdim=True)
        return qvals