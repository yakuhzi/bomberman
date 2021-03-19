import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import random
import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # what is leaky relu?
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class Memory:
    def __init__(self, action_dim, buffer_dim, batch_size):
        self.action_dim = action_dim
        self.memory = collections.deque(maxlen=buffer_dim)
        self.batch_size = batch_size
        self.experiences = collections.namedtuple("Experience", field_names=["state",
                                                                "action",
                                                                "reward",
                                                                "next_state"])


    def add(self, state, action, reward, next_state):
        e = self.experiences(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)

        return states, actions, rewards, next_states

    def sample_experience(self):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)

        return experiences

    def __len__(self):
        return len(self.memory)

