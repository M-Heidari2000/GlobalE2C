import torch
import numpy as np
from torch.utils.data import Dataset


class ReplayBuffer:
    """
        Replay buffer stores sample trajectories
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int
    ):
        self.capacity = capacity
        
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_observation = np.zeros((capacity, observation_dim), dtype=np.float32)

        self.index = 0
        self.is_filled = False

    def __len__(self):
        return self.capacity if self.is_filled else self.index
    
    def push(
        self,
        observation,
        action,
        next_observation,
    ):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.next_observation[self.index] = next_observation

        self.index = (self.index + 1) % self.capacity
        self.is_filled = self.is_filled or self.index == 0

    
    def sample(
        self,
        batch_size: int,
    ):
        assert len(self) >= batch_size, "not enough data in the buffer"
        indexes = np.random.permutation(len(self))[:batch_size]

        sampled_observations = self.observations[indexes]
        sampled_actions = self.actions[indexes]
        sampled_next_observations = self.next_observation[indexes]

        return sampled_observations, sampled_actions, sampled_next_observations
    

class StaticDataset(Dataset):
    """
        static dataset to hold triplets
    """
    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int
    ):
        self.capacity = capacity
        
        self.observations = torch.empty((capacity, observation_dim), dtype=torch.float32)
        self.actions = torch.empty((capacity, action_dim), dtype=torch.float32)
        self.next_observation = torch.empty((capacity, observation_dim), dtype=torch.float32)

        self.index = 0
        self.is_filled = False

    def __len__(self):
        return self.capacity if self.is_filled else self.index
    
    def __getitem__(self, index):
        observations = self.observations[index]
        actions = self.actions[index]
        next_observations = self.next_observation[index]

        return observations, actions, next_observations
    
    def push(
        self,
        observation,
        action,
        next_observation,
    ):
        self.observations[self.index] = torch.as_tensor(observation)
        self.actions[self.index] = torch.as_tensor(action)
        self.next_observation[self.index] = torch.as_tensor(next_observation)

        self.index = (self.index + 1) % self.capacity
        self.is_filled = self.is_filled or self.index == 0