import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import MultivariateNormal


class Encoder(nn.Module):
    """
        p(z|x)
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        hidden_dim: Optional[int]=None,
        min_var: Optional[float]=1e-3,
        dropout_p: Optional[float]=0.4,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*observation_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

        self._min_var = min_var

    def forward(self, observation):
        hidden = self.mlp_layers(observation)
        mean = self.mean_head(hidden)
        var = self.var_head(hidden) + self._min_var

        return MultivariateNormal(mean, torch.diag_embed(var))
    

class Decoder(nn.Module):
    """
        p(x|z)
    """

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        hidden_dim: Optional[int]=None,
        dropout_p: Optional[float]=1e-3,
    ):
        
        super().__init__()
        
        hidden_dim = hidden_dim if hidden_dim is not None else 2*state_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, observation_dim),
        )

    
    def forward(self, state):
        return self.mlp_layers(state)
    

class TransitionModel(nn.Module):

    """
        Estimates the globally linear dynamics matrices
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        min_var: Optional[float]=1e-3,
    ):
        
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.A = nn.Parameter(torch.randn(self.state_dim, self.state_dim))
        self.B = nn.Parameter(torch.randn(self.state_dim, self.action_dim))
        self.o = nn.Parameter(torch.randn(1, self.state_dim))
        self.w = nn.Parameter(torch.randn(self.state_dim))

        self._min_var = min_var

    def forward(
        self,
        state_dist,
        action,
        ):

        w = nn.functional.softplus(self.w) + self._min_var

        # next state mean computation
        mu = state_dist.loc
        next_state_mean = mu @ self.A.T + action @ self.B.T + self.o

        # next state covariance computation
        H = torch.diag(w)    # s * s
        sigma = state_dist.covariance_matrix    # b * s * s
        C = H + self.A @ sigma @ self.A.T
        
        next_state_dist = MultivariateNormal(next_state_mean, C)

        return next_state_dist