import torch
import torch.nn as nn
from typing import Optional
import monotonicnetworks as lmn
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


class CostModel(nn.Module):
    """
        Learnable quadratic cost function in the latent space
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        hidden_dim: Optional[int]=16,
    ):
        
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.device = device
        self.A = nn.Parameter(
            torch.eye(state_dim, device=self.device, dtype=torch.float32),
        )
        self.B = nn.Parameter(
            torch.eye(action_dim, device=self.device, dtype=torch.float32)
        )
        self.q = nn.Parameter(
            torch.randn((state_dim, 1), device=self.device, dtype=torch.float32)
        )

        # monotonic increasing function
        self.F = lmn.MonotonicWrapper(
            nn.Sequential(
                lmn.LipschitzLinear(1, hidden_dim, kind="one-inf"),
                lmn.GroupSort(2),
                lmn.LipschitzLinear(hidden_dim, hidden_dim, kind="inf"),
                lmn.GroupSort(2),
                lmn.LipschitzLinear(hidden_dim, 1, kind="inf")
            ),
            monotonic_constraints=[1],
        ).to(device=self.device)

    @property
    def Q(self):
        return self.A @ self.A.T
    
    @property
    def R(self):
        L = torch.tril(self.B)
        diagonals = nn.functional.softplus(L.diagonal()) + 1e-4
        X = 1 - torch.eye(self.action_dim, device=self.device, dtype=torch.float32)
        L = L * X + diagonals.diag()
        return L @ L.T
    
    def forward(self, state, action):
        # x: b x
        # u: b u
        # TODO: use torch.einsum for efficieny
        cost = 0.5 * state @ self.Q @ state.T + 0.5 * action @ self.R @ action.T
        cost = cost.diagonal().unsqueeze(1) + state @ self.q
        return self.F(cost)
                

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