import numpy as np
import torch


class LQRAgent:
    """
        action planning by the LQR method
    """
    def __init__(
        self,
        encoder,
        transition_model,
        cost_function,
        planning_horizon: int,
        sample: bool=True,
    ):
        self.encoder = encoder
        self.transition_model = transition_model
        self.cost_function = cost_function
        self.planning_horizon = planning_horizon
        self.sample = sample

        self.device = next(encoder.parameters()).device
        self.Ks, self.ks = self._compute_policy()
        self.step = 0

    def __call__(self, obs):

        # convert o_t to a torch tensor and add a batch dimension
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.transition_model.eval()
        
            target = self.cost_function.target
            state_dist = self.encoder(obs)
            state = state_dist.sample() if self.sample else state_dist.loc
            action = (state - target) @ self.Ks[self.step].T + self.ks[self.step].T
        
        self.step += 1
        return np.clip(action.cpu().numpy(), min=-1.0, max=1.0)
    
    def _compute_policy(self):
        state_dim, action_dim = self.transition_model.B.shape

        Ks = []
        ks = []

        V = torch.zeros((state_dim, state_dim), device=self.device)
        v = torch.zeros((state_dim, 1), device=self.device)

        C = torch.block_diag(self.cost_function.Q, self.cost_function.R)
        c = torch.zeros((state_dim + action_dim, 1), device=self.device)

        F = torch.concatenate((self.transition_model.A, self.transition_model.B), dim=1)
        f = (
            self.transition_model.o
            + (self.transition_model.A - torch.eye(state_dim, device=self.device))
            @ self.cost_function.target.T
        )

        for _ in range(self.planning_horizon-1, -1, -1):
            Q = C + F.T @ V @ F
            q = c + F.T @ V @ f + F.T @ v
            Qxx = Q[:state_dim, :state_dim]
            Qxu = Q[:state_dim, state_dim:]
            Qux = Q[state_dim:, :state_dim]
            Quu = Q[state_dim:, state_dim:]
            qx = q[:state_dim, :]
            qu = q[state_dim:, :]

            K = - torch.linalg.pinv(Quu) @ Qux
            k = - torch.linalg.pinv(Quu) @ qu
            V = Qxx + Qxu @ K + K.T @ Qux + K.T @ Quu @ K
            v = qx + Qxu @ k + K.T @ qu + K.T @ Quu @ k

            Ks.append(K)
            ks.append(k)
        
        return Ks[::-1], ks[::-1]
    
    def reset(self):
        self.step = 0