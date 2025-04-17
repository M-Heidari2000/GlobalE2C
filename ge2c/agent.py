import torch
import numpy as np
from mpc import mpc
from mpc.mpc import QuadCost, LinDx


class LQRAgent:
    """
        action planning by the LQR method
    """
    def __init__(
        self,
        encoder,
        transition_model,
        cost_model,
        planning_horizon: int,
        sample: bool=True,
    ):
        self.encoder = encoder
        self.transition_model = transition_model
        self.cost_model = cost_model
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
        
            state_dist = self.encoder(obs)
            state = state_dist.sample() if self.sample else state_dist.loc
            action = state @ self.Ks[self.step].T + self.ks[self.step].T
        
        self.step += 1
        return np.clip(action.cpu().numpy(), min=-1.0, max=1.0)
    
    def _compute_policy(self):
        state_dim, action_dim = self.transition_model.B.shape

        Ks = []
        ks = []

        V = torch.zeros((state_dim, state_dim), device=self.device)
        v = torch.zeros((state_dim, 1), device=self.device)

        C = torch.block_diag(self.cost_model.Q, self.cost_model.R)
        c = torch.cat([
            self.cost_model.q,
            torch.zeros((action_dim, 1), device=self.device)
        ])

        F = torch.cat((self.transition_model.A, self.transition_model.B), dim=1)
        f = self.transition_model.o

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


class MPCAgent:
    """
        action planning by the LQR method
    """
    def __init__(
        self,
        encoder,
        transition_model,
        cost_model,
        planning_horizon: int,
        sample: bool=False,
    ):
        self.encoder = encoder
        self.transition_model = transition_model
        self.cost_model = cost_model
        self.planning_horizon = planning_horizon

        self.sample = sample

        self.device = next(encoder.parameters()).device

        state_dim, action_dim = self.transition_model.B.shape

        C = torch.block_diag(self.cost_model.Q, self.cost_model.R).repeat(
            self.planning_horizon, 1, 1, 1,
        )

        c = torch.cat([
            self.cost_model.q.reshape(1, -1),
            torch.zeros((1, action_dim), device=self.device)
        ], dim=1).repeat(self.planning_horizon, 1, 1)

        F = torch.cat((self.transition_model.A, self.transition_model.B), dim=1).repeat(
            self.planning_horizon, 1, 1, 1
        )
        f = self.transition_model.o.repeat(self.planning_horizon, 1, 1)

        self.quadcost = QuadCost(C, c)
        self.lindx = LinDx(F, f)

        self.planner = mpc.MPC(
            n_batch=1,
            n_state=state_dim,
            n_ctrl=action_dim,
            T=self.planning_horizon,
            u_lower=-1.0,
            u_upper=1.0,
            lqr_iter=50,
            backprop=False,
            exit_unconverged=False,
        )

    def __call__(self, obs):

        # convert o_t to a torch tensor and add a batch dimension
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.transition_model.eval()
        
            state_dist = self.encoder(obs)
            state = state_dist.sample() if self.sample else state_dist.loc

            planned_x, planned_u, _ = self.planner(
                state,
                self.quadcost,
                self.lindx
            )

        return np.clip(planned_u.squeeze(1).cpu().numpy(), a_min=-1.0, a_max=1.0)