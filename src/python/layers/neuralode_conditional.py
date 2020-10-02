import os
import torch
from torch import nn
from torchdiffeq import odeint
import numpy as np

import torchdiffeq
print("torchdiffeq library location:", os.path.dirname(torchdiffeq.__file__))


class ODEFuncPointNet(nn.Module):
    def __init__(self, k=1):
        super(ODEFuncPointNet, self).__init__()
        nlin = nn.LeakyReLU()
        m = 50
        nlin = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(k, m),
            nlin,
            nn.Linear(m, m),
            nlin,
            nn.Linear(m, m),
            nlin,
            nn.Linear(m, 3),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, latent_vector, points):
        # Concatenate target global features vector to each point.
        latent_vector = torch.unsqueeze(latent_vector, dim=0).repeat((points.shape[0], 1))
        net_input = torch.cat((points, latent_vector), dim=1)
        velocity_field = self.net(net_input)
        return velocity_field


class NeuralFlowModel(nn.Module):
    def __init__(self, dim=3, latent_size=1, device=torch.device('cpu')):
        super(NeuralFlowModel, self).__init__()
        self.device = device
        self.flow_net = ODEFuncPointNet(k=dim+latent_size)
        self.flow_net = self.flow_net.to(device)
        self.latent_updated = False

    def update_latents(self, latent_sequence):
        """
        Args:
            latent_sequence: long or float tensor of shape [batch, nsteps, latent_size].
                             sequence of latents along deformation path.
                             if long, index into self.lat_params to retrieve latents.
        """
        self.latent_sequence = latent_sequence
        self.latent_updated = True

    def latent_at_t(self, t):
        """Helper fn to compute latent at t."""
        t0 = 0
        t1 = 1
        alpha = (t - t0) / (t1 - t0)  # [batch]
        latent_t0 = self.latent_sequence[0, :]  
        latent_t1 = self.latent_sequence[1, :]
        latent_val = latent_t0 + alpha * (latent_t1 - latent_t0)
        return latent_val

    def forward(self, t, points):
        """
        Args:
          t: float, deformation parameter between 0 and 1.
          points: [batch, num_points, dim]
        Returns:
          vel: [batch, num_points, dim]
        """
        # reparametrize eval along latent path as a function of a single scalar t
        if not self.latent_updated:
            raise RuntimeError('Latent not updated. '
                               'Use .update_latents() to update the source and target latents.')
        latent_val = self.latent_at_t(t)
        flow = self.flow_net(latent_val, points)  # [batch, num_pints, dim]
        flow *= torch.norm(self.latent_sequence[1, :] - self.latent_sequence[0, :])
        return flow


class NeuralFlowDeformer(nn.Module):
    def __init__(self, dim=3, latent_size=1, method='dopri5', atol=1e-5, rtol=1e-5, device=torch.device('cpu')):
        """Initialize. The parameters are the parameters for the Deformation Flow network.
        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          rtol, atol: float, relative / absolute error tolerence in ode solver.
        """
        super(NeuralFlowDeformer, self).__init__()
        self.method = method
        self.odeint = odeint
        self.timing = torch.from_numpy(np.array([0., 1.]).astype('float32'))
        self.timing = self.timing.to(device)
        self.timing_inv = torch.from_numpy(np.array([1, 0]).astype('float32'))
        self.timing_inv = self.timing_inv.to(device)
        self.rtol = rtol
        self.atol = atol
        self.device = device
        self.net = NeuralFlowModel(dim=dim, latent_size=latent_size, device=self.device)
        self.net = self.net.to(device)

    def forward(self, points, latent_sequence):
        """Forward transformation (source -> latent_path -> target).

        To perform backward transformation, simply switch the order of the lat codes.

        Args:
          points: [batch, num_points, dim]
          latent_sequence: float tensor of shape [batch, nsteps, latent_size]
        Returns:
          points_transformed: tensor of shape [nsteps, batch, num_points, dim]
        """
        self.net.update_latents(latent_sequence)
        points_transformed = self.odeint(self.net, points, self.timing, method=self.method,
                                         rtol=self.rtol, atol=self.atol)
        return points_transformed[-1]

    def inverse(self, points, latent_sequence):
        self.net.update_latents(latent_sequence)
        points_transformed = self.odeint(
            self.net, points, self.timing_inv, method=self.method, rtol=self.rtol, atol=self.atol)
        return points_transformed[-1]
