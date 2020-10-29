import os
import torch
from torch import nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint
import numpy as np

import torchdiffeq
print("torchdiffeq library location:", os.path.dirname(torchdiffeq.__file__))


class ODEFuncPointNet(nn.Module):
    def __init__(self, dim=1, latent=1, num_parts=1):
        super(ODEFuncPointNet, self).__init__()

    def forward(self, points, flow_mask, transform):
        x = transform[0]
        y = transform[1]
        z = transform[2]
        A = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]]).to(transform.device)
        b = transform[3:6]

        A = A.unsqueeze(0)
        b = b.unsqueeze(0)
        b = b.unsqueeze(2)

        # Compute deformed position.
        flow = torch.matmul(A, points.unsqueeze(2)) + b
        flow = flow.squeeze(2)
       
        flow = flow_mask.unsqueeze(1).float() * flow
        return flow


class NeuralFlowModel(nn.Module):
    def __init__(self, dim=3, latent_size=1, num_parts=3, device=torch.device('cpu')):
        super(NeuralFlowModel, self).__init__()
        self.device = device
        self.flow_net = ODEFuncPointNet(dim, latent_size, num_parts)
        self.flow_net = self.flow_net.to(device)
        self.latent_updated = False
        self.mask_updated = False
        self.latent_sequence = None
        self.flow_mask = None
        self.transform = None

    def update_latents(self, latent_sequence):
        """
        Args:
            latent_sequence: long or float tensor of shape [batch, nsteps, latent_size].
                             sequence of latents along deformation path.
                             if long, index into self.lat_params to retrieve latents.
        """
        self.latent_sequence = latent_sequence
        self.latent_updated = True

    def update_mask(self, flow_mask):
        self.flow_mask = flow_mask
        self.mask_updated = True

    def update_transform(self, transform):
        self.transform = transform

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
        if not self.latent_updated:
            raise RuntimeError('Latent not updated. '
                               'Use .update_latents() to update the source and target latents.')
        if not self.mask_updated:
            raise RuntimeError('Flow not updated. '
                               'Use .update_mask() to update the flow masks.')
        #latent_val = self.latent_at_t(t)
        flow = self.flow_net(points, self.flow_mask, self.transform)
        return flow


class NeuralFlowDeformer(nn.Module):
    def __init__(self, adjoint=False, dim=3, latent_size=1, num_parts=1, method='dopri5', \
            atol=1e-5, rtol=1e-5, device=torch.device('cpu')):
        """Initialize. The parameters are the parameters for the Deformation Flow network.
        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          rtol, atol: float, relative / absolute error tolerence in ode solver.
        """
        super(NeuralFlowDeformer, self).__init__()
        self.method = method
        if adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint
        self.timing = torch.from_numpy(np.array([0., 1.]).astype('float32'))
        self.timing = self.timing.to(device)
        self.timing_inv = torch.from_numpy(np.array([1, 0]).astype('float32'))
        self.timing_inv = self.timing_inv.to(device)
        self.rtol = rtol
        self.atol = atol
        self.device = device
        self.net = NeuralFlowModel(
            dim=dim, latent_size=latent_size, num_parts=num_parts, device=self.device)
        self.net = self.net.to(device)
    
    def forward(self, points, latent_sequence, flow_mask, transform):
        """Forward transformation (source -> latent_path -> target).

        To perform backward transformation, simply switch the order of the lat codes.

        Args:
          points: [batch, num_points, 3]
          latent_sequence: float tensor of shape [batch, nsteps, latent_size]
          flow_mask: [batch, num_points, num_parts]
        Returns:
          points_transformed: tensor of shape [nsteps, batch, num_points, dim]
        """
        self.net.update_latents(latent_sequence)
        self.net.update_mask(flow_mask)
        self.net.update_transform(transform)
        points_transformed = self.odeint(self.net, points, self.timing, method=self.method,
                                         rtol=self.rtol, atol=self.atol)
        return points_transformed[-1]
