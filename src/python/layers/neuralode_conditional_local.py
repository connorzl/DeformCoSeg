import os
import torch
from torch import nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint
import numpy as np

import torchdiffeq
print("torchdiffeq library location:", os.path.dirname(torchdiffeq.__file__))


class ODEFuncPointNet(nn.Module):
    def __init__(self, k=1):
        super(ODEFuncPointNet, self).__init__()
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

    def forward(self, latent_vector, points, t):
        # Concatenate target global features vector to each point.
        t_repeat = torch.reshape(t, (1, 1)).repeat(points.shape[0], 1)
        net_input = torch.cat([points, t_repeat, latent_vector], dim=1)
        velocity_field = self.net(net_input)
        return velocity_field

class NeuralFlowModel(nn.Module):
    def __init__(self, dim=3, latent_size=1, out=3, device=torch.device('cpu')):
        super(NeuralFlowModel, self).__init__()
        self.device = device
        self.flow_net = ODEFuncPointNet(k=dim+latent_size)
        self.flow_net = self.flow_net.to(device)
        self.latent_updated = False
        self.lat_params = None

    def update_latents(self, latent_sequence):
        """
        Args:
            latent_sequence: long or float tensor of shape [batch, nsteps, latent_size].
                             sequence of latents along deformation path.
                             if long, index into self.lat_params to retrieve latents.
        """
        self.latent_sequence = latent_sequence
        self.latent_updated = True

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
        flow = self.flow_net(self.latent_sequence, points, t)
        return flow

class NeuralFlowDeformer(nn.Module):
    def __init__(self, adjoint=False, dim=3, latent_size=1, out=3, method='dopri5', \
            atol=1e-4, rtol=1e-4, device=torch.device('cpu')):
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
        self.timing = torch.from_numpy(np.array([0., 0.2, 0.4, 0.6, 0.8, 1.]).astype('float32'))
        self.timing = self.timing.to(device)
        self.rtol = rtol
        self.atol = atol
        self.device = device
        self.net = NeuralFlowModel(dim=dim, latent_size=latent_size, out=out, device=self.device)
        self.net = self.net.to(device)
    
    def forward(self, points, latent_sequence):
        """Forward transformation (source -> latent_path -> target).

        To perform backward transformation, simply switch the order of the lat codes.

        Args:
          latent_sequence: float tensor of shape [batch, nsteps, latent_size]
          points: [batch, num_points, dim]
        Returns:
          points_transformed: tensor of shape [nsteps, batch, num_points, dim]
        """
        self.net.update_latents(latent_sequence)
        points_transformed = self.odeint(self.net, points, self.timing, method=self.method,
                                         rtol=self.rtol, atol=self.atol)
        return points_transformed[-1]

