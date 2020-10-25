import os
import torch
from torch import nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint
import numpy as np

from layers.pointnet_ae import PointNetNoBatchNorm

import torchdiffeq
print("torchdiffeq library location:", os.path.dirname(torchdiffeq.__file__))



class ODEFuncPointNet(nn.Module):
    def __init__(self, dim=1, latent=1, num_parts=1):
        super(ODEFuncPointNet, self).__init__()
        self.net = PointNetNoBatchNorm(num_parts * 12)

    def forward(self, latent_vector, points, flow_mask):
        """
        Input:
          latent_vector: 1024 shape latent code at the current timestep.
          points: V x 3 vertices at the current timestep.
          flow_mask: V mask for the laptop screen
        Output:
          flow: V x 3 velocity for the current timestep.
        """
        # V x 1024
        latent_vector = torch.unsqueeze(latent_vector, dim=0).repeat((points.shape[0], 1))
        net_input = torch.cat((points, latent_vector), dim=1)
        transform = self.net(net_input.unsqueeze(0)).squeeze(0)
        
        # Compute 3x3 rotation matrix.
        rot = transform[0:9]
        rot = torch.reshape(rot, (1, 3, 3))
        (U, S, V) = torch.svd(rot)
        VT = torch.transpose(V, 1, 2)
        sign = torch.sign(torch.det(torch.matmul(U, VT)))
        S_rot = S.clone()
        S_rot[:, 0] = 1.0
        S_rot[:, 1] = 1.0
        S_rot[:, 2] = sign
        S_rot = torch.diag_embed(S_rot) 
        rot = torch.matmul(torch.matmul(U, S_rot), VT)

        # Compute translation.
        trans = transform[9:12]
        trans = torch.reshape(trans, (1, 3, 1))

        # Compute deformed position.
        transformed_verts = torch.matmul(rot, points.unsqueeze(2)) + trans
        transformed_verts = transformed_verts.squeeze(2)
        
        flow = flow_mask.unsqueeze(1) * (transformed_verts - points)
        return flow


class NeuralFlowModel(nn.Module):
    def __init__(self, dim=3, latent_size=1, num_parts=3, device=torch.device('cpu')):
        super(NeuralFlowModel, self).__init__()
        self.device = device
        #self.flow_net = ImNet(dim=dim, in_features=latent_size, out_features=out)
        self.flow_net = ODEFuncPointNet(dim, latent_size, num_parts)
        self.flow_net = self.flow_net.to(device)
        self.latent_updated = False
        self.mask_updated = False
        self.latent_sequence = None
        self.flow_mask = None

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
        latent_val = self.latent_at_t(t)
        flow = self.flow_net(latent_val, points, self.flow_mask)
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
    
    def forward(self, points, latent_sequence, flow_mask):
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
        points_transformed = self.odeint(self.net, points, self.timing, method=self.method,
                                         rtol=self.rtol, atol=self.atol)
        return points_transformed[-1]
