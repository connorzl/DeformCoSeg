import torch
from torch import nn
from torchdiffeq import odeint

import numpy as np

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        m = 50
        nlin = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(4, m),
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

    def forward(self, t, y):
        new_t = t.repeat(y.shape[0],1)
        yt = torch.cat((y,new_t), 1)
        res = self.net(yt - 0.5)
        return res
        #return self.net(yt-0.5)

class ODEFuncPointNet(nn.Module):
    def __init__(self, k=1027):
        super(ODEFuncPointNet, self).__init__()
        nlin = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(k, 1024),
            nlin,
            nn.Linear(1024, 512),
            nlin,
            nn.Linear(512, 256),
            nlin,
            nn.Linear(256, 128),
            nlin,
            nn.Linear(128, 64),
            nlin,
            nn.Linear(64, 3)
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
    def __init__(self, dim=3, latent_size=1024, device=torch.device('cpu')):
        super(NeuralFlowModel, self).__init__()
      
        self.flow_net = ODEFuncPointNet()
        self.latent_updated = False
        self.device = device


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
        t = t.to(self.device)

        # find the interpolation coefficient between the latents at the two ends of the bin
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
        return flow


class NeuralODE():
    def __init__(self, device=torch.device('cpu')):
        super(NeuralODE, self).__init__()
        self.timing = torch.from_numpy(np.array([0, 1]).astype('float32'))
        self.timing_inv = torch.from_numpy(np.array([1, 0]).astype('float32'))
        self.timing = self.timing.to(device)
        self.timing_inv = self.timing_inv.to(device)
        self.func = ODEFunc()
        self.func = self.func.to(device)
        self.device = device

    def to_device(self, device):
        self.func = self.func.to(device)
        self.timing = self.timing.to(device)
        self.timing_inv = self.timing_inv.to(device)
        self.device = device
        
    def parameters(self):
        return self.func.parameters()

    def forward(self, u):
        return odeint(self.func, u, self.timing, method="rk4", rtol=1e-4, atol=1e-4)[1]

    def inverse(self, u):
        return odeint(self.func, u, self.timing_inv, method="rk4", rtol=1e-4, atol=1e-4)[1]

    def integrate(self, u, t1, t2, device):
        new_time = torch.from_numpy(np.array([t1,t2]).astype('float32')).to(device)
        return odeint(self.func, u, new_time, method="rk4", rtol=1e-4, atol=1e-4)[1]


class NeuralFlowDeformer(nn.Module):
    def __init__(self, dim=3, latent_size=1024, method='dopri5', atol=1e-5, rtol=1e-5, device=torch.device('cpu')):
        """Initialize. The parameters are the parameters for the Deformation Flow network.
        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          rtol, atol: float, relative / absolute error tolerence in ode solver.
        """
        super(NeuralFlowDeformer, self).__init__()
        self.method = method
        self.odeint = odeint
        self.__timing = torch.from_numpy(np.array([0., 1.]).astype('float32'))
        self.__timing_inv = torch.from_numpy(np.array([1, 0]).astype('float32'))
        self.rtol = rtol
        self.atol = atol
        self.device = device
        self.net = NeuralFlowModel(dim=dim, latent_size=latent_size, device=device)
        self.net = self.net.to(device)

    @property
    def timing(self):
        return self.__timing

    @timing.setter
    def timing(self, timing):
        assert(isinstance(timing, torch.Tensor))
        assert(timing.ndim == 1)
        self.__timing = timing

    @property
    def timing_inv(self):
        return self.__timing_inv

    @timing_inv.setter
    def timing_inv(self, timing_inv):
        assert(isinstance(timing_inv, torch.Tensor))
        assert(timing_inv.ndim == 1)
        self.__timing_inv = timing_inv

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
        points_transformed = odeint(
            self.net, points, self.timing_inv, method=self.method, rtol=self.rtol, atol=self.atol)
        return points_transformed[-1]
