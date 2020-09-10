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
    def __init__(self, k=1028):
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

    def forward(self, t, y):
        vertices = y[0]
        features = y[1]

        # Target global features vector should have no change.
        delta_features = torch.zeros_like(features)

        # Concatenate target global features vector to each point.
        features = torch.unsqueeze(features, 0)
        features = features.repeat((vertices.shape[0], 1))
        net_input = torch.cat((vertices, features), dim=1)

        new_t = t.repeat(net_input.shape[0],1)
        yt = torch.cat((net_input,new_t), 1)
        velocity_field = self.net(yt - 0.5)

        return (velocity_field, delta_features)


class NeuralODE():
    def __init__(self, device=torch.device('cpu'), use_pointnet=False):
        super(NeuralODE, self).__init__()
        self.timing = torch.from_numpy(np.array([0, 1]).astype('float32'))
        self.timing_inv = torch.from_numpy(np.array([1, 0]).astype('float32'))
        self.timing = self.timing.to(device)
        self.timing_inv = self.timing_inv.to(device)

        if use_pointnet:
            self.func = ODEFuncPointNet()
        else: 
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
        if isinstance(u, tuple):
            v, f = odeint(self.func, u, self.timing, method="rk4", rtol=1e-4, atol=1e-4)
            return v[1], f[1]
        else:
            return odeint(self.func, u, self.timing, method="rk4", rtol=1e-4, atol=1e-4)[1]

    def inverse(self, u):
        if isinstance(u, tuple):
            v, f = odeint(self.func, u, self.timing_inv, method="rk4", rtol=1e-4, atol=1e-4)
            return v[1], f[1]
        else:
            return odeint(self.func, u, self.timing_inv, method="rk4", rtol=1e-4, atol=1e-4)[1]

    def integrate(self, u, t1, t2, device):
        new_time = torch.from_numpy(np.array([t1,t2]).astype('float32')).to(device)
        return odeint(self.func, u, new_time, method="rk4", rtol=1e-4, atol=1e-4)[1]