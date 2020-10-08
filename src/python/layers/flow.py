import torch
from torch import nn
import torch.nn.functional as F
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pointnet_plus

class FlowMLP(nn.Module):

    def __init__(self):
        super(FlowMLP, self).__init__()
        self.conv1 = nn.Conv1d(2051, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, 64, 1)
        self.fc1 = nn.Linear(64, 3)

    """
        Input: B x N x (3 + 1024 + 1024)
        Output: B x N x 3
    """
    def forward(self, x):
        net = x.permute(0, 2, 1)
        nlin = nn.LeakyReLU()
        
        # Result: B x 1024 x N
        net = nlin(self.conv1(net))
        net = nlin(self.conv2(net))
        net = nlin(self.conv3(net))
        net = nlin(self.conv4(net))
        net = net.permute(0, 2, 1)

        # Result: B x 1024 x 3
        net = nlin(self.fc1(net))
        return net
    

class FlowNetwork(nn.Module):

    def __init__(self, network_type='mlp'):
        super(FlowNetwork, self).__init__()

        if network_type == 'mlp':
            self.net = FlowMLP()
        elif network_type == 'pointnet_plus':
            self.net = pointnet_plus.PointNet2()
        else:
            raise ValueError('ERROR: unknown network_type: %s!' % network_type)

    """
        Input: B x N x 3, 1024, 1024
        Output: B x N x 3
    """
    def forward(self, V_input, z_src, z_targ):
        batch = V_input.shape[0]
        n_pts = V_input.shape[1]
        z_src = z_src.view(1, 1, -1)
        z_src = z_src.repeat(batch, n_pts, 1)
        z_targ = z_targ.view(1, 1, -1)
        z_targ = z_targ.repeat(batch, n_pts, 1)
            
        deform_input = torch.cat([V_input, z_src, z_targ], dim=2)
        deformed = self.net(deform_input)
        return deformed

