from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

    """
        Input: B x 3 x N
    """
    def forward(self, x):
        n_pts = x.size()[2]
        
        # B x 64 x N
        x = F.relu(self.conv1(x))
        # B x 128 x N 
        x = F.relu(self.conv2(x))
        # B x 1024 x N
        x = self.conv3(x)

        # B x 1024
        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)
        return x


if __name__ == '__main__':
    # Batch x Point x Number of Points
    sim_data = Variable(torch.rand(32,3,2500))
    sim_data_64d = Variable(torch.rand(32, 64, 2500))

    pointfeat = PointNetfeat()
    out = pointfeat(sim_data)
    print('global feat', out.size())
