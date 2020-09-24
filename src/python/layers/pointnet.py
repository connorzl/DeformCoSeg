from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        # Convolution layers to get features B x F x N
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Max pool along points dimension: B x F x 1
        x = torch.max(x, 2, keepdim=True)[0]
        # Basically a squeeze operator
        x = x.view(-1, 1024)

        # Fully connected layers: B x F'
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer: B x 9
        x = self.fc3(x)
        
        # Initialize output matrix as identity: B x 9
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden

        # Output matrix: B x 3 x 3
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

    def forward(self, x):
        n_pts = x.size()[2]
        # B x 3 x 3
        trans = self.stn(x)
        # B x 3 x n_pts
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        # B x 64 x n_pts
        x = F.relu(self.conv1(x))

        # B x 1024
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


if __name__ == '__main__':
    # Batch x Point x Number of Points
    sim_data = Variable(torch.rand(32,3,2500))
    sim_data_64d = Variable(torch.rand(32, 64, 2500))

    """
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))
    """

    pointfeat = PointNetfeat()
    out = pointfeat(sim_data)
    print('global feat', out.size())