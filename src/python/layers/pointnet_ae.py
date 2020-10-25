import torch
from torch import nn
import torch.nn.functional as F
import sys, os


class PointNetNoBatchNorm(nn.Module):
    def __init__(self, latent_size=12):
        super(PointNetNoBatchNorm, self).__init__()
        
        self.conv1 = nn.Conv1d(1027, 256, 1)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.conv4 = nn.Conv1d(64, 32, 1)

        self.fc1 = nn.Linear(32, latent_size)

    """
        Input: B x N x 1027
        Output: B x F
    """
    def forward(self, pcs):
        net = pcs.permute(0, 2, 1)

        net = torch.relu(self.conv1(net))
        net = torch.relu(self.conv2(net))
        net = torch.relu(self.conv3(net))
        net = torch.relu(self.conv4(net))

        net = net.max(dim=-1)[0]

        net = torch.relu(self.fc1(net))
        return net


class PointNet(nn.Module):
    def __init__(self, latent_size=1024):
        super(PointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, latent_size)
        self.bn5 = nn.BatchNorm1d(latent_size)

    """
        Input: B x N x 3
        Output: B x F
    """
    def forward(self, pcs):
        net = pcs.permute(0, 2, 1)

        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=-1)[0]

        net = torch.relu(self.bn5(self.fc1(net)))
        
        return net


class FCDecoder(nn.Module):

    def __init__(self, num_point=2048, latent_size=1024):
        super(FCDecoder, self).__init__()
        print('Using FCDecoder-NoBN!')

        self.mlp1 = nn.Linear(latent_size, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, num_point*3)

    def forward(self, feat):
        batch_size = feat.shape[0]

        net = feat
        net = torch.relu(self.mlp1(net))
        net = torch.relu(self.mlp2(net))
        net = self.mlp3(net).view(batch_size, -1, 3)

        return net


class Network(nn.Module):

    def __init__(self, conf, latent_size):
        super(Network, self).__init__()
        self.conf = conf

        self.encoder = PointNet(latent_size=latent_size)

        if conf.decoder_type == 'fc':
            self.decoder = FCDecoder(num_point=conf.num_point, latent_size=latent_size)
        else:
            raise ValueError('ERROR: unknown decoder_type %s!' % decoder_type)

    """
        Input: B x N x 3
        Output: B x N x 3, B x F
    """
    def forward(self, input_pcs):
        feats = self.encoder(input_pcs)
        output_pcs = self.decoder(feats)
        return output_pcs, feats  
