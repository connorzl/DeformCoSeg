import torch
import torch.nn.functional as F
from torch import nn

class PointNetSeg(nn.Module):
    def __init__(self, num_features):
        super(PointNetSeg, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        self.conv5 = nn.Conv1d(3+64+128+256+1024+1024, 512, 1)
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.conv7 = nn.Conv1d(256, num_features, 1)
    """
        Input: B x N x 3
        Output: B x N x K (logits)
    """
    def forward(self, x):
        num_point = x.shape[1]
        x1 = x.permute(0, 2, 1)
        x2 = F.leaky_relu(self.conv1(x1))
        x3 = F.leaky_relu(self.conv2(x2))
        x4 = F.leaky_relu(self.conv3(x3))
        x5 = F.leaky_relu(self.conv4(x4))
        global_feat = x5.max(dim=-1)[0]
        repeat_global_feat = global_feat.unsqueeze(dim=-1).repeat(1, 1, num_point)
        all_feats = torch.cat([x1, x2, x3, x4, x5, repeat_global_feat], dim=1)
        net = F.leaky_relu(self.conv5(all_feats))
        net = F.leaky_relu(self.conv6(net))
        net = self.conv7(net).permute(0, 2, 1)
        return net


class PointNetCorrelate(nn.Module):
    def __init__(self, input_features, output_features):
        super(PointNetCorrelate, self).__init__()
        self.conv1 = nn.Conv1d(input_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        self.conv5 = nn.Conv1d(input_features+64+128+256+1024+1024, 512, 1)
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.conv7 = nn.Conv1d(256, output_features, 1)
    """
        Input: B x N x 3 + input_features + 1
        Output: B x N x K (logits)
    """
    def forward(self, x):
        num_point = x.shape[1]
        x1 = x.permute(0, 2, 1)
        x2 = F.leaky_relu(self.conv1(x1))
        x3 = F.leaky_relu(self.conv2(x2))
        x4 = F.leaky_relu(self.conv3(x3))
        x5 = F.leaky_relu(self.conv4(x4))
        global_feat = x5.max(dim=-1)[0]
        repeat_global_feat = global_feat.unsqueeze(dim=-1).repeat(1, 1, num_point)
        all_feats = torch.cat([x1, x2, x3, x4, x5, repeat_global_feat], dim=1)
        net = F.leaky_relu(self.conv5(all_feats))
        net = F.leaky_relu(self.conv6(net))
        net = self.conv7(net).permute(0, 2, 1)
        return net


class PointNetMask(nn.Module):
    def __init__(self, input_features, output_parts):
        super(PointNetMask, self).__init__()
        self.conv1 = nn.Conv1d(input_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        self.conv5 = nn.Conv1d(input_features+64+128+256+1024+1024, 512, 1)
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.conv7 = nn.Conv1d(256, output_parts, 1)
    """
        Input: B x N x 3
        Output: B x N x K (logits)
    """
    def forward(self, x):
        num_point = x.shape[1]
        x1 = x.permute(0, 2, 1)
        x2 = F.leaky_relu(self.conv1(x1))
        x3 = F.leaky_relu(self.conv2(x2))
        x4 = F.leaky_relu(self.conv3(x3))
        x5 = F.leaky_relu(self.conv4(x4))
        global_feat = x5.max(dim=-1)[0]
        repeat_global_feat = global_feat.unsqueeze(dim=-1).repeat(1, 1, num_point)
        all_feats = torch.cat([x1, x2, x3, x4, x5, repeat_global_feat], dim=1)
        net = F.leaky_relu(self.conv5(all_feats))
        net = F.leaky_relu(self.conv6(net))
        net = self.conv7(net).permute(0, 2, 1)
        net = net.softmax(dim=2)
        return net

