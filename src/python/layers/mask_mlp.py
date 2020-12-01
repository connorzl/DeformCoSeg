import torch
import torch.nn.functional as F
from torch import nn

class MaskNet(nn.Module):
    def __init__(self, latent_size, num_parts):
        super(MaskNet, self).__init__()
        self.conv1 = nn.Conv1d(latent_size, 16, 1)
        self.conv2 = nn.Conv1d(16, 8, 1)
        self.conv3 = nn.Conv1d(8, num_parts, 1)

    """
        Input: B x N x 3
        Output: B x N x K (logits)
    """
    def forward(self, x):
        num_point = x.shape[1]
        net = x.permute(0, 2, 1)
        net = F.leaky_relu(self.conv1(net))
        net = F.leaky_relu(self.conv2(net))
        net = F.leaky_relu(self.conv3(net))
        net = net.permute(0, 2, 1)
        net = F.softmax(net, dim=2)
        return net


