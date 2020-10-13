import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[1024, 512, 256, 256],
                bn=True,
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[256, 256, 256, 128],
                bn=True,
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128, 128, 128, 64], 
                bn=True,
                use_xyz=True,
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LeakyReLU(True),
            nn.Linear(32, 12, bias=False),
            nn.LeakyReLU(True),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        res = self.fc_layer(features.squeeze(-1))
        return res

