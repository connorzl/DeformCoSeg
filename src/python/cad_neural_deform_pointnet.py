import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.graph_loss2_layer import GraphLoss2Layer, Finalize
from layers.reverse_loss_layer import ReverseLossLayer
from layers.maf import MAF
from layers.neuralode_fast import NeuralODE
from layers.pointnet import PointNetfeat, feature_transform_regularizer
import pyDeform

import torch
import numpy as np
from time import time

import argparse

parser = argparse.ArgumentParser(description='Rigid Deformation.')
parser.add_argument('--source', default='../data/cad-source.obj')
parser.add_argument('--target', default='../data/cad-target.obj')
parser.add_argument('--output', default='./cad-output.obj')
parser.add_argument('--rigidity', default='0.1')
parser.add_argument('--device', default='cuda')
parser.add_argument('--save_path', default='./cad-output.ckpt')

args = parser.parse_args()

source_path = args.source
reference_path = args.target
output_path = args.output
rigidity = float(args.rigidity)
save_path = args.save_path
device = torch.device(args.device)

FEATURES_REG_LOSS_WEIGHT = 0.001

# Vertices has shape [V, 3], each element is (v_x, v_y, v_z)
# Faces has shape [F, 3], each element is (i, j, k)
# Edges has shape [E, 2], each element is (i, j)
# V2G1 maps each vertex to the voxel number that it belongs to, has shape [F, 1]
# Skeleton mesh vertices has shape [GV, 3], each skeleton vertex is the average of vertices in its voxel.
# Skeleton mesh edges has shape [GE, 2]
V1, F1, E1, V2G1, GV1, GE1 = pyDeform.LoadCadMesh(source_path)
V2, F2, E2, V2G2, GV2, GE2 = pyDeform.LoadCadMesh(reference_path)

# PointNet layer.
pointnet = PointNetfeat(global_feat=True, feature_transform=True)
pointnet.eval()

# Deformation losses layer.
graph_loss = GraphLoss2Layer(V1,F1,GV1,GE1,V2,F2,GV2,GE2,rigidity,device)
param_id1 = graph_loss.param_id1
param_id2 = graph_loss.param_id2

reverse_loss = ReverseLossLayer()

# Flow layer.
func = NeuralODE(device, use_pointnet=True)

optimizer = optim.Adam(func.parameters(), lr=1e-3)

# Prepare input for encoding the skeleton meshes, shape = [1, 3, n_pts].
GV1_pointnet_input = GV1.unsqueeze(0)
GV1_pointnet_input = GV1_pointnet_input.transpose(2, 1)
GV2_pointnet_input = GV2.unsqueeze(0)
GV2_pointnet_input = GV2_pointnet_input.transpose(2, 1)

GV1_origin = GV1.clone()
GV2_origin = GV2.clone()

niter = 1000

GV1_device = GV1.to(device)
GV2_device = GV2.to(device)
loss_min = 1e30
eps = 1e-6
for it in range(0, niter):
	optimizer.zero_grad()

	# Encode both skeleton meshes using PointNet.
	GV1_features, _, GV1_trans_feat = pointnet(GV1_pointnet_input)
	GV2_features, _, GV2_trans_feat = pointnet(GV2_pointnet_input)
	GV1_features = torch.squeeze(GV1_features)
	GV2_features = torch.squeeze(GV2_features)

	GV1_features_device = GV1_features.to(device)
	GV2_features_device = GV2_features.to(device)

	# Compute and integrate velocity field for deformation.
	GV1_deformed, GV2_features_deformed = func.forward((GV1, GV2_features_device))
	# The target features should stay the same, since they are only used to integrate GV1.
	if torch.norm(GV2_features_device - GV2_features_deformed) > eps:
		raise ValueError("Non-identity target features deformation.")

	# Same as above, but in opposite direction.
	GV2_deformed, GV1_features_deformed  = func.inverse((GV2, GV1_features_device))
	if torch.norm(GV1_features_device - GV1_features_deformed) > eps:
		raise ValueError("Non-identity target features deformation.")

	loss1_forward = graph_loss(GV1_deformed, GE1, GV2, GE2, 0)
	loss1_backward = reverse_loss(GV1_deformed, GV2_origin, device)
	loss1_features_reg = FEATURES_REG_LOSS_WEIGHT * feature_transform_regularizer(GV1_trans_feat)

	loss2_forward = graph_loss(GV1, GE1, GV2_deformed, GE2, 1)
	loss2_backward = reverse_loss(GV2_deformed, GV1_origin, device)
	loss2_features_reg = FEATURES_REG_LOSS_WEIGHT * feature_transform_regularizer(GV2_trans_feat)

	loss = loss1_forward + loss1_backward + loss2_forward + loss1_features_reg + loss2_backward + loss2_features_reg

	loss.backward()
	optimizer.step()

	if it % 100 == 0 or True:
		print('iter=%d, loss1_forward=%.6f loss1_backward=%.6f loss2_forward=%.6f loss2_backward=%.6f'
			%(it, np.sqrt(loss1_forward.item() / GV1.shape[0]),
				np.sqrt(loss1_backward.item() / GV2.shape[0]),
				np.sqrt(loss2_forward.item() / GV2.shape[0]),
				np.sqrt(loss2_backward.item() / GV1.shape[0])))

		current_loss = loss.item()

if save_path != '':
	torch.save({'func':func, 'optim':optimizer}, save_path)

GV1_deformed = func.forward(GV1_device)
GV1_deformed = torch.from_numpy(GV1_deformed.data.cpu().numpy())
V1_copy = V1.clone()
#Finalize(V1_copy, F1, E1, V2G1, GV1_deformed, 1.0, param_id2)

pyDeform.NormalizeByTemplate(V1_copy, param_id1.tolist())
V1_origin = V1_copy.clone()

#V1_copy = V1_copy.to(device)
func.func = func.func.cpu()
V1_copy = func.forward(V1_copy)
V1_copy = torch.from_numpy(V1_copy.data.cpu().numpy())

src_to_src = torch.from_numpy(np.array([i for i in range(V1_origin.shape[0])]).astype('int32'))

pyDeform.SolveLinear(V1_origin, F1, E1, src_to_src, V1_copy, 1, 1)
pyDeform.DenormalizeByTemplate(V1_origin, param_id2.tolist())
pyDeform.SaveMesh(output_path, V1_origin, F1)