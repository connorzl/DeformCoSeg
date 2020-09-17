import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))

from torch import nn
import torch.optim as optim
from torch.autograd import Function

import torch
from layers.graph_loss2_layer import GraphLoss2Layer, Finalize
from layers.reverse_loss_layer import ReverseLossLayer
from layers.maf import MAF
from layers.neuralode_fast import NeuralFlowDeformer
from layers.pointnet import PointNetfeat, feature_transform_regularizer
import pyDeform

import numpy as np
from time import time
import trimesh

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
def load_mesh(mesh_path):
	mesh = trimesh.load(mesh_path, process=False)
	verts = torch.from_numpy(mesh.vertices.astype(np.float32))
	edges = torch.from_numpy(mesh.edges.astype(np.int32))
	faces = torch.from_numpy(mesh.faces.astype(np.int32))
	return verts, faces, edges

V1, F1, E1 = load_mesh(source_path)
V2, F2, E2 = load_mesh(reference_path)
GV1 = V1.clone()
GE1 = E1.clone()
GV2 = V2.clone()
GE2 = E2.clone()

# PointNet layer.
pointnet = PointNetfeat(global_feat=True, feature_transform=True)
pointnet = pointnet.to(device)
pointnet.eval()

# Deformation losses layer.
graph_loss = GraphLoss2Layer(
    V1, F1, GV1, GE1, V2, F2, GV2, GE2, rigidity, device)
param_id1 = graph_loss.param_id1
param_id2 = graph_loss.param_id2

reverse_loss = ReverseLossLayer()

# Flow layer.
func = NeuralFlowDeformer(device=device)
func.to(device)

optimizer = optim.Adam(func.parameters(), lr=1e-3)

# Prepare input for encoding the skeleton meshes, shape = [1, 3, n_pts].
GV1_pointnet_input = GV1.unsqueeze(0)
GV1_pointnet_input = GV1_pointnet_input.transpose(2, 1).to(device)
GV2_pointnet_input = GV2.unsqueeze(0)
GV2_pointnet_input = GV2_pointnet_input.transpose(2, 1).to(device)

GV1_origin = GV1.clone()
GV2_origin = GV2.clone()

niter = 100

GV1_device = GV1.to(device)
GV2_device = GV2.to(device)
for it in range(0, niter):
    optimizer.zero_grad()

    # Encode both skeleton meshes using PointNet.
    GV1_features_device, _, GV1_trans_feat = pointnet(GV1_pointnet_input)
    GV2_features_device, _, GV2_trans_feat = pointnet(GV2_pointnet_input)
    GV1_features_device = torch.zeros(1, 32).to(device)
    GV2_features_device = torch.zeros(1, 32).to(device)
    source_target_latents = torch.cat([GV1_features_device, GV2_features_device], dim=0)
    #target_source_latents = torch.cat([GV2_features_device, GV1_features_device], dim=0)

    # Compute and integrate velocity field for deformation.
    GV1_deformed = func.forward(GV1_device, source_target_latents)
    #GV2_deformed = func.inverse(GV2_device, target_source_latents)
    loss1_forward = graph_loss(GV1_deformed, GE1, GV2, GE2, 0)
    loss1_backward = reverse_loss(GV1_deformed, GV2_origin, device)
    loss1_features_reg = FEATURES_REG_LOSS_WEIGHT * \
        feature_transform_regularizer(GV1_trans_feat)

    #loss2_forward = graph_loss(GV1, GE1, GV2_deformed, GE2, 1)
    #loss2_backward = reverse_loss(GV2_deformed, GV1_origin, device)
    loss2_features_reg = FEATURES_REG_LOSS_WEIGHT * \
        feature_transform_regularizer(GV2_trans_feat)

    loss = loss1_forward + loss1_backward + loss1_features_reg + loss2_features_reg
    #loss = loss1_forward + loss1_backward + loss2_forward + \
    #    loss1_features_reg + loss2_backward + loss2_features_reg

    loss.backward()
    optimizer.step()

    if it % 100 == 0 or True:
        print('iter=%d, loss1_forward=%.6f loss1_backward=%.6f'
              % (it, np.sqrt(loss1_forward.item() / GV1.shape[0]),
                 np.sqrt(loss1_backward.item() / GV2.shape[0])))
        """
        print('iter=%d, loss1_forward=%.6f loss1_backward=%.6f loss2_forward=%.6f loss2_backward=%.6f'
              % (it, np.sqrt(loss1_forward.item() / GV1.shape[0]),
                 np.sqrt(loss1_backward.item() / GV2.shape[0]),
                 np.sqrt(loss2_forward.item() / GV2.shape[0]),
                 np.sqrt(loss2_backward.item() / GV1.shape[0])))
        """
        current_loss = loss.item()
"""
# Evaluate final result.
if save_path != '':
    torch.save({'func': func, 'optim': optimizer}, save_path)

V1_copy_direct = V1.clone() 
V1_copy_direct_origin = V1_copy_direct.clone()

# Deform original mesh directly, different from paper.
pyDeform.NormalizeByTemplate(V1_copy_direct, param_id1.tolist())

func.func = func.func.cpu()
# Considering extracting features for the original target mesh here.
V1_copy_direct, _ = func.forward((V1_copy_direct, GV2_features_device.cpu()))
V1_copy_direct = torch.from_numpy(V1_copy_direct.data.cpu().numpy())

src_to_src = torch.from_numpy(
    np.array([i for i in range(V1_copy_direct_origin.shape[0])]).astype('int32'))

pyDeform.SolveLinear(V1_copy_direct_origin, F1, E1, src_to_src, V1_copy_direct, 1, 1)
pyDeform.DenormalizeByTemplate(V1_copy_direct_origin, param_id2.tolist())
pyDeform.SaveMesh(output_path, V1_copy_direct_origin, F1)
"""
