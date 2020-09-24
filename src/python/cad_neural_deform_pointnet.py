import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'util')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))

from torch import nn
import torch.optim as optim
from torch.autograd import Function

import torch
from layers.graph_loss2_layer import GraphLoss2Layer, Finalize
from layers.reverse_loss_layer import ReverseLossLayer
from layers.maf import MAF
from layers.neuralode_fast import NeuralFlowDeformer
from layers.pointnet import PointNetfeat
from util.samplers import load_mesh
import pyDeform

import numpy as np
from timeit import default_timer as timer

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


V1, F1, E1, V1_surf = load_mesh(source_path)
V1 = torch.from_numpy(V1)
F1 = torch.from_numpy(F1)
E1 = torch.from_numpy(E1)
V1_surf = torch.from_numpy(V1_surf)

V2, F2, E2, V2_surf = load_mesh(source_path)
V2 = torch.from_numpy(V2)
F2 = torch.from_numpy(F2)
E2 = torch.from_numpy(E2)
V2_surf = torch.from_numpy(V2_surf)

GV1 = V1.clone()
GE1 = E1.clone()
GV2 = V2.clone()
GE2 = E2.clone()

print("Source num vertices:", V1.shape)
print("Target num vertices:", V2.shape)

# PointNet layer.
pointnet = PointNetfeat()
pointnet = pointnet.to(device)

# Deformation losses layer.
graph_loss = GraphLoss2Layer(
    V1, F1, GV1, GE1, V2, F2, GV2, GE2, rigidity, device)
param_id1 = graph_loss.param_id1
param_id2 = graph_loss.param_id2

reverse_loss = ReverseLossLayer()

# Flow layer.
func = NeuralFlowDeformer(dim=3, latent_size=1024, device=device)
func.to(device)

# Prepare input for encoding the skeleton meshes, shape = [1, 3, n_pts].
V1_pointnet_input = V1_surf.unsqueeze(0)
V1_pointnet_input = V1_pointnet_input.transpose(2, 1).to(device)
V2_pointnet_input = V2_surf.unsqueeze(0)
V2_pointnet_input = V2_pointnet_input.transpose(2, 1).to(device)

GV1_origin = GV1.clone()
GV2_origin = GV2.clone()

GV1_device = GV1.to(device)
GV2_device = GV2.to(device)

optimizer = optim.Adam(func.parameters(), lr=1e-3)
niter = 1000
for it in range(0, niter):
    optimizer.zero_grad()

    model_start = timer() 
    # Encode both skeleton meshes using PointNet.
    V1_features_device = pointnet(V1_pointnet_input)
    V2_features_device = pointnet(V2_pointnet_input)
    source_target_latents = torch.cat([V1_features_device, V2_features_device], dim=0)
    GV1_deformed = func.forward(GV1_device, source_target_latents)
    model_end = timer()
    
    losses_forward_start = timer()
    loss1_forward = graph_loss(GV1_deformed, GE1, GV2, GE2, 0)
    loss1_backward = reverse_loss(GV1_deformed, GV2_origin, device)
    loss = loss1_forward + loss1_backward
    losses_forward_end = timer()

    loss_backward_start = timer()
    loss.backward()
    loss_backward_end = timer()

    optimizer_start = timer()
    optimizer.step()
    optimizer_end = timer()

    print("model_time:", model_end - model_start)
    print("losses_forward_time:", losses_forward_end - losses_forward_start)
    print("loss_backward_time:", loss_backward_end - loss_backward_start)
    print("optimizer_time:", optimizer_end - optimizer_start)

    if it % 100 == 0 or True:
        print('iter=%d, loss1_forward=%.6f loss1_backward=%.6f'
              % (it, np.sqrt(loss1_forward.item() / GV1.shape[0]),
                 np.sqrt(loss1_backward.item() / GV2.shape[0])))
        current_loss = loss.item()

# Evaluate final result.
if save_path != '':
    torch.save({'func': func, 'optim': optimizer}, save_path)

V1_copy_direct = V1.clone() 
V1_copy_direct_origin = V1_copy_direct.clone()

# Deform original mesh directly, different from paper.
pyDeform.NormalizeByTemplate(V1_copy_direct, param_id1.tolist())

V1_features_device = pointnet(V1_pointnet_input)
V2_features_device = pointnet(V2_pointnet_input)
source_target_latents = torch.cat([V1_features_device, V2_features_device], dim=0)

V1_copy_direct = V1_copy_direct.to(device)
V1_copy_direct = func.forward(V1_copy_direct, source_target_latents)

# Move to CPU, initialize references mapping.
V1_copy_direct = torch.from_numpy(V1_copy_direct.detach().cpu().numpy())
src_to_src = torch.from_numpy(
    np.array([i for i in range(V1_copy_direct_origin.shape[0])]).astype('int32'))

pyDeform.SolveLinear(V1_copy_direct_origin, F1, E1, src_to_src, V1_copy_direct, 1, 1)
pyDeform.DenormalizeByTemplate(V1_copy_direct_origin, param_id2.tolist())
pyDeform.SaveMesh(output_path, V1_copy_direct_origin, F1)

