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
from layers.graph_loss2_layer import GraphLoss2LayerMulti, Finalize
from layers.reverse_loss_layer import ReverseLossLayer
from layers.maf import MAF
from layers.neuralode_fast import NeuralFlowDeformer
from layers.pointnet import PointNetfeat, feature_transform_regularizer
from util.samplers import fps, sample_faces
import pyDeform

import numpy as np
from time import time
import trimesh

import argparse

parser = argparse.ArgumentParser(description='Rigid Deformation.')
parser.add_argument('--source', default='../data/cad-source.obj')
parser.add_argument('--target', default=[], action='append')
parser.add_argument('--output', default='./cad-output')
parser.add_argument('--rigidity', default='0.1')
parser.add_argument('--device', default='cuda')
parser.add_argument('--save_path', default='./cad-output.ckpt')

args = parser.parse_args()

source_path = args.source
reference_paths = args.target
output_path = args.output
rigidity = float(args.rigidity)
save_path = args.save_path
device = torch.device(args.device)

FEATURES_REG_LOSS_WEIGHT = 0.001


def load_mesh(mesh_path, intermediate=10000, final=2048):
    mesh = trimesh.load(mesh_path, process=False)
    V = mesh.vertices.astype(np.float32)
    F = mesh.faces.astype(np.int32)
    E = mesh.edges.astype(np.int32)
    
    # Surface vertices for PointNet input.
    V_sample, _, _ = sample_faces(V, F, n_samples=intermediate)
    V_sample, _ = fps(V_sample, final)
    V_sample = V_sample.astype(np.float32)

    return torch.from_numpy(V), torch.from_numpy(F), torch.from_numpy(E), torch.from_numpy(V_sample)

V1, F1, E1, V1_surf = load_mesh(source_path)
GV1 = V1.clone()
GE1 = E1.clone()

V_targs = []
V_surf_targs = []
F_targs = []
E_targs = []
GV_targs = []
GE_targs = []
n_targs = len(reference_paths)

# TODO(connorzl): pad GV and GE properly for batching.
for reference_path in reference_paths:
    V, F, E, V_surf = load_mesh(reference_path)
    V_targs.append(V)
    F_targs.append(F)
    E_targs.append(E)
    V_surf_targs.append(V_surf)
    GV_targs.append(V.clone())
    GE_targs.append(E.clone())

# Deformation losses layer.
graph_loss = GraphLoss2LayerMulti(
    V1, F1, GV1, GE1, V_targs, F_targs, GV_targs, GE_targs, rigidity, device)
param_id1 = graph_loss.param_id1
param_id_targs = graph_loss.param_id_targs

reverse_loss = ReverseLossLayer()

# Flow layer.
func = NeuralFlowDeformer(dim=3, latent_size=1, device=device)
func.to(device)

optimizer = optim.Adam(func.parameters(), lr=1e-3)

# Clone skeleton vertices for computing loss.
GV1_origin = GV1.clone()
GV_origin_targs = []
for GV_targ in GV_targs:
    GV_origin = GV_targ.clone()
    GV_origin_targs.append(GV_origin)

# Move skeleton vertices to device for deformation.
GV1_device = GV1.to(device)
GV_device_targs = []
for GV in GV_targs:
    GV_targ = GV.to(device)
    GV_device_targs.append(GV_targ)

niter = 1000
loss_min = 1e30
eps = 1e-6
print("Starting training!")
#all_source_target_latents = torch.FloatTensor([[0, 0.5], [0, 1.0]]).to(device)
all_source_target_latents = torch.FloatTensor([[0, 0.25], [0, 0.5], [0, 0.75], [0, 1.0]]).to(device)
for it in range(0, niter):
    optimizer.zero_grad()

    loss = 0

    for i in range(n_targs):
        source_target_latents = all_source_target_latents[i].unsqueeze(1)
        
        # Compute and integrate velocity field for deformation.
        GV1_deformed = func.forward(GV1_device, source_target_latents)

        # Same as above, but in opposite direction.
        #GV2_deformed, GV1_features_deformed = func.inverse(
        #    (GV_device_targs[i], GV1_features_device))
    
        # Source to target.
        loss1_forward = graph_loss(
            GV1_deformed, GE1, GV_targs[i], GE_targs[i], i, 0)
        loss1_backward = reverse_loss(GV1_deformed, GV_origin_targs[i], device)

        # Target to source.
        #loss2_forward = graph_loss(GV1, GE1, GV2_deformed, GE_targs[i], i, 1)
        #loss2_backward = reverse_loss(GV2_deformed, GV1_origin, device)

        # Total loss.
        loss += loss1_forward + loss1_backward #+ loss2_forward + loss2_backward

        if it % 100 == 0 or True:
            print('iter= % d, target_index= % d loss1_forward= % .6f loss1_backward= % .6f'
                  % (it, i, np.sqrt(loss1_forward.item() / GV1.shape[0]),
                     np.sqrt(loss1_backward.item() / GV_targs[i].shape[0])))
    loss.backward()
    optimizer.step()

# Evaluate final result.
if save_path != '':
    torch.save({'func': func, 'optim': optimizer}, save_path)

V1_copy = V1.clone() 
src_to_src = torch.from_numpy(
    np.array([i for i in range(V1_copy.shape[0])]).astype('int32'))

# Deform original mesh directly, different from paper.
pyDeform.NormalizeByTemplate(V1_copy, param_id1.tolist())

V1_copy = V1_copy.to(device)

results = []

for i in range(n_targs):
    source_target_latents = all_source_target_latents[i].unsqueeze(1)
    V1_deformed = func.forward(V1_copy, source_target_latents)

    # Move to CPU
    results.append(torch.from_numpy(V1_deformed.detach().cpu().numpy()))

for i, deformed_result in enumerate(results):
    V1_copy_origin = V1.clone()
    pyDeform.SolveLinear(V1_copy_origin, F1, E1, src_to_src, deformed_result, 1, 1)
    pyDeform.DenormalizeByTemplate(V1_copy_origin, param_id_targs[i].tolist())

    target_output_path = output_path[:-4] + "_" + str(i+1).zfill(2) + ".obj"
    pyDeform.SaveMesh(target_output_path, V1_copy_origin, F1)
