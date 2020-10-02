import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'util')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))


import torch.optim as optim
import torch
from layers.graph_loss_layer import GraphLossLayerMulti
from layers.reverse_loss_layer import ReverseLossLayer
from layers.neuralode_conditional import NeuralFlowDeformer
from layers.pointnet_ae import Network
from util.load_data import load_mesh_tensors
from util.save_data import save_results
import pyDeform

import numpy as np
from time import time

import argparse
from types import SimpleNamespace

parser = argparse.ArgumentParser(description='Rigid Deformation.')
parser.add_argument('--source', default='../data/cad-source.obj')
parser.add_argument('--target', default=[], action='append')
parser.add_argument('--output', default='./cad-output')
parser.add_argument('--rigidity', default='0.1')
parser.add_argument('--device', default='cuda')
parser.add_argument('--save_path', default='./cad-output.ckpt')
parser.add_argument('--pretrained_pointnet_ckpt_path', default='')
parser.add_argument('--num_iter', default=1000)
args = parser.parse_args()

source_path = args.source
reference_paths = args.target
output_path = args.output
rigidity = float(args.rigidity)
save_path = args.save_path
device = torch.device(args.device)

V1, F1, E1, V1_surf = load_mesh_tensors(source_path)

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
    V, F, E, V_surf = load_mesh_tensors(reference_path)
    V_targs.append(V)
    F_targs.append(F)
    E_targs.append(E)
    V_surf_targs.append(V_surf)
    GV_targs.append(V_targs[-1].clone())
    GE_targs.append(E_targs[-1].clone())

# PointNet layer.
pointnet_conf = SimpleNamespace(num_point=2048, decoder_type='fc', loss_type='emd')
pointnet = Network(pointnet_conf)
pointnet.load_state_dict(torch.load(args.pretrained_pointnet_ckpt_path, map_location=device))
pointnet.eval()
pointnet = pointnet.to(device)

# Deformation losses layer.
graph_loss = GraphLossLayerMulti(
    V1, F1, GV1, GE1, V_targs, F_targs, GV_targs, GE_targs, rigidity, device)
param_id1 = graph_loss.param_id1
param_id_targs = graph_loss.param_id_targs

reverse_loss = ReverseLossLayer()

# Flow layer.
func = NeuralFlowDeformer(dim=3, latent_size=1024, device=device)
func.to(device)

optimizer = optim.Adam(func.parameters(), lr=1e-3)

# Prepare PointNet input.
GV1_pointnet_input = V1_surf.unsqueeze(0).to(device)
GV_pointnet_input_targs = torch.stack(V_surf_targs, dim=0).to(device)

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

_, GV1_features = pointnet(GV1_pointnet_input)
_, GV_targ_features = pointnet(GV_pointnet_input_targs)
GV1_features = GV1_features.detach()
GV_targ_features = GV_targ_features.detach()
source_target_latents = []
for i in range(n_targs):
    source_target_latents.append(torch.cat([GV1_features, GV_targ_features[i, :].unsqueeze(0)], dim=0))

print("Starting training!")
for it in range(int(args.num_iter)):
    optimizer.zero_grad()

    loss = 0
    for i in range(n_targs):
        # Compute and integrate velocity field for deformation.
        GV1_deformed = func.forward(GV1_device, source_target_latents[i])

        # Source to target.
        loss1_forward = graph_loss(
            GV1_deformed, GE1, GV_targs[i], GE_targs[i], i, 0)
        loss1_backward = reverse_loss(GV1_deformed, GV_origin_targs[i], device)

        # Total loss.
        loss += loss1_forward + loss1_backward 

        if it % 100 == 0 or True:
            print('iter= % d, target_index= % d loss1_forward= % .6f loss1_backward= % .6f'
                  % (it, i, np.sqrt(loss1_forward.item() / GV1.shape[0]),
                     np.sqrt(loss1_backward.item() / GV_targs[i].shape[0])))
    loss.backward()
    optimizer.step()

# Evaluate final result.
if save_path != '':
    torch.save({'func': func, 'optim': optimizer}, save_path)

for i in range(n_targs):
    latent_code = torch.cat([GV1_features, GV_targ_features[i, :].unsqueeze(0)], dim=0)
    output = output_path[:-4] + "_" + str(i+1).zfill(2) + ".obj"

    save_results(V1, F1, E1, V_targs[i], F_targs[i], func, param_id1.tolist(), \
            param_id_targs[i].tolist(), output, device, latent_code)

