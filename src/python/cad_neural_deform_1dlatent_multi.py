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
from util.load_data import load_mesh_tensors
from util.save_data import save_results
import pyDeform

import numpy as np
from timeit import default_timer as timer

import argparse

parser = argparse.ArgumentParser(description='Rigid Deformation.')
parser.add_argument('--source', default='../data/cad-source.obj')
parser.add_argument('--target', default=[], action='append')
parser.add_argument('--output', default='./cad-output')
parser.add_argument('--rigidity', default='0.1')
parser.add_argument('--device', default='cuda')
parser.add_argument('--save_path', default='./cad-output.ckpt')
parser.add_argument('--num_iter', default=1000)
args = parser.parse_args()

source_path = args.source
reference_paths = args.target
output_path = args.output
rigidity = float(args.rigidity)
save_path = args.save_path
device = torch.device(args.device)


V1, F1, E1, _ = load_mesh_tensors(source_path)
GV1 = V1.clone()
GE1 = E1.clone()

V_targs = []
F_targs = []
E_targs = []
GV_targs = []
GE_targs = []
n_targs = len(reference_paths)
for reference_path in reference_paths:
    V, F, E, _ = load_mesh_tensors(reference_path)
    V_targs.append(V)
    F_targs.append(F)
    E_targs.append(E)
    GV_targs.append(V.clone())
    GE_targs.append(E.clone())

# Deformation losses layer.
graph_loss = GraphLossLayerMulti(
    V1, F1, GV1, GE1, V_targs, F_targs, GV_targs, GE_targs, rigidity, device)
param_id1 = graph_loss.param_id1
param_id_targs = graph_loss.param_id_targs

reverse_loss = ReverseLossLayer()

# Flow layer.
func = NeuralFlowDeformer(dim=3, latent_size=1, method="dopri5", device=device)
func.to(device)

optimizer = optim.Adam(func.parameters(), lr=1e-3)

# Clone skeleton vertices for computing reverse loss.
GV_origin_targs = []
for GV_targ in GV_targs:
    GV_origin_targs.append(GV_targ.clone())

# Move skeleton vertices to device for deformation.
GV1_device = GV1.to(device)

print("Starting training!")

# Compute 1D latent codes
all_source_target_latents = []
stepsize = 1.0 / len(reference_paths)
for i in range(len(reference_paths)):
    start = 0
    end = (i + 1) * stepsize
    all_source_target_latents.append([start, end])
all_source_target_latents = torch.FloatTensor(all_source_target_latents).to(device)
print("all_source_target_latents:", all_source_target_latents)

for it in range(int(args.num_iter)):
    optimizer.zero_grad()
    loss = 0

    for i in range(n_targs):
        source_target_latents = all_source_target_latents[i].unsqueeze(1)
        GV1_deformed = func.forward(GV1_device, source_target_latents)
    
        # Source to target.
        loss1_forward = graph_loss(
            GV1_deformed, GE1, GV_targs[i], GE_targs[i], i, 0)
        loss1_backward = reverse_loss(GV1_deformed, GV_origin_targs[i], device)
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
    latent_code = all_source_target_latents[i].unsqueeze(1)
    output = output_path[:-4] + "_" + str(i+1).zfill(2) + ".obj"
    
    save_results(V1, F1, E1, V_targs[i], F_targs[i], func, param_id1.tolist(), \
            param_id_targs[i].tolist(), output, device, latent_code)

