import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))

from torch import nn
import torch.optim as optim
from torch.autograd import Function

import torch
from layers.graph_loss2_layer import GraphLoss2LayerSimple, Finalize
from layers.reverse_loss_layer import ReverseLossLayer
from layers.maf import MAF
from layers.neuralode_fast import NeuralODE
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

def load_mesh(mesh_path):
    mesh = trimesh.load(source_path)
    verts = torch.from_numpy(mesh.vertices.astype(np.float32))
    edges = torch.from_numpy(mesh.edges.astype(np.int32))
    faces = torch.from_numpy(mesh.faces.astype(np.int32))
    return verts, edges, faces

source_verts, source_edges, source_faces = load_mesh(source_path)
targ_verts, targ_edges, targ_faces = load_mesh(reference_path)

graph_loss = GraphLoss2LayerSimple(source_verts, source_faces, source_edges,
                                   targ_verts, targ_faces, targ_edges, rigidity, device)
param_id1 = graph_loss.param_id1
param_id2 = graph_loss.param_id2

reverse_loss = ReverseLossLayer()

func = NeuralODE(device)

optimizer = optim.Adam(func.parameters(), lr=1e-3)
source_verts_origin = source_verts.clone()
targ_verts_origin = targ_verts.clone()

niter = 1000

source_verts_device = source_verts.to(device)
targ_verts_device = targ_verts.to(device)
for it in range(0, niter):
    optimizer.zero_grad()

    source_verts_deformed = func.forward(source_verts_device)
    targ_verts_deformed = func.inverse(targ_verts_device)
    
    loss1_forward = graph_loss(source_verts_deformed, source_edges, targ_verts, targ_edges, 0)
    loss1_backward = reverse_loss(source_verts_deformed, targ_verts_origin, device)

    #loss2_forward = graph_loss(source_verts, source_edges, targ_verts_deformed, targ_edges, 1)
    #loss2_backward = reverse_loss(targ_verts_deformed, source_verts_origin, device)

    loss = loss1_forward + loss1_backward #+ loss2_forward + loss2_backward

    loss.backward()
    optimizer.step()

    if it % 100 == 0 or True:
        """
        print('iter=%d, loss1_forward=%.6f loss1_backward=%.6f loss2_forward=%.6f loss2_backward=%.6f'
            %(it, np.sqrt(loss1_forward.item() / source_verts.shape[0]),
                np.sqrt(loss1_backward.item() / targ_verts.shape[0]),
                np.sqrt(loss2_forward.item() / targ_verts.shape[0]),
                np.sqrt(loss2_backward.item() / source_verts.shape[0])))
        """
        print(it)
        current_loss = loss.item()

if save_path != '':
    torch.save({'func':func, 'optim':optimizer}, save_path)

flow_path = output_path[:-4] + "_flow.txt"
flow_final_path = output_path[:-4] + "_flow_final.txt"

# Deform original mesh directly, different from paper.
source_verts_copy = source_verts.clone() 
source_verts_copy_origin = source_verts_copy.clone()
pyDeform.NormalizeByTemplate(source_verts_copy, param_id1.tolist())

func.func = func.func.cpu()
source_verts_copy = func.forward(source_verts_copy)

# Save intermediate flow from network.
flow = source_verts_copy - source_verts
flow = torch.cat((source_verts, flow), dim=1)
flow_file = open(flow_path, 'w')
np.savetxt(flow_file, flow.detach().numpy())
flow_file.close()

# Solve linear system for final flow.
source_verts_copy = torch.from_numpy(source_verts_copy.data.cpu().numpy())
src_to_src = torch.from_numpy(
    np.array([i for i in range(source_verts_copy_origin.shape[0])]).astype('int32'))
pyDeform.SolveLinear(source_verts_copy_origin, source_faces, source_edges, src_to_src, source_verts_copy, 1, 1)
pyDeform.DenormalizeByTemplate(source_verts_copy_origin, param_id2.tolist())

# Save final flow.
flow_final = source_verts_copy_origin - source_verts
flow_final = torch.cat((source_verts, flow_final), dim=1)
flow_file = open(flow_final_path, 'w')
np.savetxt(flow_file, flow_final.detach().numpy())
flow_file.close()

# Save output mesh.
pyDeform.SaveMesh(output_path, source_verts_copy_origin, source_faces)
