import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))

import torch.optim as optim
import torch
from layers.graph_loss2_layer import GraphLoss2Layer
from layers.reverse_loss_layer import ReverseLossLayer
from layers.neuralode import NeuralODE
import pyDeform

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


V1, F1, E1, V2G1, GV1, GE1 = pyDeform.LoadCadMesh(source_path)
V2, F2, E2, V2G2, GV2, GE2 = pyDeform.LoadCadMesh(reference_path)

graph_loss = GraphLoss2Layer(V1,F1,GV1,GE1,V2,F2,GV2,GE2,rigidity,device)
param_id1 = graph_loss.param_id1
param_id2 = graph_loss.param_id2

reverse_loss = ReverseLossLayer()

func = NeuralODE(device)

optimizer = optim.Adam(func.parameters(), lr=1e-3)
GV1_origin = GV1.clone()
GV2_origin = GV2.clone()

niter = 1000

GV1_device = GV1.to(device)
GV2_device = GV2.to(device)
loss_min = 1e30
for it in range(0, niter):
	optimizer.zero_grad()

	GV1_deformed = func.forward(GV1_device)
	GV2_deformed = func.inverse(GV2_device)

	loss1_forward = graph_loss(GV1_deformed, GE1, GV2, GE2, 0)
	loss1_backward = reverse_loss(GV1_deformed, GV2_origin, device)

	loss2_forward = graph_loss(GV1, GE1, GV2_deformed, GE2, 1)
	loss2_backward = reverse_loss(GV2_deformed, GV1_origin, device)

	loss = loss1_forward + loss1_backward + loss2_forward + loss2_backward

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

V1_copy_direct = V1.clone() 
V1_copy_direct_origin = V1_copy_direct.clone()

flow_path = output_path[:-4] + "_flow.txt")
flow_final_path = output_path[:-4] + "_flow_final.txt")

# Deform original mesh directly, different from paper.
pyDeform.NormalizeByTemplate(V1_copy_direct, param_id1.tolist())

func.func = func.func.cpu()
# Considering extracting features for the original target mesh here.
V1_copy_direct = func.forward(V1_copy_direct)
flow = V1_copy_direct - V1
flow = torch.cat((V1, flow), dim=1)
flow_file = open(flow_path, 'w')
np.savetxt(flow_file, flow.detach().numpy())
flow_file.close()

V1_copy_direct = torch.from_numpy(V1_copy_direct.data.cpu().numpy())
src_to_src = torch.from_numpy(
    np.array([i for i in range(V1_copy_direct_origin.shape[0])]).astype('int32'))

pyDeform.SolveLinear(V1_copy_direct_origin, F1, E1, src_to_src, V1_copy_direct, 1, 1)
pyDeform.DenormalizeByTemplate(V1_copy_direct_origin, param_id2.tolist())

flow_final = V1_copy_direct_origin - V1
flow_final = torch.cat((V1, flow_final), dim=1)
flow_file = open(flow_final_path, 'w')
np.savetxt(flow_file, flow_final.detach().numpy())
flow_file.close()

pyDeform.SaveMesh(output_path, V1_copy_direct_origin, F1)
