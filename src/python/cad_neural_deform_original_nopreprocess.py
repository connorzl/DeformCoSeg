import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))

import torch.optim as optim
import torch
from layers.graph_loss_layer import GraphLossLayer
from layers.reverse_loss_layer import ReverseLossLayer
from layers.neuralode import NeuralODE
from util.load_data import load_mesh_tensors
from util.save_data import save_results
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
parser.add_argument('--num_iter', default=1000)
args = parser.parse_args()

source_path = args.source
reference_path = args.target
output_path = args.output
rigidity = float(args.rigidity)
save_path = args.save_path
device = torch.device(args.device)

V1, F1, E1, _ = load_mesh_tensors(source_path)
V2, F2, E2, _ = load_mesh_tensors(reference_path)
GV1 = V1.clone()
GE1 = E1.clone()
GV2 = V2.clone()
GE2 = E2.clone()

graph_loss = GraphLossLayer(V1,F1,GV1,GE1,V2,F2,GV2,GE2,rigidity,device)
param_id1 = graph_loss.param_id1
param_id2 = graph_loss.param_id2

reverse_loss = ReverseLossLayer()

func = NeuralODE(device)

optimizer = optim.Adam(func.parameters(), lr=1e-3)
GV1_origin = GV1.clone()
GV2_origin = GV2.clone()

GV1_device = GV1.to(device)
GV2_device = GV2.to(device)
for it in range(int(args.num_iter)):
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

save_results(V1, F1, E1, V2, F2, func, param_id1.tolist(), param_id2.tolist(), \
        output_path, device)

