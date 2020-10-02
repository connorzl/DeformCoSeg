import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'util')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))

import torch.optim as optim
import torch
from layers.graph_loss_layer import GraphLossLayerPairs
from layers.reverse_loss_layer import ReverseLossLayer
from layers.neuralode import NeuralFlowDeformer
from layers.pointnet import PointNetfeat
from util.load_data import compute_deformation_pairs, load_neural_deform_data
from util.save_data import save_results
import pyDeform
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Rigid Deformation.')
parser.add_argument('--input', default=[], action='append')
parser.add_argument('--output_prefix', default='./cad-output')
parser.add_argument('--all_pairs', action='store_true')
parser.add_argument('--rigidity', default='0.1')
parser.add_argument('--device', default='cuda')
parser.add_argument('--save_path', default='./cad-output.ckpt')
parser.add_argument('--num_iter', default=1000)
args = parser.parse_args()

output_prefix = args.output_prefix
rigidity = float(args.rigidity)
save_path = args.save_path
device = torch.device(args.device)

# Load meshes.
(V_all, F_all, E_all, V_surf_all), (GV_all, GE_all, GV_origin_all, GV_device_all) = \
        load_neural_deform_data(args.input, device)

# Compute all deformation pairs.
deformation_pairs = compute_deformation_pairs(args.all_pairs, len(args.input))

# PointNet layer.
pointnet = PointNetfeat()
pointnet = pointnet.to(device)

# Deformation losses layer.
graph_loss = GraphLossLayerPairs(V_all, F_all, GV_all, GE_all, rigidity, device)
param_ids = graph_loss.param_ids
reverse_loss = ReverseLossLayer()

# Flow layer.
func = NeuralFlowDeformer(dim=3, latent_size=1024, device=device)
func.to(device)
optimizer = optim.Adam(func.parameters(), lr=1e-3)

# Prepare input for encoding the skeleton meshes, shape = [1, 3, n_pts].
GV_pointnet_inputs = torch.stack(V_surf_all, dim=0)
GV_pointnet_inputs = GV_pointnet_inputs.transpose(2, 1).to(device)

print("Starting training!")
for it in range(int(args.num_iter)):
    optimizer.zero_grad()

    # Encode source skeleton mesh.
    GV_features = pointnet(GV_pointnet_inputs)

    loss = 0
    for i, (src, targ) in enumerate(deformation_pairs):
        # Compute and integrate velocity field for deformation.
        source_target_latents = torch.stack([GV_features[src], GV_features[targ]], dim=0)
        GV_deformed = func.forward(GV_device_all[src], source_target_latents)
        
        # Source to target.
        loss_forward = graph_loss(
            GV_deformed, GE_all[src], GV_all[targ], GE_all[targ], src, targ, 0)
        loss_backward = reverse_loss(GV_deformed, GV_origin_all[targ], device)

        # Total loss.
        loss += loss_forward + loss_backward
        if it % 100 == 0 or True:
            print('iter= %d, source_index= %d, target_index= %d, loss_forward= %.6f, loss_backward= %.6f'
                  % (it, src, targ, np.sqrt(loss_forward.item() / GV_all[src].shape[0]),
                     np.sqrt(loss_backward.item() / GV_all[targ].shape[0])))
    loss.backward()
    optimizer.step()

# Evaluate final result.
if save_path != '':
    torch.save({'func': func, 'optim': optimizer}, save_path)

GV_features = pointnet(GV_pointnet_inputs)
for i, (src, targ) in enumerate(deformation_pairs):
    latent_code = torch.stack([GV_features[src], GV_features[targ]], dim=0)
    output = output_prefix + "_" + str(src).zfill(2) + "_" + str(targ).zfill(2) + ".obj"
    save_results(V_all[src], F_all[src], E_all[src], V_all[targ], F_all[targ], func, \
            param_ids[src].tolist(), param_ids[targ].tolist(), output, device, latent_code)

