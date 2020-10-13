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
from layers.flow import FlowNetwork
from layers.pointnet_ae import Network
import layers.pointnet_plus_frame
from util.load_data import compute_deformation_pairs, load_neural_deform_data, load_segmentation
from util.save_data import save_snapshot_results
import pyDeform
import numpy as np
import argparse
from types import SimpleNamespace

parser = argparse.ArgumentParser(description='Rigid Deformation.')
parser.add_argument('--input', default=[], action='append')
parser.add_argument('--output_prefix', default='./cad-output')
parser.add_argument('--all_pairs', action='store_true')
parser.add_argument('--rigidity', default='0.1')
parser.add_argument('--device', default='cuda')
parser.add_argument('--save_path', default='./cad-output.ckpt')
parser.add_argument('--pretrained_pointnet_ckpt_path', default='')
parser.add_argument('--num_iter', default=1000)
args = parser.parse_args()

output_prefix = args.output_prefix
rigidity = float(args.rigidity)
save_path = args.save_path
device = torch.device(args.device)

# Load meshes.
(V_all, F_all, E_all, V_surf_all), (GV_all, GE_all) = \
        load_neural_deform_data(args.input, device)
part_sizes_all = load_segmentation(args.input, [0, 1, 2, 3, 4])

# Compute all deformation pairs.
deformation_pairs = compute_deformation_pairs(args.all_pairs, len(args.input))

# PointNet layer.
pointnet_conf = SimpleNamespace(num_point=2048, decoder_type='fc', loss_type='emd')
pointnet = Network(pointnet_conf)
pointnet.load_state_dict(torch.load(args.pretrained_pointnet_ckpt_path, map_location=device))
pointnet.eval()
pointnet = pointnet.to(device)

# Deformation losses layer, which normalizes vertices.
graph_loss = GraphLossLayerPairs(V_all, F_all, GV_all, GE_all, rigidity, device)
param_ids = graph_loss.param_ids
reverse_loss = ReverseLossLayer()

# For computing loss.
GV_origin_all = []
GV_parts_device_all = []
for i, GV in enumerate(GV_all):
    print("GV:", GV.shape)
    print("part_sizes_all:", part_sizes_all[i])
    GV_origin_all.append(GV.clone())
    GV_parts_device = torch.split(GV.clone(), part_sizes_all[i], dim=0)
    GV_parts_device = [part.to(device) for part in GV_parts_device]
    GV_parts_device_all.append(GV_parts_device)

# Flow layer.
deformers = []
num_parts = len(GV_parts_device_all[0])
for part in GV_parts_device_all:
    #func = FlowNetwork("mlp")
    func = FlowNetwork("pointnet_plus_frame")
    func.to(device)
    deformers.append(func)

all_deformer_params = []
for deformer in deformers:
    all_deformer_params += list(deformer.parameters())
optimizer = optim.Adam(all_deformer_params, lr=1e-3)

# Prepare PointNet input.
GV_pointnet_inputs = torch.stack(V_surf_all, dim=0).to(device)
_, GV_features = pointnet(GV_pointnet_inputs)
GV_features = GV_features.detach()

def extract_rot_trans(frame):
    """
        Input: 1 x 12
        Output: 1 x 3 x 3, 1 x 3 x 1
    """
    rot = frame[:, 0:9].unsqueeze(1)
    rot = torch.reshape(rot, (1, 3, 3))

    (U, S, V) = torch.svd(rot)
    VT = torch.transpose(V, 1, 2)
    sign = torch.sign(torch.det(torch.matmul(U, VT)))
    
    S_rot = S.clone()
    S_rot[:, 0] = 1.0
    S_rot[:, 1] = 1.0
    S_rot[:, 2] = sign
    S_rot = torch.diag_embed(S_rot)
    
    rot = torch.matmul(torch.matmul(U, S_rot), VT)
    trans = frame[:, 9:12].unsqueeze(2)
    return rot, trans

print("Starting training!")
for it in range(int(args.num_iter)):
    optimizer.zero_grad()

    loss = 0
    for i, (src, targ) in enumerate(deformation_pairs):
        # Perform deformation.
        deformed_parts = []
        for j in range(num_parts):
            V_src = GV_parts_device_all[src][j]
            V_targ = GV_parts_device_all[targ][j]
            frame_src = deformers[j].forward(V_src.unsqueeze(0), GV_features[src])
            frame_targ = deformers[j].forward(V_targ.unsqueeze(0), GV_features[targ])
            
            # Extract 3D coordinate frame + origin for source part and target part.
            src_rot, src_trans = extract_rot_trans(frame_src) 
            targ_rot, targ_trans = extract_rot_trans(frame_targ) 
            
            # Transform source vertices to world coordinates.
            V_deformed = torch.unsqueeze(V_src, dim=2)
            V_deformed = torch.matmul(torch.transpose(src_rot, 1, 2), V_deformed - src_trans)
            V_deformed = torch.matmul(targ_rot, V_deformed) + targ_trans
            deformed_parts.append(torch.squeeze(V_deformed, dim=2))
        GV_deformed = torch.cat(deformed_parts, dim=0)

        # Compute losses.
        loss_forward = graph_loss(
            GV_deformed, GE_all[src], GV_all[targ], GE_all[targ], \
                    param_ids[src].tolist(), param_ids[targ].tolist(), 0)
        loss_backward = reverse_loss(GV_deformed, GV_origin_all[targ], device)
        loss += loss_forward + loss_backward 
        print('iter= %d, source_index= %d, target_index= %d, loss_forward= %.6f, loss_backward= %.6f'
              % (it, src, targ, np.sqrt(loss_forward.item() / GV_all[src].shape[0]),
                 np.sqrt(loss_backward.item() / GV_all[targ].shape[0])))

        if it % 50 == 0 or it == int(args.num_iter) - 1:
            if it % 50 == 0:
                print("Saving snapshot...")
                output = output_prefix + "_snapshot_" + str(it).zfill(4) + "_" + \
                    str(src).zfill(2) + "_" + str(targ).zfill(2) + ".obj"
            elif it == int(args.num_iter) - 1:
                print("Saving final result...")
                output = output_prefix + "_" + str(it).zfill(4) + "_" + \
                    str(src).zfill(2) + "_" + str(targ).zfill(2) + ".obj"
            save_snapshot_results(V_all[src], GV_deformed, F_all[src], E_all[src], \
                    V_all[targ], F_all[targ], param_ids[src].tolist(), \
                    param_ids[targ].tolist(), output)
    loss.backward()
    optimizer.step()

# Evaluate final result.
if save_path != '':
    torch.save({'func': func, 'optim': optimizer}, save_path)

