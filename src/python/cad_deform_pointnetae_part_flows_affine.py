import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'util')
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))
import torch.optim as optim
import torch
from collections import OrderedDict
from layers.graph_loss_layer import GraphLossLayerBatch
from layers.reverse_loss_layer import ReverseLossLayer

from layers.pointnet_ae import Network
#from layers.pointnet_local_features import PointNetSeg
#from layers.pointnet_plus_correlate import PointNetCorrelate
from layers.pointnet_affine import PointNetFrame#, PointNetMask
#from pointnet2_ops.pointnet2_modules import PointnetFPModule
#from layers.pointnet_plus_mask import PointNet2
#from layers.pointnet_plus_frame import PointNetFrame

from torch.utils.data import DataLoader
from util.load_data import collate
from util.save_data import save_snapshot_results
from util.dataloader import SAPIENMesh, RandomPairSampler
import pyDeform
import numpy as np
import argparse
from types import SimpleNamespace

parser = argparse.ArgumentParser(description='Rigid Deformation.')
parser.add_argument('--input', default='')
parser.add_argument('--output_prefix', default='./cad-output')
parser.add_argument('--rigidity', default='0.1')
parser.add_argument('--device', default='cuda')
parser.add_argument('--epochs', default=1000)
parser.add_argument('--pointnet_ckpt', default='')
parser.add_argument('--batchsize', default=1)
args = parser.parse_args()

EPOCH_SNAPSHOT_INTERVAL = 50
RANDOM_SEED = 1
LATENT_SIZE = 32

output_prefix = args.output_prefix
rigidity = float(args.rigidity)
device = torch.device(args.device)
batchsize = int(args.batchsize)
epochs = int(args.epochs)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

train_dataset = SAPIENMesh(args.input)
train_sampler = RandomPairSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,
                          drop_last=True, sampler=train_sampler, collate_fn=collate)

def extract_rot_trans(frame, i):
    """
        Input: 1 x 12
        Output: 1 x 3 x 3, 1 x 3 x 1
    """
       
    rot_start = 12 * i
    rot_end = rot_start + 9
    trans_start = rot_end 
    trans_end = trans_start + 3

    rot = frame[:, rot_start:rot_end].unsqueeze(2)
    rot = torch.reshape(rot, (1, 3, 3))
    (U, S, V) = torch.svd(rot)
    VT = torch.transpose(V, 1, 2)
    det_UVT = torch.det(torch.matmul(U, VT))

    S_rot = S.clone()
    S_rot[:, 0] = 1.0
    S_rot[:, 1] = 1.0
    S_rot[:, 2] = det_UVT
    S_rot = torch.diag_embed(S_rot)
    rot = torch.matmul(torch.matmul(U, S_rot), VT)
    """
    d = frame[:, 0:3].squeeze(0)
    d = d / torch.norm(d)
    t = frame[:, 3]
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    R_00 = cos_t + d[0] * d[0] * (1.0 - cos_t)
    R_01 = d[0] * d[1] * (1.0 - cos_t) - d[2] * sin_t
    R_02 = d[0] * d[2] * (1.0 - cos_t) + d[1] * sin_t
    
    R_10 = d[0] * d[1] * (1.0 - cos_t) + d[2] * sin_t
    R_11 = cos_t + d[1] * d[1] * (1.0 - cos_t)
    R_12 = d[1] * d[2] * (1.0 - cos_t) - d[0] * sin_t
    
    R_20 = d[2] * d[0] * (1.0 - cos_t) - d[1] * sin_t
    R_21 = d[2] * d[1] * (1.0 - cos_t) + d[0] * sin_t
    R_22 = cos_t + d[2] * d[2] * (1.0 - cos_t)
    rot = torch.tensor([[R_00, R_01, R_02], [R_10, R_11, R_12],
                        [R_20, R_21, R_22]]).to(frame.device).unsqueeze(0)
    """
    trans = frame[:, 4:7].unsqueeze(2)
    return rot, trans

parameters = []
"""
pointnet_local_features = PointNetSeg(LATENT_SIZE)
parameters += list(pointnet_local_features.parameters())
pointnet_local_features = pointnet_local_features.to(device)

pointnet_correlate = PointNetCorrelate(input_features=LATENT_SIZE+1, output_features=LATENT_SIZE)
parameters += list(pointnet_correlate.parameters())
pointnet_correlate = pointnet_correlate.to(device)

feature_propagator = PointnetFPModule(mlp=[LATENT_SIZE, LATENT_SIZE, LATENT_SIZE], bn=False)
parameters += list(feature_propagator.parameters())
feature_propagator.to(device)
"""
# PointNet layer.
pointnet_conf = SimpleNamespace(
    num_point=2048, decoder_type='fc', loss_type='emd')
pointnet = Network(pointnet_conf, 1024)
pointnet.load_state_dict(torch.load(
    args.pointnet_ckpt, map_location=device))
pointnet.eval()
pointnet = pointnet.to(device)

# Mask Network.
NUM_PARTS = 2
#mask_network = PointNet2(NUM_PARTS)
#mask_network = mask_network.to(device)

# Flow layer.
keyboard_network = PointNetFrame(1028, NUM_PARTS-1)
keyboard_network = keyboard_network.to(device)

screen_network = PointNetFrame(1028, NUM_PARTS-1)
screen_network = screen_network.to(device)

graph_loss = GraphLossLayerBatch(rigidity, device)
reverse_loss = ReverseLossLayer()

parameters += list(screen_network.parameters())
optimizer = optim.Adam(parameters, lr=1e-3)

for epoch in range(epochs):
    for batch_idx, data_tensors in enumerate(train_loader):    
        optimizer.zero_grad()
        loss = 0
        loss_forward = 0
        loss_backward = 0
        
        # Retrieve data for deformation
        src, tar, src_param, tar_param, src_data, tar_data, \
            src_sample, tar_sample, src_mask, tar_mask = data_tensors
        V_src, F_src, E_src, GV_src, GE_src = src_data
        V_tar, F_tar, E_tar, GV_tar, GE_tar = tar_data
        """
        # Extract local per-point features for each shape.
        src_sample_device = src_sample.to(device)
        tar_sample_device = tar_sample.to(device)
        src_sample_features = pointnet_local_features(src_sample_device)
        tar_sample_features = pointnet_local_features(tar_sample_device)

        # Correlation input.
        num_samples = src_sample_device.shape[1]
        src_sample_correlation = torch.cat([src_sample_device, src_sample_features, torch.zeros(
            batchsize, num_samples, 1).to(device)], dim=2)
        tar_sample_correlation = torch.cat([tar_sample_device, tar_sample_features, torch.ones(
            batchsize, num_samples, 1).to(device)], dim=2)
        correlation_sample_input = torch.cat([src_sample_correlation, tar_sample_correlation], dim=1)
        sample_flow_features = pointnet_correlate(correlation_sample_input)
        sample_flow_features = sample_flow_features[:, :num_samples, :]
        """
        # Prepare PointNet input.
        GV_pointnet_inputs = torch.cat([src_sample, tar_sample], dim=0).to(device)
        _, GV_features = pointnet(GV_pointnet_inputs)

        # Deform each (src, tar) pair.
        for k in range(batchsize):
            # Copies of normalized GV for deformation training.
            GV_tar_origin = GV_tar[k].clone()
            GV_src_device = GV_src[k].to(device)
           
            src_sample_device = src_sample[k].to(device)
            tar_sample_device = tar_sample[k].to(device)
            src_mask_device = src_mask[k].float().to(device)

            # Predict masks.
            """
            GV_feature = torch.stack(
                [GV_features[k], GV_features[batchsize + k]], dim=0) 
            GV_features_src = GV_features[k].view(1, 1, -1)
            GV_features_src = GV_features_src.repeat(1,  GV_src_device.shape[0], 1)
            mask_input = torch.cat([GV_src_device.unsqueeze(0), GV_features_src], dim=2)
            print("mask_input:", mask_input.shape)
            predicted_mask = mask_network(mask_input).squeeze(0)
            print("predicted_mask:", predicted_mask.shape)
            predicted_mask = predicted_mask.softmax(dim=1)
            """

            # Deform.
            """
            sample_flow_features_k = sample_flow_features[k].permute(1, 0).contiguous()
            GV_flow_features = feature_propagator(GV_src_device.unsqueeze(
                0), src_sample_device[k].unsqueeze(0), None, sample_flow_features_k.unsqueeze(0))
            GV_flow_features = GV_flow_features.squeeze(0).permute(1, 0)    
            GV_flow_features = torch.cat([GV_src_device, GV_flow_features], dim=1)
            """
            n_src_pts = src_sample_device.shape[0]
            GV_src_features = GV_features[k].view(1, -1).repeat(n_src_pts, 1)
            transform_src_input = torch.cat(
                [src_sample_device, GV_src_features, torch.zeros(n_src_pts, 1).to(device)], dim=1)

            n_tar_pts = tar_sample_device.shape[0]
            GV_tar_features = GV_features[batchsize + k].view(1, -1).repeat(n_tar_pts, 1)
            transform_tar_input = torch.cat(
                [tar_sample_device, GV_tar_features, torch.ones(n_tar_pts, 1).to(device)], dim=1)
            
            transform_input = torch.cat([transform_src_input, transform_tar_input], dim=0).unsqueeze(0)
            
            transform_keyboard = keyboard_network(transform_input)
            rot_keyboard, trans_keyboard = extract_rot_trans(transform_keyboard, 0)
            keyboard = torch.matmul(rot_keyboard, GV_src_device.unsqueeze(2)) + trans_keyboard
            keyboard = src_mask_device[:, 0].unsqueeze(1) * keyboard.squeeze(2)
            
            transform_screen = screen_network(transform_input)
            rot_screen, trans_screen = extract_rot_trans(transform_screen, 0)
            screen = torch.matmul(rot_screen, GV_src_device.unsqueeze(2)) + trans_screen
            screen = src_mask_device[:, 1].unsqueeze(1) * screen.squeeze(2)
            
            GV_deformed = keyboard + screen
            
            """      
            # This is for predicting a 3D frame for each shape
            src_transform_input = GV_features[k].unsqueeze(0).repeat(GV_src_device.shape[0], 1)
            src_transform_input = torch.cat([GV_src_device, src_transform_input], dim=1)
            src_transform = screen_transform_network(src_transform_input.unsqueeze(0))
            rot_src, trans_src = extract_rot_trans(src_transform, 0)

            tar_transform_input = GV_features[batchsize + k].unsqueeze(0).repeat(GV_tar_device.shape[0], 1)
            tar_transform_input = torch.cat([GV_tar_device, tar_transform_input], dim=1)
            tar_transform = screen_transform_network(tar_transform_input.unsqueeze(0))
            rot_tar, trans_tar = extract_rot_trans(tar_transform, 0)

            screen_deformed = GV_src_device.unsqueeze(2)
            screen_deformed = torch.matmul(torch.transpose(rot_src, 1, 2), screen_deformed - trans_src)
            screen_deformed = torch.matmul(rot_tar, screen_deformed) + trans_tar
            screen_deformed = src_mask_device[:, 1].unsqueeze(1) * screen_deformed.squeeze(2)

            GV_deformed = screen_deformed + src_mask_device[:, 0].unsqueeze(1) * GV_src_device
            """
            # Compute losses
            loss_forward += graph_loss(
                GV_deformed, GE_src[k], GV_tar[k], GE_tar[k], src_param[k], tar_param[k], 0)
            loss_backward += reverse_loss(GV_deformed, GV_tar_origin, device)
            loss += loss_forward + loss_backward
            
            # Save results.
            if epoch % EPOCH_SNAPSHOT_INTERVAL == 0 or epoch == epochs - 1:
                print("Saving snapshot...")
                if epoch == epochs - 1:
                    output = output_prefix + "_final_"
                else:
                    output = output_prefix + "_snapshot_"
                output += str(epoch).zfill(4) + "_" + \
                    str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(2) + ".obj"
                with torch.no_grad():
                    save_snapshot_results(GV_deformed, GV_src[k], V_src[k], F_src[k],
                                          V_tar[k], F_tar[k], tar_param[k], output)    
        
        loss.backward()
        optimizer.step()
                
        print("Epoch: {} | Batch: {} | Shape_Pair: ({}, {}) | "
                "Loss_forward: {:.6f} | Loss_backward: {:.6f}".format(
                  epoch, batch_idx, src, tar, np.sqrt(
                      loss_forward.item() / GV_src_device.shape[0] / batchsize),
                  np.sqrt(loss_backward.item() /
                          GV_tar_origin.shape[0] / batchsize)))

        if epoch % EPOCH_SNAPSHOT_INTERVAL == 0 or epoch == epochs - 1:
            if epoch == epochs - 1:
                output = output_prefix + "_final_" + str(epoch).zfill(4) + ".ckpt"
            else:
                output = output_prefix + "_snapshot_" + str(epoch).zfill(4) + ".ckpt"
            torch.save({"screen_network": screen_network, 
                        "optim": optimizer}, output)
