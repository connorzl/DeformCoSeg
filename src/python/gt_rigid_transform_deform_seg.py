import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))
import torch.optim as optim
import torch
from layers.graph_loss_layer import GraphLossLayerBatch
from layers.reverse_loss_layer import ReverseLossLayer

from layers.neuralode_conditional_local import NeuralFlowDeformer
from layers.pointnet_local_features import PointNetSeg
from layers.pointnet_plus_correlate import PointNetCorrelate
from layers.mask_mlp import MaskNet

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
parser.add_argument('--batchsize', default=1)
parser.add_argument('--epochs', default=1000)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

EPOCH_SNAPSHOT_INTERVAL = 25
RANDOM_SEED = 1
LATENT_SIZE = 32
NUM_PARTS = 2

output_prefix = args.output_prefix
rigidity = float(args.rigidity)
device = torch.device(args.device)
batchsize = int(args.batchsize)
epochs = int(args.epochs)

# For reproducability.
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Dataloader.
train_dataset = SAPIENMesh(args.input)
train_sampler = RandomPairSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,
                          drop_last=True, sampler=train_sampler, collate_fn=collate)

def compute_rigidity_loss(source, deformed, mask, name, part_rot, part_trans):
    deformed_rigid = torch.matmul(part_rot, source.unsqueeze(2)) + part_trans
    deformed_rigid = deformed_rigid.squeeze(2)
    dist = torch.norm(mask * deformed_rigid - mask * deformed)
    return dist

parameters = []

# Extract features for each shape.
pointnet_local_features = PointNetSeg(LATENT_SIZE)
parameters += list(pointnet_local_features.parameters())
pointnet_local_features = pointnet_local_features.to(device)

# Mask network.
masknet = MaskNet(LATENT_SIZE, NUM_PARTS)
parameters += list(masknet.parameters())
masknet.to(device)

deformers = []
for i in range(NUM_PARTS):
    deformer = NeuralFlowDeformer(
        adjoint=False, latent_size=LATENT_SIZE + 1, device=device).to(device)
    parameters += list(deformer.parameters())
    deformers.append(deformer)

# Losses.
graph_loss = GraphLossLayerBatch(rigidity, device)
reverse_loss = ReverseLossLayer()
optimizer = optim.Adam(parameters, lr=1e-3)

# Training loop.
for epoch in range(epochs):
    for batch_idx, data_tensors in enumerate(train_loader):    
        optimizer.zero_grad()
        loss = 0
        loss_forward = 0
        loss_backward = 0
        loss_mask = 0
        loss_rigidity = 0
        
        # Retrieve data for deformation
        src, tar, src_param, tar_param, src_data, tar_data, \
            src_sample, tar_sample, src_mask, tar_mask, src_trans, tar_trans = data_tensors
        V_src, F_src, E_src, GV_src, GE_src = src_data
        V_tar, F_tar, E_tar, GV_tar, GE_tar = tar_data
            
        # Deform each (src, tar) pair.
        for k in range(batchsize):
            # Copies of normalized GV for deformation training.
            GV_tar_origin = GV_tar[k].clone()
            GV_src_device = GV_src[k].to(device)
           
            GV_src_features = pointnet_local_features(GV_src_device.unsqueeze(0))
            predicted_masks = masknet(GV_src_features).squeeze(0) 
            #predicted_masks = src_mask[k].to(device).float()
           
            # Rigid transform
            rigid_transform = tar_trans[k]
            part_rot = np.zeros((2, 3, 3))
            part_origin = np.zeros((2, 3, 1))
            for part in range(0, 2):
                for i in range(0, 3):
                    for j in range(0, 3):
                        part_rot[part][i][j] = rigid_transform[part][3*i + j]
                for i in range(0, 3):
                    part_origin[part][i] = rigid_transform[part][9 + i] 
            part_rot = torch.from_numpy(part_rot).float().to(device)
            part_origin = torch.from_numpy(part_origin).float().to(device)
           
            normalize_params = pyDeform.GetNormalizeParams(src_param[k])
            scale = normalize_params[0]
            trans = torch.from_numpy(np.asarray(normalize_params[1:])).float().to(device)
            trans = torch.unsqueeze(trans, dim=0)
            trans = torch.unsqueeze(trans, dim=2)
            part_origin = (part_origin - trans) / scale 

            # Ground truth rigid transforms for each part.
            keyboard_rot = part_rot[0, :].unsqueeze(0)
            keyboard_trans = -torch.matmul(part_rot[0, :], part_origin[0, :]) + part_origin[0, :]
            screen_rot = part_rot[1, :].unsqueeze(0) 
            screen_trans = -torch.matmul(part_rot[1, :], part_origin[1, :]) + part_origin[1, :]
          
            part_0 = torch.matmul(keyboard_rot, GV_src_device.unsqueeze(2)) + keyboard_trans
            part_0 = predicted_masks[:, 0].unsqueeze(1) * part_0.squeeze(2)
            part_1 = torch.matmul(screen_rot, GV_src_device.unsqueeze(2)) + screen_trans
            part_1 = predicted_masks[:, 1].unsqueeze(1) * part_1.squeeze(2)
            GV_deformed = part_0 + part_1

            # Perform segmentation and compute rigid transform loss
            """
            loss_rigidity += compute_rigidity_loss(
                GV_src_device, GV_deformed, predicted_masks[:, 0].unsqueeze(1), "keyboard" + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(2), keyboard_rot, keyboard_trans)
            loss_rigidity += compute_rigidity_loss(
                GV_src_device, GV_deformed, predicted_masks[:, 1].unsqueeze(1), "screen" + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(2), screen_rot, screen_trans)
            """

            ones = torch.ones(predicted_masks.shape[0], 1).to(device)
            mask_norm = torch.norm(predicted_masks, dim=1)
            loss_mask += torch.norm(ones - mask_norm)            
            
            # Compute losses.
            loss_forward += graph_loss(
                GV_deformed, GE_src[k], GV_tar[k], GE_tar[k], src_param[k], tar_param[k], 0)
            loss_backward += reverse_loss(GV_deformed, GV_tar_origin, device)
            loss += loss_forward + loss_backward + 0.1 * loss_rigidity
            
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

        print("Epoch: {}, Batch: {}, Shape_Pair: ({}, {}), "
              "Loss_forward: {: .6f}, Loss_backward: {: .6f}, "
              "Loss_rigidity: {: .6f}, Loss_mask: {: .6f}".format(
                  epoch, batch_idx, src, tar, np.sqrt(
                      loss_forward.item() / GV_src[0].shape[0] / batchsize),
                  np.sqrt(loss_backward.item() /
                          GV_tar[0].shape[0] / batchsize),
                  np.sqrt(loss_rigidity.item() /
                          GV_src[0].shape[0] / batchsize),
                  np.sqrt(loss_mask.item() / GV_src[0].shape[0] / batchsize)))

        if epoch % EPOCH_SNAPSHOT_INTERVAL == 0 or epoch == epochs - 1:
            if epoch == epochs - 1:
                output = output_prefix + "_final_" + str(epoch).zfill(4) + ".ckpt"
            else:
                output = output_prefix + "_snapshot_" + \
                    str(epoch).zfill(4) + ".ckpt"
            torch.save({"pointnet_local_features": pointnet_local_features,
                        "masknet": masknet,
                        "optim": optimizer}, output)
