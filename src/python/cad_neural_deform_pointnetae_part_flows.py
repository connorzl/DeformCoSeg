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
#from layers.neuralode_conditional_mask import NeuralFlowDeformer
from layers.neuralode_conditional_mask_regularization import NeuralFlowDeformer
from layers.pointnet_ae import Network
from layers.pointnet_plus_mask import PointNet2
from torch.utils.data import DataLoader
from util.load_data import compute_deformation_pairs, load_neural_deform_data, collate
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

output_prefix = args.output_prefix
rigidity = float(args.rigidity)
device = torch.device(args.device)
batchsize = int(args.batchsize)
epochs = int(args.epochs)

train_dataset = SAPIENMesh(args.input)
train_sampler = RandomPairSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,
                          drop_last=True, sampler=train_sampler, collate_fn=collate)

# PointNet layer.
pointnet_conf = SimpleNamespace(
    num_point=2048, decoder_type='fc', loss_type='emd')
pointnet = Network(pointnet_conf, 1024)
pointnet.load_state_dict(torch.load(
    args.pointnet_ckpt, map_location=device))
pointnet.eval()
for param in pointnet.parameters():
    param.requires_grad = False
pointnet = pointnet.to(device)

# Mask Network.
NUM_PARTS = 2
mask_network = PointNet2(NUM_PARTS).to(device)

# Flow layer.
deformer = NeuralFlowDeformer(
    adjoint=False, dim=3, latent_size=1024, num_parts=NUM_PARTS-1, device=device)
deformer.to(device)
optimizer = optim.Adam(deformer.parameters(), lr=1e-3)

graph_loss = GraphLossLayerBatch(rigidity, device)
reverse_loss = ReverseLossLayer()

for epoch in range(epochs):
    for batch_idx, data_tensors in enumerate(train_loader):    
        optimizer.zero_grad()
        loss = 0
        loss_mask = 0
        loss_forward = 0
        loss_backward = 0
        
        # Retrieve data for deformation
        src, tar, src_param, tar_param, src_data, tar_data, \
            V_src_sample, V_tar_sample, src_mask, tar_mask = data_tensors
        V_src, F_src, E_src, GV_src, GE_src = src_data
        V_tar, F_tar, E_tar, GV_tar, GE_tar = tar_data
        
        # Prepare PointNet input.
        GV_pointnet_inputs = torch.cat([V_src_sample, V_tar_sample], dim=0).to(device)
        _, GV_features = pointnet(GV_pointnet_inputs)
        GV_features = GV_features.detach()
       
        # Deform each (src, tar) pair.
        for k in range(batchsize):
            # Copies of normalized GV for deformation training.
            GV_tar_origin = GV_tar[k].clone()
            GV_src_device = GV_src[k].to(device)

            GV_feature = torch.stack(
                [GV_features[k], GV_features[batchsize + k]], dim=0) 

            screen_mask = src_mask[k][:, 1].to(device)

            # Predict masks.
            """
            GV_features_src = GV_features[k].view(1, 1, -1)
            GV_features_src = GV_features_src.repeat(1,  GV_src_device.shape[0], 1)
            mask_input = torch.cat([GV_src_device.unsqueeze(0), GV_features_src], dim=2)
            predicted_mask = mask_network(mask_input).squeeze(0)
            predicted_mask = predicted_mask.softmax(dim=1)
            """

            # Deform.
            # We only care about 1 of the masks, since the other will be multiplied with 0 flow.
            #GV_deformed = deformer.forward(GV_src_device, GV_feature, predicted_mask[:, 0])
            GV_deformed = deformer.forward(GV_src_device, GV_feature, screen_mask)

            # Compute losses
            """
            ones = torch.ones(GV_src_device.shape[0], 1).to(device)
            mask_norm = torch.norm(predicted_mask, dim=1)
            loss_mask += torch.norm(ones - mask_norm)
            """
            loss_forward += graph_loss(
                GV_deformed, GE_src[k], GV_tar[k], GE_tar[k], src_param[k], tar_param[k], 0)
            loss_backward += reverse_loss(GV_deformed, GV_tar_origin, device)
            loss += loss_forward + loss_backward + loss_mask
           
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
                    save_snapshot_results(GV_deformed, V_src[k], F_src[k],
                                          V_tar[k], F_tar[k], tar_param[k], output)    
        
        loss.backward()
        optimizer.step()
                
        print("Epoch: {} | Batch: {} | Shape_Pair: ({}, {}) | "
              "Loss_forward: {:.6f} | Loss_backward: {:.6f} | Loss_mask: {:.6f}".format(
                  epoch, batch_idx, src, tar, np.sqrt(
                      loss_forward.item() / GV_src_device.shape[0] / batchsize),
                  np.sqrt(loss_backward.item() /
                          GV_tar_origin.shape[0] / batchsize),
                  np.sqrt(0 / GV_src_device.shape[0] / batchsize)))

        if epoch % EPOCH_SNAPSHOT_INTERVAL == 0 or epoch == epochs - 1:
            if epoch == epochs - 1:
                output = output_prefix + "_final_" + str(epoch).zfill(4) + ".ckpt"
            else:
                output = output_prefix + "_snapshot_" + str(epoch).zfill(4) + ".ckpt"
            torch.save({"deformer": deformer, "mask_network": mask_network,
                        "optim": optimizer}, output)
