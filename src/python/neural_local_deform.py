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
from layers.pointnet_local_features import PointNetSeg#, PointNetCorrelate
#from layers.dgcnn import DGCNN
from layers.pointnet_plus_knn import PointNet2SemSegSSG
#from layers.pointnet_plus_correlate import PointNetCorrelate
#from layers.pointnet_plus_correlate_parts import PointNetCorrelateParts

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
#NUM_PARTS = 2

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

parameters = []

# Extract features for each shape.
pointnet_local_features = PointNetSeg(LATENT_SIZE)
parameters += list(pointnet_local_features.parameters())
pointnet_local_features = pointnet_local_features.to(device)

# Extract features that correlate two shapes.
#pointnet_correlation = DGCNN(input_features=3+LATENT_SIZE+1, output_features=LATENT_SIZE)
pointnet_correlation = PointNet2SemSegSSG({'feat_dim': LATENT_SIZE+1, 'output_dim': LATENT_SIZE})
#pointnet_correlation = PointNetCorrelate(input_features=LATENT_SIZE+1, output_features=LATENT_SIZE)
#pointnet_correlation = PointNetCorrelateParts(NUM_PARTS, LATENT_SIZE+1, output_latent_size=LATENT_SIZE)
parameters += list(pointnet_correlation.parameters())
pointnet_correlation = pointnet_correlation.to(device)

# Flow layer.
deformer = NeuralFlowDeformer(adjoint=False, dim=4, latent_size=LATENT_SIZE, device=device)
parameters += list(deformer.parameters())
deformer.to(device)
#deformers = []
#for i in range(NUM_PARTS):
#    deformer = NeuralFlowDeformer(adjoint=False, dim=4, latent_size=LATENT_SIZE, device=device)
#    parameters += list(deformer.parameters())
#    deformer.to(device)
#    deformers.append(deformer)

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
        
        # Retrieve data for deformation
        src, tar, src_param, tar_param, src_data, tar_data, \
            _, _, _, _ = data_tensors
        V_src, F_src, E_src, GV_src, GE_src = src_data
        V_tar, F_tar, E_tar, GV_tar, GE_tar = tar_data
       
        # Deform each (src, tar) pair.
        for k in range(batchsize):
            # Copies of normalized GV for deformation training.
            GV_tar_origin = GV_tar[k].clone()
            GV_src_device = GV_src[k].to(device)
            GV_tar_device = GV_tar[k].to(device)
            
            GV_src_features = pointnet_local_features(GV_src_device.unsqueeze(0))
            GV_tar_features = pointnet_local_features(GV_tar_device.unsqueeze(0))
            
            GV_src_correlation = torch.cat([GV_src_device.unsqueeze(0), GV_src_features, torch.zeros(
                1, GV_src_features.shape[1], 1).to(device)], dim=2)
            GV_tar_correlation = torch.cat([GV_tar_device.unsqueeze(0), GV_tar_features, torch.ones(
                1, GV_tar_features.shape[1], 1).to(device)], dim=2)
            GV_correlation_input = torch.cat([GV_src_correlation, GV_tar_correlation], dim=1)
            GV_flow_features = pointnet_correlation(GV_correlation_input).squeeze(0)
            
            # Deform
            #GV_flow_features = torch.split(GV_flow_features, LATENT_SIZE)
            #GV_deformed = torch.zeros(GV_src_device.shape).to(device)
            #for i in range(NUM_PARTS):
            #    GV_deformed += deformers[i].forward(GV_src_device, GV_flow_features[i])
                
            # Deform.
            GV_flow_src_features = GV_flow_features[:GV_src_device.shape[0]]
            GV_deformed = deformer.forward(GV_src_device, GV_flow_src_features)

            # Compute losses.
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

        print("Epoch: {}, Batch: {}, Shape_Pair: ({}, {}), "
              "Loss_forward: {: .6f}, Loss_backward: {: .6f}".format(
                  epoch, batch_idx, src, tar, np.sqrt(
                      loss_forward.item() / GV_src[0].shape[0] / batchsize),
                  np.sqrt(loss_backward.item() / GV_tar[0].shape[0] / batchsize)))

        if epoch % EPOCH_SNAPSHOT_INTERVAL == 0 or epoch == epochs - 1:
            if epoch == epochs - 1:
                output = output_prefix + "_final_" + str(epoch).zfill(4) + ".ckpt"
            else:
                output = output_prefix + "_snapshot_" + \
                    str(epoch).zfill(4) + ".ckpt"
            torch.save({"deformer": deformer, "pointnet_local_features": pointnet_local_features,
                        "pointnet_correlation": pointnet_correlation,  "optim": optimizer}, output)
