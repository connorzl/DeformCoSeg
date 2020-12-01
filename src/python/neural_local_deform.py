import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))
import torch.optim as optim
import torch
from layers.graph_loss_layer import GraphLossLayerBatch, IntermediateLossLayer
from layers.reverse_loss_layer import ReverseLossLayer

from layers.neuralode_conditional_local import NeuralFlowDeformer
from layers.pointnet_local_features import PointNetSeg
from layers.dgcnn import DGCNN
#from layers.pointnet_plus_knn import PointNet2SemSegSSG
#from layers.pointnet_plus_correlate import PointNetCorrelate
from pointnet2_ops.pointnet2_modules import PointnetFPModule

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.load_data import collate
from util.save_data import save_snapshot_results
from util.dataloader import SAPIENMesh, RandomPairSampler
import pyDeform
import numpy as np
import argparse
from types import SimpleNamespace
import time

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

output_prefix = args.output_prefix
rigidity = float(args.rigidity)
device = torch.device(args.device)
batchsize = int(args.batchsize)
epochs = int(args.epochs)

# For reproducability.
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(args.output_prefix), "tensorboard"))

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
pointnet_correlation = DGCNN(input_features=3+LATENT_SIZE+1, output_features=LATENT_SIZE)
#pointnet_correlation = PointNet2SemSegSSG({'feat_dim': LATENT_SIZE+1, 'output_dim': LATENT_SIZE})
#pointnet_correlation = PointNetCorrelate(input_features=LATENT_SIZE+1, output_features=LATENT_SIZE)
parameters += list(pointnet_correlation.parameters())
pointnet_correlation = pointnet_correlation.to(device)

# Flow layer.
deformer = NeuralFlowDeformer(adjoint=False, dim=4, latent_size=LATENT_SIZE, device=device)
parameters += list(deformer.parameters())
deformer.to(device)

feature_propagator = PointnetFPModule(mlp=[LATENT_SIZE, LATENT_SIZE, LATENT_SIZE], bn=False)
parameters += list(feature_propagator.parameters())
feature_propagator.to(device)

# Losses.
graph_loss = GraphLossLayerBatch(rigidity, device)
reverse_loss = ReverseLossLayer()

intermediate_loss = IntermediateLossLayer(rigidity)

optimizer = optim.Adam(parameters, lr=1e-3)

# Training loop.
global_step = np.zeros(1, dtype=np.uint32)
for epoch in range(epochs):
    for batch_idx, data_tensors in enumerate(train_loader):    
        optimizer.zero_grad()
        loss = 0
        loss_forward = 0
        loss_backward = 0
        loss_intermediate = 0
        
        # Retrieve data for deformation
        src, tar, src_param, tar_param, src_data, tar_data, \
            src_sample, tar_sample, _, _ = data_tensors
        V_src, F_src, E_src, GV_src, GE_src = src_data
        V_tar, F_tar, E_tar, GV_tar, GE_tar = tar_data
            
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
        sample_flow_features = pointnet_correlation(correlation_sample_input)
        sample_flow_features = sample_flow_features[:, :num_samples, :]
       
        # Deform each (src, tar) pair.
        for k in range(batchsize):
            # Copies of normalized GV for deformation training.
            GV_tar_origin = GV_tar[k].clone()
            GV_src_device = GV_src[k].to(device)
            
            # Propogate features from uniform samples to GV_src_device
            sample_flow_features_k = sample_flow_features[k].permute(1, 0).contiguous()
            GV_flow_features = feature_propagator(GV_src_device.unsqueeze(
                0), src_sample_device[k].unsqueeze(0), None, sample_flow_features_k.unsqueeze(0))
            GV_flow_features = GV_flow_features.squeeze(0).permute(1, 0)
            
            # Deform.
            deformed = deformer.forward(GV_src_device, GV_flow_features)
            GV_deformed = deformed[-1]

            # Compute losses.
            loss_forward += graph_loss(
                GV_deformed, GE_src[k], GV_tar[k], GE_tar[k], src_param[k], tar_param[k], 0)
            loss_backward += reverse_loss(GV_deformed, GV_tar_origin, device) 
            #for i in range(1, len(deformed)-1):
            #    loss_intermediate += intermediate_loss(deformed[i], GE_src[k], src_param[k])
            loss += loss_forward + loss_backward #+ loss_intermediate
           
            # Save results.
            if (epoch % EPOCH_SNAPSHOT_INTERVAL == 0 or epoch == epochs - 1) and k < 5:
                print("Saving snapshot...")
                if epoch == epochs - 1:
                    output = output_prefix + "_final_"
                else:
                    output = output_prefix + "_snapshot_"
                output += str(epoch).zfill(4) + "_" + \
                    str(src[k]).zfill(4) + "_" + str(tar[k]).zfill(4) + ".obj"
                with torch.no_grad():
                    save_snapshot_results(GV_deformed, GV_src[k], V_src[k], F_src[k],
                                          V_tar[k], F_tar[k], tar_param[k], output)    

        # Write to Tensorboard.
        writer.add_scalar("train/loss_sum", loss.item(),
                          global_step=int(global_step),)
        writer.add_scalar("train/loss_avg",
                          np.sqrt(loss.item() / batchsize),
                          global_step=int(global_step),)
        deform_abs = torch.mean(torch.norm(GV_deformed - GV_src_device, dim=-1))
        writer.add_scalar("train/def_mean", deform_abs.item(),
                          global_step=int(global_step),)

        # Backprop.
        start = time.time()
        loss.backward()
        print("time:", time.time() - start)
        global_step += 1
        optimizer.step()

        print("Epoch: {}, Batch: {}, Shape_Pair: ({}, {}), "
              "Loss_forward: {:.6f}, Loss_backward: {:.6f}, Loss_intermediate: {:.6f}".format(
                  epoch, batch_idx, src, tar, np.sqrt(
                      loss_forward.item() / GV_src[0].shape[0] / batchsize),
                  np.sqrt(loss_backward.item() /
                          GV_tar[0].shape[0] / batchsize),
                  0))
                  #np.sqrt(loss_intermediate.item() / GV_src[0].shape[0] / batchsize)))

        if epoch % EPOCH_SNAPSHOT_INTERVAL == 0 or epoch == epochs - 1:
            if epoch == epochs - 1:
                output = output_prefix + "_final_" + str(epoch).zfill(4) + ".ckpt"
            else:
                output = output_prefix + "_snapshot_" + \
                    str(epoch).zfill(4) + ".ckpt"
            torch.save({"deformer": deformer,
                        "pointnet_local_features": pointnet_local_features,
                        "pointnet_correlation": pointnet_correlation,
                        "feature_propagator": feature_propagator,
                        "optim": optimizer}, output)
