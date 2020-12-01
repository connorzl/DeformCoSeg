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

from layers.neuralode_conditional_local import NeuralFlowDeformer
from layers.pointnet_local_features import PointNetSeg
from layers.pointnet_plus_correlate import PointNetCorrelate

from torch.utils.data import DataLoader
from util.load_data import compute_deformation_pairs, load_neural_deform_data, collate
from util.save_data import save_snapshot_results
from util.dataloader import SAPIENMesh, RandomPairSampler
import pyDeform
import numpy as np
import argparse
import random
from sklearn.manifold import TSNE
from types import SimpleNamespace

parser = argparse.ArgumentParser(description='Rigid Deformation.')
parser.add_argument('--input', default='')
parser.add_argument('--output_prefix', default='./cad-output')
parser.add_argument('--ckpt', default='')
parser.add_argument('--rigidity', default='0.1')
parser.add_argument('--device', default='cuda')
parser.add_argument('--batchsize', default=1)
args = parser.parse_args()

RANDOM_SEED = 1

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

output_prefix = args.output_prefix
rigidity = float(args.rigidity)
device = torch.device(args.device)
batchsize = int(args.batchsize)

train_dataset = SAPIENMesh(args.input)
train_sampler = RandomPairSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,
                          drop_last=True, sampler=train_sampler, collate_fn=collate)

# PointNet layer.
ckpt = torch.load(args.ckpt, map_location=device)
deformer = ckpt["deformer"].to(device)
deformer.eval()
pointnet_local_features = ckpt["pointnet_local_features"].to(device)
pointnet_local_features.eval()
pointnet_correlation = ckpt["pointnet_correlation"].to(device)
pointnet_correlation.eval()
feature_propagator = ckpt["feature_propagator"].to(device)
feature_propagator.eval()

# Prepare data for deformation
graph_loss = GraphLossLayerBatch(rigidity, device)
reverse_loss = ReverseLossLayer()

for batch_idx, data_tensors in enumerate(train_loader):    
    with torch.no_grad():
        loss_forward = 0
        loss_backward = 0
        loss_mask = 0        
        
        # Retrieve data for deformation
        src, tar, src_param, tar_param, src_data, tar_data, \
            src_sample, tar_sample, src_mask, tar_mask = data_tensors
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
            points_transformed = deformer.forward(GV_src_device, GV_flow_features)
            
            GV_deformed = points_transformed[-1]

            # Compute losses.
            loss_forward += graph_loss(
                GV_deformed, GE_src[k], GV_tar[k], GE_tar[k], src_param[k], tar_param[k], 0)
            loss_backward += reverse_loss(GV_deformed, GV_tar_origin, device)
        
            output = output_prefix + "_eval_" + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(2)
           
            for i in range(len(points_transformed)-1):
                save_snapshot_results(points_transformed[i], GV_src[k], V_src[k], F_src[k],
                                      V_tar[k], F_tar[k], tar_param[k], output + "_intermediate_" + str(i) + ".obj")
            save_snapshot_results(GV_deformed, GV_src[k], V_src[k], F_src[k],
                                  V_tar[k], F_tar[k], tar_param[k], output + ".obj")   

        print("Batch: {} | Shape_Pair: ({}, {}) | "
                "Loss_forward: {:.6f} | Loss_backward: {:.6f}".format(
                  batch_idx, src, tar, np.sqrt(
                      loss_forward.item() / GV_src_device.shape[0] / batchsize),
                  np.sqrt(loss_backward.item() /
                          GV_tar_origin.shape[0] / batchsize)))
