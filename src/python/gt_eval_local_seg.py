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
from pointnet2_ops.pointnet2_modules import PointnetFPModule
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
parser.add_argument('--ckpt', default='')
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

def compute_weighted_centroid(V, mask):
    sum_weights = torch.sum(mask)
    V_weighted = mask * V
    centroid_x = torch.sum(V_weighted[:, 0]) / sum_weights
    centroid_y = torch.sum(V_weighted[:, 1]) / sum_weights
    centroid_z = torch.sum(V_weighted[:, 2]) / sum_weights
    centroid = torch.tensor([centroid_x, centroid_y, centroid_z])
    return centroid

def compute_centered_vertices(V, centroid):
    return V - centroid.unsqueeze(0)

def compute_rigidity_loss(source, deformed, mask, name, part_rot, part_trans):
    """
    # Fit a rigid transform to the predicted deformation.
    source_centroid = compute_weighted_centroid(source, mask).to(device)
    deformed_centroid = compute_weighted_centroid(deformed, mask).to(device)
    X = compute_centered_vertices(source, source_centroid)
    Y = compute_centered_vertices(deformed, deformed_centroid)
    #X_save = X.detach().clone().cpu()
    #Y_save = Y.detach().clone().cpu()
    #np.savetxt(args.output_prefix + "_" + name + "_source_centered.txt", X_save)
    #np.savetxt(args.output_prefix + "_" + name + "_deformed_centered.txt", Y_save)
    
    W = torch.diag_embed(mask.squeeze(1))
    XT = torch.transpose(X, 0, 1) 
    #XTW = torch.transpose(torch.matmul(XT, W), 0, 1)
    #XTW_save = XTW.detach().clone().cpu()
    #np.savetxt(args.output_prefix + "_" + name + "_source_centered_masked.txt", XTW_save)
    #YT = torch.transpose(Y, 0, 1)
    #YTW = torch.transpose(torch.matmul(YT, W), 0, 1)
    #YTW_save = YTW.detach().clone().cpu()
    #np.savetxt(args.output_prefix + "_" + name + "_deformed_centered_masked.txt", YTW_save)
    cov_mat = torch.matmul(torch.matmul(XT, W), Y)
    (U, S, V) = torch.svd(cov_mat)
    UT = torch.transpose(U, 0, 1)
    det_VUT = torch.det(torch.matmul(V, UT))
    S_rot = torch.tensor([1.0, 1.0, det_VUT]).to(device)
    S_rot = torch.diag_embed(S_rot)
    rot = torch.matmul(V, torch.matmul(S_rot, UT))
    trans = deformed_centroid.unsqueeze(1) - torch.matmul(rot, source_centroid.unsqueeze(1))

    deformed_rigid = torch.matmul(rot.unsqueeze(0), source.unsqueeze(2)) + trans.unsqueeze(0)
    deformed_rigid = deformed_rigid.squeeze(2)
    """
    deformed_rigid = torch.matmul(part_rot, source.unsqueeze(2)) + part_trans
    deformed_rigid = deformed_rigid.squeeze(2)
    deformed_rigid_save = deformed_rigid.detach().clone().cpu()
    np.savetxt(args.output_prefix + "_" + name + "_rigid_transformed.txt", deformed_rigid_save)
    deformed_save = deformed.detach().clone().cpu()
    np.savetxt(args.output_prefix + "_" + name + "_deformed.txt", deformed_save)
    dist = torch.norm(mask * deformed_rigid - mask * deformed)
    print("dist:", dist)
    return dist

ckpt = torch.load(args.ckpt, map_location=device)
#deformer_0 = ckpt["deformer_0"].to(device)
#deformer_0.eval()
#deformer_1 = ckpt["deformer_1"].to(device)
#deformer_1.eval()

pointnet_local_features = ckpt["pointnet_local_features"].to(device)
pointnet_local_features.eval()
#pointnet_correlation = ckpt["pointnet_correlation"].to(device)
#pointnet_correlation.eval()
#feature_propagator = ckpt["feature_propagator"].to(device)
#feature_propagator.eval()
masknet = ckpt["masknet"].to(device)
masknet.eval()

# Losses.
graph_loss = GraphLossLayerBatch(rigidity, device)
reverse_loss = ReverseLossLayer()

# Training loop.
for batch_idx, data_tensors in enumerate(train_loader):    
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
    sample_flow_features = pointnet_correlation(correlation_sample_input)
    sample_flow_features = sample_flow_features[:, :num_samples, :]
    """
    # Deform each (src, tar) pair.
    for k in range(batchsize):
        # Copies of normalized GV for deformation training.
        GV_tar_origin = GV_tar[k].clone()
        GV_src_device = GV_src[k].to(device)
        
        # Predict mask.
        #src_sample_features_k = src_sample_features[k].permute(1, 0).contiguous()
        #GV_src_features = feature_propagator(GV_src_device.unsqueeze(
        #    0), src_sample_device[k], None, src_sample_features_k.unsqueeze(0))
        GV_src_features = pointnet_local_features(GV_src_device.unsqueeze(0))
        predicted_masks = masknet(GV_src_features).squeeze(0)
        #predicted_masks = src_mask[k].to(device).float()
        """
        # Propogate features from uniform samples to GV_src_device
        sample_flow_features_k = sample_flow_features[k].permute(1, 0).contiguous()
        GV_flow_features = feature_propagator(GV_src_device.unsqueeze(
            0), src_sample_device[k].unsqueeze(0), None, sample_flow_features_k.unsqueeze(0))
        GV_flow_features = GV_flow_features.squeeze(0).permute(1, 0)
        """
        # Ground truth Rigid transform
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
        
        # Update GT rigid transform for normalization.
        normalize_params = pyDeform.GetNormalizeParams(src_param[k])
        scale = normalize_params[0]
        trans = torch.from_numpy(np.asarray(normalize_params[1:])).float().to(device)
        trans = torch.unsqueeze(trans, dim=0)
        trans = torch.unsqueeze(trans, dim=2)
        part_origin = (part_origin - trans) / scale 

        keyboard_rot = part_rot[0, :].unsqueeze(0)
        keyboard_trans = -torch.matmul(part_rot[0, :], part_origin[0, :]) + part_origin[0, :]
        screen_rot = part_rot[1, :].unsqueeze(0) 
        screen_trans = -torch.matmul(part_rot[1, :], part_origin[1, :]) + part_origin[1, :]
        
        # Deform.
        part_0 = torch.matmul(keyboard_rot, GV_src_device.unsqueeze(2)) + keyboard_trans
        part_0 = part_0.squeeze(2)
        part_1 = torch.matmul(screen_rot, GV_src_device.unsqueeze(2)) + screen_trans
        part_1 = part_1.squeeze(2)

        unmasked_flow_0 = part_0 - GV_src_device
        unmasked_flow_1 = part_1 - GV_src_device
        masked_flow_0 = predicted_masks[:, 0].unsqueeze(1) * unmasked_flow_0
        masked_flow_1 = predicted_masks[:, 1].unsqueeze(1) * unmasked_flow_1

        np.savetxt(output_prefix + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(
            2) + "_flow_0_unmasked.txt", unmasked_flow_0.detach().clone().cpu())
        np.savetxt(output_prefix + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(
            2) + "_flow_1_unmasked.txt", unmasked_flow_1.detach().clone().cpu())
        np.savetxt(output_prefix + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(
            2) + "_flow_0_masked.txt", masked_flow_0.detach().clone().cpu())
        np.savetxt(output_prefix + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(
            2) + "_flow_1_masked.txt", masked_flow_1.detach().clone().cpu())
        np.savetxt(output_prefix + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(
            2) + "_predicted_mask.txt", predicted_masks[:, 0].detach().clone().cpu())
        GV_deformed = predicted_masks[:, 0].unsqueeze(1) * part_0 \
            + predicted_masks[:, 1].unsqueeze(1) * part_1

        # Perform segmentation and compute rigid transform loss
        loss_rigidity += compute_rigidity_loss(
            GV_src_device, GV_deformed, predicted_masks[:, 0].unsqueeze(1), "keyboard" + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(2), keyboard_rot, keyboard_trans)
        loss_rigidity += compute_rigidity_loss(
            GV_src_device, GV_deformed, predicted_masks[:, 1].unsqueeze(1), "screen" + str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(2), screen_rot, screen_trans)

        ones = torch.ones(predicted_masks.shape[0], 1).to(device)
        mask_norm = torch.norm(predicted_masks, dim=1)
        loss_mask += torch.norm(ones - mask_norm)            
        
        # Compute losses.
        loss_forward += graph_loss(
            GV_deformed, GE_src[k], GV_tar[k], GE_tar[k], src_param[k], tar_param[k], 0)
        loss_backward += reverse_loss(GV_deformed, GV_tar_origin, device)
        loss += loss_forward + loss_backward + loss_rigidity
       
        # Save results.
        print("Saving snapshot...")
        output = output_prefix + "_eval_" + \
            str(src[k]).zfill(2) + "_" + str(tar[k]).zfill(2) + ".obj"
        with torch.no_grad():
            save_snapshot_results(GV_deformed, GV_src[k], V_src[k], F_src[k],
                                  V_tar[k], F_tar[k], tar_param[k], output)    

    print("Batch: {}, Shape_Pair: ({}, {}), "
          "Loss_forward: {: .6f}, Loss_backward: {: .6f}, "
          "Loss_rigidity: {: .6f}, Loss_mask: {: .6f}".format(
              batch_idx, src, tar, np.sqrt(
                  loss_forward.item() / GV_src[0].shape[0] / batchsize),
              np.sqrt(loss_backward.item() /
                      GV_tar[0].shape[0] / batchsize),
              np.sqrt(loss_rigidity.item() /
                      GV_src[0].shape[0] / batchsize),
              np.sqrt(loss_mask.item() / GV_src[0].shape[0] / batchsize)))
