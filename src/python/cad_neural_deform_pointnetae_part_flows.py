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

from layers.neuralode_conditional import NeuralFlowDeformer
#from layers.neuralode_conditional_mask import NeuralFlowDeformer
from layers.pointnet_ae import Network
#from layers.pointnet_plus_mask import PointNet2
from layers.pointnet_local_features import PointNetMask
from util.cd.chamfer import chamfer_distance

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


def compute_centroid(V):
    xs = V[:, 0]
    ys = V[:, 1]
    zs = V[:, 2]
    return torch.tensor([torch.mean(xs), torch.mean(ys), torch.mean(zs)])

def compute_centered_vertices(V, centroid):
    return V - centroid.unsqueeze(0)

def compute_rigidity_loss(source, deformed):
    # Fit a rigid transform to the predicted deformation.
    source_centroid = compute_centroid(source).to(device)
    deformed_centroid = compute_centroid(deformed).to(device)
    X = torch.transpose(compute_centered_vertices(source, source_centroid), 0, 1) 
    YT = compute_centered_vertices(deformed, deformed_centroid)
    cov_mat = torch.matmul(X, YT)

    (U, S, V) = torch.svd(cov_mat)
    UT = torch.transpose(U, 0, 1)            
    det_VUT = torch.det(torch.matmul(V, UT))
    S_rot = torch.tensor([1.0, 1.0, det_VUT]).to(device)
    S_rot = torch.diag_embed(S_rot)

    rot = torch.matmul(V, torch.matmul(S_rot, UT))
    trans = deformed_centroid.unsqueeze(1) - torch.matmul(rot, source_centroid.unsqueeze(1))

    deformed_rigid = torch.matmul(rot.unsqueeze(0), source.unsqueeze(2)) + trans.unsqueeze(0)
    deformed_rigid = deformed_rigid.squeeze(2)
    dist_1, dist_2 = chamfer_distance(deformed.unsqueeze(0), deformed_rigid.unsqueeze(0), transpose=False)
    chamfer_loss = torch.sum(dist_1, dim=1) + torch.sum(dist_2, dim=1)
    return chamfer_loss


parameters = []

# PointNet layer.
pointnet_conf = SimpleNamespace(
    num_point=2048, decoder_type='fc', loss_type='emd')
pointnet = Network(pointnet_conf, 1024)
if args.pointnet_ckpt != "":
    print("Loading pretrained PointNetAE!")
    pointnet.load_state_dict(torch.load(
        args.pointnet_ckpt, map_location=device))
    pointnet.eval()
else:
    print("Training joint PointNetAE!")
    parameters += list(pointnet.parameters())
pointnet = pointnet.to(device)

# Mask Network.
NUM_PARTS = 2
#mask_network = PointNet2(NUM_PARTS)
mask_network = PointNetMask(3+1024, NUM_PARTS)
parameters += list(mask_network.parameters()) 
mask_network = mask_network.to(device)

# Flow layer.
#deformer = NeuralFlowDeformer(
#    adjoint=False, dim=3, latent_size=1024, num_parts=NUM_PARTS, device=device)
deformer = NeuralFlowDeformer(
    adjoint=False, dim=3, latent_size=1024, device=device)
parameters += list(deformer.parameters()) 
deformer = deformer.to(device)

graph_loss = GraphLossLayerBatch(rigidity, device)
reverse_loss = ReverseLossLayer()

optimizer = optim.Adam(parameters, lr=1e-3)

for epoch in range(epochs):
    for batch_idx, data_tensors in enumerate(train_loader):    
        optimizer.zero_grad()
        loss = 0
        loss_forward = 0
        loss_backward = 0
        loss_rigidity = 0

        # Retrieve data for deformation
        src, tar, src_param, tar_param, src_data, tar_data, \
            V_src_sample, V_tar_sample, src_mask, tar_mask = data_tensors
        V_src, F_src, E_src, GV_src, GE_src = src_data
        V_tar, F_tar, E_tar, GV_tar, GE_tar = tar_data
        
        # Prepare PointNet input.
        GV_pointnet_inputs = torch.cat([V_src_sample, V_tar_sample], dim=0).to(device)
        _, GV_features = pointnet(GV_pointnet_inputs)
       
        # Deform each (src, tar) pair.
        for k in range(batchsize):
            # Copies of normalized GV for deformation training.
            GV_tar_origin = GV_tar[k].clone()
            GV_src_device = GV_src[k].to(device)
            GV_feature = torch.stack(
                [GV_features[k], GV_features[batchsize + k]], dim=0) 

            # Predict masks.
            GV_features_src = GV_features[k].view(1, 1, -1)
            GV_features_src = GV_features_src.repeat(1,  GV_src_device.shape[0], 1)
            mask_input = torch.cat([GV_src_device.unsqueeze(0), GV_features_src], dim=2)
            predicted_mask = mask_network(mask_input).squeeze(0)

            # Deform.
            GV_deformed = deformer.forward(GV_src_device, GV_feature)
            #GV_deformed = deformer.forward(GV_src_device, GV_feature, predicted_mask)
            #GV_deformed = deformer.forward(GV_src_device, GV_feature, src_mask[k].to(device))
          
            # Encourage overall flow to follow a rigid transform, and then apply mask.
            loss_rigidity += compute_rigidity_loss(GV_src_device, GV_deformed)
            GV_deformed = GV_src_device + predicted_mask[:, 0].unsqueeze(1) * (GV_deformed - GV_src_device) 

            # Compute losses
            loss_forward += graph_loss(
                GV_deformed, GE_src[k], GV_tar[k], GE_tar[k], src_param[k], tar_param[k], 0)
            loss_backward += reverse_loss(GV_deformed, GV_tar_origin, device)
            loss += loss_forward + loss_backward + loss_rigidity

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
                "Loss_forward: {:.6f} | Loss_backward: {:.6f} | Loss_rigidity: {:.6f}".format(
                  epoch, batch_idx, src, tar, np.sqrt(
                      loss_forward.item() / GV_src_device.shape[0] / batchsize),
                  np.sqrt(loss_backward.item() /
                          GV_tar_origin.shape[0] / batchsize),
                  np.sqrt(loss_rigidity.item() / GV_src_device.shape[0] / batchsize)))

        if epoch % EPOCH_SNAPSHOT_INTERVAL == 0 or epoch == epochs - 1:
            if epoch == epochs - 1:
                output = output_prefix + "_final_" + str(epoch).zfill(4) + ".ckpt"
            else:
                output = output_prefix + "_snapshot_" + str(epoch).zfill(4) + ".ckpt"
            torch.save({"deformer": deformer, "mask_network": mask_network,
                        "optim": optimizer}, output)
