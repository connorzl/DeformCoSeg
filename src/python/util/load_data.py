import os
import sys
import trimesh
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from samplers import sample_faces, fps
import numpy as np


def compute_deformation_pairs(all_pairs, n):
    pairs = []
    if all_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
                #pairs.append((j, i))
    else:
        for i in range(1, n):
            pairs.append((0, i))
    return pairs


def load_mesh(mesh_path, intermediate=10000, final=2048):
    mesh = trimesh.load(mesh_path, process=False)
    V = mesh.vertices.astype(np.float32)
    F = mesh.faces.astype(np.int32)
    E = mesh.edges.astype(np.int32)

    # Surface vertices for PointNet input.
    V_sample, _, _ = sample_faces(V, F, n_samples=intermediate)
    V_sample, _ = fps(V_sample, final)
    V_sample = V_sample.astype(np.float32)
    
    return V, F, E, V_sample


def load_neural_deform_data(mesh_paths, device, intermediate=10000, final=2048):
    # Load meshes.
    V_all = []
    V_surf_all = []
    F_all = []
    E_all = []
    GV_all = []
    GE_all = []
    GV_origin_all = []
    GV_device_all = []
    for mesh_path in mesh_paths:
        V, F, E, V_surf = load_mesh(mesh_path)
        V = torch.from_numpy(V)
        F = torch.from_numpy(F)
        E = torch.from_numpy(E)
        V_surf = torch.from_numpy(V_surf)

        V_all.append(V)
        F_all.append(F)
        E_all.append(E)
        V_surf_all.append(V_surf)
        
        # Graph vertices and edges for deformation.
        GV_all.append(V_all[-1].clone())
        GE_all.append(E_all[-1].clone())
        # Graph vertices origins for computing losses.
        GV_origin_all.append(GV_all[-1].clone())
        # Graph vertices on device for deformation.
        GV_device_all.append(GV_all[-1].to(device))

    return (V_all, F_all, E_all, V_surf_all), (GV_all, GE_all, GV_origin_all, GV_device_all)


def load_segmentation(mesh_path, i, V):
    files_dir = os.path.dirname(mesh_path)
    segmentation_file = "segmentation_" + str(i).zfill(2) + ".txt"
    segmentation_file = os.path.join(files_dir, segmentation_file)
    
    # Count number of labels, assume labels start from 0.
    labels_set = set()
    with open(segmentation_file, "r") as f:
        lines = [line.rstrip("\n").split(",") for line in f]
        for line in lines:
            label = int(line[1])
            labels_set.add(label)

    # Maps each part to a list of its vertices
    V_parts = [[] for _ in labels_set]

    with open(segmentation_file, "r") as f:
        lines = [line.rstrip("\n").split(",") for line in f]
        labels = np.zeros((V.shape[0], 1))
        for line in lines:
            index = int(line[0]) - 1
            label = int(line[1])
            V_parts[label].append(V[index])
    
    for i in range(len(V_parts)):
        V_parts[i] = np.asarray(V_parts[i]).astype(np.float32)
    
    V_parts_combined = np.concatenate(V_parts, axis=0)
    return V_parts_combined, V_parts


def load_neural_deform_seg_data(mesh_paths, device, intermediate=10000, final=2048):
    V_parts_all = []
    V_parts_combined_all = []
    V_surf_all = []
    F_all = []
    E_all = []
    GV_parts_combined_all = []
    GE_all = []
    GV_origin_all = []
    GV_parts_device_all = []
    for i, mesh_path in enumerate(mesh_paths): 
        V, F, E, V_surf = load_mesh(mesh_path)
        V_parts_combined, V_parts = load_segmentation(mesh_path, i, V)
        
        for j in range(len(V_parts)):
            V_parts[j] = torch.from_numpy(V_parts[j])
        V_parts_all.append(V_parts)
        V_parts_combined_all.append(torch.from_numpy(V_parts_combined))
        F_all.append(torch.from_numpy(F))
        E_all.append(torch.from_numpy(E))
        V_surf_all.append(torch.from_numpy(V_surf))
        
        # Graph vertices and edges for computing forward fitting loss.
        GV_parts_combined_all.append(V_parts_combined_all[-1].clone())
        GE_all.append(E_all[-1].clone())
        # Graph vertices origins for computing backward fitting loss.
        GV_origin_all.append(GV_parts_combined_all[-1].clone())

        # Graph vertices on device for deformation.
        GV_parts_device = []
        for j in range(len(V_parts)):
            GV_parts_device.append(V_parts[j].to(device))
        GV_parts_device_all.append(GV_parts_device)
    return (V_parts_all, V_parts_combined_all, F_all, E_all, V_surf_all), \
            (GV_parts_combined_all, GE_all, GV_origin_all, GV_parts_device_all)


