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


def load_mesh_tensors(mesh_path, intermediate=10000, final=2048):
    mesh = trimesh.load(mesh_path, process=False)
    V = mesh.vertices.astype(np.float32)
    F = mesh.faces.astype(np.int32)
    E = mesh.edges.astype(np.int32)

    # Surface vertices for PointNet input.
    V_sample, _, _ = sample_faces(V, F, n_samples=intermediate)
    V_sample, _ = fps(V_sample, final)
    V_sample = V_sample.astype(np.float32)
    
    V = torch.from_numpy(V)
    F = torch.from_numpy(F)
    E = torch.from_numpy(E)
    V_sample = torch.from_numpy(V_sample)
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
        V, F, E, V_surf = load_mesh_tensors(mesh_path)
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
