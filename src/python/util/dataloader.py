"""ShapeNet deformation dataloader"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')))
import torch
import pyDeform
from torch.utils.data import Dataset, Sampler
import numpy as np
import trimesh
import glob
from collections import OrderedDict
from load_data import load_neural_deform_data, compute_deformation_pairs, load_segmentation_mask
import multiprocessing
import time 

class SAPIENBase(Dataset):
    """Pytorch Dataset base for loading ShapeNet shape pairs.
    """

    def __init__(self, data_root):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
        """
        self.data_root = data_root
        self.files = self._get_filenames(self.data_root)

    @staticmethod
    def _get_filenames(data_root):
        return sorted(glob.glob(os.path.join(data_root, "*.obj"), recursive=True))

    def __len__(self):
        return self.n_shapes

    @property
    def n_shapes(self):
        return len(self.files)

    def idx_to_combinations(self, idx):
        """Convert s linear index to a pair of indices."""
        if hasattr(idx, "__len__"):
            i = np.array([0 for _ in range(len(idx))], dtype=int)
            j = np.array(j, dtype=int)
        else:
            i = 0
            j = int(idx)
        return i, j

    def combinations_to_idx(self, i, j):
        """Convert a pair of indices to a linear index."""
        if hasattr(j, "__len__"):
            idx = np.array(j, dtype=int)
        else:
            idx = int(j)
        return idx
   

class SAPIENMesh(SAPIENBase):
    """Pytorch Dataset for sampling entire meshes."""

    def __init__(self, data_root):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
        """
        super(SAPIENMesh, self).__init__(data_root=data_root)
        pool = multiprocessing.Pool(processes=4)
        self.data = pool.map(load_neural_deform_data, self.files)
        self.segmentation_masks = pool.map(load_segmentation_mask, self.files)

        # Preprocess all the shapes.
        print("Preprocessing shapes!")
        time_start = time.time()
        self.param_ids = []
        for (V, F, _, _, GV, GE) in self.data:
            param_id = pyDeform.InitializeDeformTemplate(V, F, 0, 64)
            pyDeform.NormalizeByTemplate(GV, param_id)
            pyDeform.StoreGraphInformation(GV, GE, param_id)
            self.param_ids.append(param_id)
        print("Done preprocessing shapes:", time.time() - time_start)

    def get_pairs(self, i, j):
        data_i, seg_i = self.get_single(i)
        data_j, seg_j = self.get_single(j)
        return i, j, self.param_ids[i], self.param_ids[j], data_i, data_j, seg_i, seg_j

    def get_single(self, i):
        data_i = [x.clone() for x in self.data[i]]
        if self.segmentation_masks[i] is not None:
            mask_i = self.segmentation_masks[i].clone()
        else:
            mask_i = None
        return self.data[i], self.segmentation_masks[i]

    def __getitem__(self, idx):
        """Get a random pair of meshes.
        Args:
          idx: int, index of the shape pair to return. must be smaller than len(self).
        Returns:
          i: index for first mesh.
          j: index for second mesh.
          verts_i: [#vi, 3] float tensor for vertices from the first mesh.
          faces_i: [#fi, 3] int32 tensor for faces from the first mesh.
          edges_i: [#ei, 2] int32 tensor for edges from the first mesh.
          v_sample_i: [2048, 3] float tensor for points sampled from the first mesh.
          gverts_i: [#vi, 3] float tensor for vertices from the first mesh.
          gedges_i: [#ei, 2] int32 tensor for edges from the first mesh.
          verts_j: [#vj, 3 or 6] float tensor for vertices from the second mesh.
          faces_j: [#fj, 3 or 6] int32 tensor for faces from the second mesh.
          edges_j: [#ej, 2] int32 tensor for edges from the second mesh.
          v_sample_j: [2048, 3] float tensor for points sampled from the second mesh.
          gverts_j: [#vj, 3] float tensor for vertices from the second mesh.
          gedges_j: [#ej, 2] int32 tensor for edges from the second mesh.
        """
        i, j = self.idx_to_combinations(idx)
        return self.get_pairs(i, j)


class RandomPairSampler(Sampler):
    """Data sampler for sampling random pairs."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.pairs = compute_deformation_pairs(0, dataset.n_shapes)
    def __iter__(self):
        pairs = np.random.permutation(self.pairs)
        src_idxs = []
        tar_idxs = []
        for (i, j) in pairs:
            src_idxs.append(i)
            tar_idxs.append(j)
        combo_ids = self.dataset.combinations_to_idx(src_idxs, tar_idxs)
        return iter(combo_ids)

    def __len__(self):
        return len(self.dataset)
