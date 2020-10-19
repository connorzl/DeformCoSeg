"""ShapeNet deformation dataloader"""
import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import trimesh
import glob
from collections import OrderedDict
from load_data import load_neural_deform_data, compute_deformation_pairs


class SAPIENBase(Dataset):
    """Pytorch Dataset base for loading ShapeNet shape pairs.
    """

    def __init__(self, data_root, single_source_idx=-1):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
          single_source_idx: int, if non-negative, then only load (src, targ_i)
        """
        self.data_root = data_root
        self.files = self._get_filenames(self.data_root)
        self.single_source_idx = single_source_idx

    @staticmethod
    def _get_filenames(data_root):
        return sorted(glob.glob(os.path.join(data_root, "*.obj"), recursive=True))

    def __len__(self):
        if self.single_source_idx >= 0:
            return self.n_shapes
        else:
            return self.n_shapes ** 2

    @property
    def n_shapes(self):
        return len(self.files)

    def idx_to_combinations(self, idx):
        """Convert s linear index to a pair of indices."""
        if self.single_source_idx == -1:
            if hasattr(idx, "__len__"):
                i = []
                j = []
                for k in range(len(idx)):
                    i.append(np.floor(idx[k] / self.n_shapes))
                    j.append(idx[k] - i[k] * self.n_shapes)
            else:
                i = np.floor(idx / self.n_shapes)
                j = idx - i * self.n_shapes
        else:
            if hasattr(idx, "__len__"):
                i = len(idx) * [self.single_source_idx]
            else:
                i = self.single_source_idx
            j = idx

        if hasattr(idx, "__len__"):
            i = np.array(i, dtype=int)
            j = np.array(j, dtype=int)
        else:
            i = int(i)
            j = int(j)
        return i, j

    def combinations_to_idx(self, i, j):
        """Convert a pair of indices to a linear index."""
        if self.single_source_idx == -1:
            idx = []
            for k in range(len(i)):
                idx.append(i[k] * self.n_shapes + j[k])
        else:
            idx = j
        
        if hasattr(idx, "__len__"):
            idx = np.array(idx, dtype=int)
        else:
            idx = int(idx)
        return idx
    
class SAPIENMesh(SAPIENBase):
    """Pytorch Dataset for sampling entire meshes."""

    def __init__(self, data_root, single_source_idx=-1):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
        """
        super(SAPIENMesh, self).__init__(data_root=data_root, single_source_idx=single_source_idx)

    def get_pairs(self, i, j):
        data_i = self.get_single(i)
        data_j = self.get_single(j)
        return i, j, data_i, data_j

    def get_single(self, i):
        return load_neural_deform_data(self.files[i])

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
        self.pairs = compute_deformation_pairs(dataset.single_source_idx, dataset.n_shapes)
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
        return len(dataset)

