from .base import Dataset3D
import os
import torch
import numpy as np
import open3d as o3d
from typing import Dict, Any
from scipy.spatial import KDTree
from tqdm import tqdm
import matplotlib.pyplot as plt

class Dataset(Dataset3D):
    """
    Specified for current data that are measured from Ouster
    
    """
    def get_scan(self, idx: int) -> Dict[str, Any]:
        # In current data, a lidar scan is saved in a .npy file as the follows:
        # 0th column, 1st column: x, y coordinate of the origins of LiDAR beams
        # 2nd column, 3rd column: phi and theta angle in spherical coordinates
        # 4th column: distance of the surface that the beam reached or 0 if no surface reached

        pc = self.load_file(idx)
        origin = torch.column_stack([pc[:, :2], torch.zeros_like(pc[:, 0])])
        z = torch.sin(pc[:, 2])
        y = torch.cos(pc[:, 2]) * torch.sin(pc[:, 3])
        x = torch.cos(pc[:, 2]) * torch.cos(pc[:, 3])
        dirs = torch.column_stack((x, y, z))
        r = pc[:, 4]

        # filter out the too far points
        f = (r<=self.pc_range[1])

        # filter out the points outside the mask
        if self.mask is not None:
            sy, sx = self.mask.shape[:2]
            x = ((-pc[:, 3] + torch.pi) / (torch.pi * 2) * sx).to(torch.long) 
            y = ((0.5-pc[:, 2] / torch.pi) * sy).to(torch.long)
            mask = self.mask[y, x]
            f = f & mask

        # load or calculate the weights of the rays
        if self.opt.train.use_weight:
            os.makedirs(os.path.join(self.path, "weight"), exist_ok=True)
            weight_path = os.path.join(self.path, "weight", f"{self.list[idx]:04}.npy")
            if os.path.exists(weight_path):
                weight = torch.from_numpy(np.load(weight_path))
            else:
                pc = origin + dirs * r[..., None]
                weight = self.cal_weight(pc.numpy(), (r>0).numpy())
                np.save(weight_path, weight)
                weight = torch.from_numpy(weight).float()
        else:
            weight = torch.ones_like(r)

        return {
            "dir": dirs[f], # mandatory
            "z": r[f], # mandatory
            "size": r[f].shape[0], # mandatory
            "weight": weight[f], # if not applicable, fill it with torch.ones()
            "origin": origin[f] # if not applicable, fill it with torch.zeros()
        }
    
    def load_file(self, idx: int) -> torch.Tensor:
        """load the LiDAR scan of certain index

        Args:
            idx (int): index of the scan

        Returns:
            torch.Tensor: torch.Tensor with raw data, dtype is torch.float
        """
        file_path = os.path.join(self.path, "scans", f"{self.list[idx]:04}.npy")
        pc = torch.from_numpy(np.load(file_path)).to(torch.float)
        return pc

    def cal_weight(self,
                    pts: np.ndarray,
                    filter: np.ndarray,
                    k: int = 16,
                    alpha: float = 0.3,
                    clip_pct: float = 2.0,
                    gamma: float = 0.7,
                    orient_to_origin: bool = True) -> np.ndarray:
        """
        e = (1-a)*normal_inconsistency + a*surface_variation, norm + gamma
        - normal_inconsistency = (1 - mean_cos)/2
        - surface_variation = l0 / (l0+l1+l2) (l0 <= l1 <= l2: eigenvalues of covariance matrix)
        Args:
            pts (np.ndarray): (N,3) xyz coordinates
            filter (np.ndarray): (N,) boolean array for valid points
            k (int, optional): number of neighbors for local PCA. Defaults to 16.
            alpha (float, optional): weight for surface variation. Defaults to 0.3.
            clip_pct (float, optional): percentage for clipping the weights. Defaults to 2.0.
            gamma (float, optional): gamma correction factor. Defaults to 0.7.
            orient_to_origin (bool, optional): whether to orient normals to the origin. Defaults to True.
        """

        full_N = pts.shape[0]
        if not np.any(filter):
            return np.zeros((full_N,), dtype=np.float32)

        filtered_pts = pts[filter]

        N = len(filtered_pts)
        k_eff = min(k, max(3, N))
        idx = knn_indices(filtered_pts, k=k_eff)

        normals = np.zeros((N, 3), dtype=np.float32)
        sv = np.zeros((N,), dtype=np.float32)

        for i in range(N):
            w, v = pca_eigs(filtered_pts[idx[i]])
            n = v[:, 0]  # smallest eigenvector
            if orient_to_origin and np.dot(n, filtered_pts[i]) > 0:
                n = -n
            normals[i] = n / (np.linalg.norm(n) + 1e-8)
            sv[i] = float(w[0] / (w.sum() + 1e-12))

        # cosine similarity average
        nbr_n = normals[idx] # (N,k,3)
        cosm = np.einsum('nkj,nj->nk', nbr_n, normals).mean(axis=1)
        e_n = (1.0 - np.clip(cosm, -1.0, 1.0)) * 0.5 # [0,1]

        # surface variation + cosine similarity 
        e_raw = (1.0 - alpha) * e_n + alpha * sv

        # clipping + normalization + gamma
        e = e_raw.copy()
        if clip_pct > 0:
            lo = np.percentile(e, clip_pct)
            hi = np.percentile(e, 100.0 - clip_pct)
            if hi > lo:
                e = np.clip(e, lo, hi)
        e01 = (e - e.min()) / (e.max() - e.min() + 1e-12)
        if gamma != 1.0:
            e01 = np.clip(e01, 0.0, 1.0) ** float(gamma)

        out = np.zeros((full_N,), dtype=np.float32)
        out[filter] = e01.astype(np.float32)
        return out

def knn_indices(pts: np.ndarray, k: int) -> np.ndarray:
    """
    (N,3) -> (N,k) neighbor indices. Uses scipy cKDTree if available, otherwise brute force.
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=k)
        return idx
    except Exception:
        # For experimental use: recommended only for small scale
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        idx = np.argpartition(D, kth=range(k), axis=1)[:, :k]
        return idx

def pca_eigs(neigh: np.ndarray):
    """
    Neighbors (m,3) -> covariance eigen decomposition: (eigs asc, eigvecs).
    """
    X = neigh - neigh.mean(axis=0)
    C = (X.T @ X) / max(len(neigh) - 1, 1)
    w, v = np.linalg.eigh(C)   # w asc: λ0<=λ1<=λ2
    return w, v