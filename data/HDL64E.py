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
        phi = np.array(pc[:, 2], dtype=np.float32) # for self.cal_weight_by_HDL64E_lutable
        theta = np.array(pc[:, 3], dtype=np.float32) # for self.cal_weight_by_HDL64E_lutable
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
                weight = self.cal_weight(pc.numpy(), (r>0).numpy(), phi, theta)
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
    
    # def cal_weight(self, pointcloud: np.ndarray, filter_: np.ndarray) -> np.ndarray:
    #     """calculate the weights of the rays

    #     Args:
    #         pointcloud (np.ndarray): shape (-1, 3), xyz coordinates
    #         filter_ (np.ndarray): filter that tells the reached points

    #     Returns:
    #         np.ndarray: weights of the rays
    #     """
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pointcloud[filter_])
    #     pcd.estimate_normals()
    #     pcd.normalize_normals()
    #     nm = np.asarray(pcd.normals)

    #     nms = np.zeros((filter_.shape[0], 3))
    #     nms[filter_] = nm
    #     nms = nms.reshape((128, -1, 3))
    #     nms_padding = np.zeros((nms.shape[0]+2, nms.shape[1]+2, 3))
    #     nms_padding[1:-1, 1:-1] = nms
    #     edges = np.stack([
    #         nms_padding[0:-2, 1:-1]*nms,
    #         nms_padding[0:-2, 0:-2]*nms,
    #         nms_padding[1:-1, 2:]*nms,
    #         nms_padding[1:-1, :-2]*nms,
    #         nms_padding[2:, 2:]*nms,
    #         nms_padding[2:, :-2]*nms,
    #         nms_padding[:-2, 2:]*nms,
    #         nms_padding[2:, 1:-1]*nms
    #     ])
    #     edges = edges.sum(axis=-1).mean(axis=0).reshape(-1)
    #     e = (1-edges)/2
    #     e[~filter_] = 0
    #     weight = e * 0.8 + 0.2 # hard-coded here!
    #     return weight


    def cal_weight(self, pointcloud: np.ndarray, filter_: np.ndarray, phi, theta) -> np.ndarray:
        CHANNELS=64
        def create_vcf_lut():
            """
            Create a Lookup Table (LUT) based on VCF values for the HDL-64E LiDAR.
            """
            laser_data = [
                # Upper 32 Lasers (Header 0xEEFF)
                (1, -7.1581), (2, -6.8178), (3, 0.3178), (4, 0.6581), (5, -6.4777), (6, -6.1376), (7, -8.5208),
                (8, -8.1799), (9, -5.7976), (10, -5.4777), (11, -7.8391), (12, -7.4985), (13, -3.0802), (14, -2.7406),
                (15, -5.1198), (16, -4.7783), (17, -2.4010), (18, -2.0614), (19, -4.4386), (20, -4.0897), (21, -1.7274),
                (22, -1.3820), (23, -3.7594), (24, -3.4198), (25, 0.9985), (26, 1.3391), (27, -1.0423), (28, -0.7026),
                (29, 1.6799), (30, 2.0208), (31, -3.6240), (32, -0.0223),
                
                # Lower 32 Lasers (Header 0xDDFF)
                (33, -22.7379), (34, -22.2261), (35, -11.5139), (36, -11.0021), (37, -21.7147), (38, -21.2037),
                (39, -24.7903), (40, -24.2763), (41, -20.6930), (42, -20.1829), (43, -23.7629), (44, -23.2051),
                (45, -16.6153), (46, -16.1059), (47, -19.6726), (48, -19.1627), (49, -15.5965), (50, -15.0859),
                (51, -18.6530), (52, -18.1435), (53, -14.7573), (54, -14.0674), (55, -17.6341), (56, -17.1247),
                (57, -10.4898), (58, -9.9770), (59, -13.5573), (60, -13.0469), (61, -9.4637), (62, -8.9497),
                (63, -12.5363), (64, -12.0253)
            ]
            
            laser_data.sort(key=lambda x: x[1])
            return np.array(laser_data)

        def find_closest_laser(phi_array, lut):
            vcf_values = lut[:, 1]
            laser_numbers = lut[:, 0]

            # Find the index of the closest VCF values for each phi in phi_array
            closest_indices = np.argmin(np.abs(vcf_values[:, None] - np.degrees(phi_array)), axis=0)
            return laser_numbers[closest_indices].astype(int) - 1
                    
        pcd = o3d.geometry.PointCloud()
        pts_filtered = pointcloud[filter_]
        pcd.points = o3d.utility.Vector3dVector(pts_filtered)
        pcd.estimate_normals()
        pcd.normalize_normals()
        normals_filtered = np.asarray(pcd.normals)  # shape: (M, 3), where M = number of valid points

        x, y, z = pts_filtered[:, 0], pts_filtered[:, 1], pts_filtered[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        phi_vals = np.arcsin(z / (r + 1e-8))  # to avoid division by zero
        theta_vals = np.arctan2(y, x)         # range: [-π, π]
        num_az_bins = 4000
        channel_angles = np.deg2rad(create_vcf_lut()[:, 1].squeeze())

        c_idx = np.abs(phi_vals[:, None] - channel_angles[None, :]).argmin(axis=1)
        theta_norm = (theta_vals + np.pi) / (2.0 * np.pi)  # normalized to [0, 1]
        az_idx = (theta_norm * num_az_bins).astype(np.int32)
        az_idx = np.clip(az_idx, 0, num_az_bins - 1)

        # [Step 4] Accumulate normal vectors in a 2D array (channel x azimuth)
        normal_map = np.zeros((CHANNELS, num_az_bins, 3), dtype=np.float32)
        count_map  = np.zeros((CHANNELS, num_az_bins), dtype=np.int32)

        for i in range(pts_filtered.shape[0]):
            ci, ai = c_idx[i], az_idx[i]
            normal_map[ci, ai] += normals_filtered[i]
            count_map[ci, ai] += 1

        # Average the accumulated normals where valid
        valid_mask = (count_map > 0)
        normal_map[valid_mask] /= count_map[valid_mask][:, None]

        # [Step 5] 8-neighbor comparison to detect edges
        # Pad the array so we can safely look at neighbors of edge cells
        pad = 1
        nms_pad = np.pad(normal_map, ((pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=0)
        
        # Extract the center region matching the original shape
        center = nms_pad[pad:-pad, pad:-pad]
        neighbors = [
            nms_pad[0:-2, 1:-1],
            nms_pad[0:-2, 0:-2],
            nms_pad[1:-1, 2:],
            nms_pad[1:-1, :-2],
            nms_pad[2:, 2:],
            nms_pad[2:, :-2],
            nms_pad[:-2, 2:],
            nms_pad[2:, 1:-1],
        ]
        edge_scores = np.zeros((CHANNELS, num_az_bins), dtype=np.float32)
        for nbr in neighbors:
            # Dot product with the center normals, then sum
            edge_scores += np.sum(nbr * center, axis=-1)
        edge_scores /= 8.0
        
        # Convert dot product to an edge score: (1 - dot_avg) / 2
        e_map = (1.0 - edge_scores) / 2.0
        
        # [Step 6] For each original filtered point, retrieve the edge score from the 2D map
        #           Points that fail the filter will keep weight as 0
        e_vals = np.zeros(pointcloud.shape[0], dtype=np.float32)
        f_indices = filter_.nonzero()[0]
        for i in range(pts_filtered.shape[0]):
            ci, ai = c_idx[i], az_idx[i]
            e_vals[f_indices[i]] = e_map[ci, ai]

        # Final weight = 0.8 * e_vals + 0.2
        weight = e_vals * 0.8 + 0.2
        return weight
