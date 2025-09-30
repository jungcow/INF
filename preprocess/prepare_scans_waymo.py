import numpy as np
import open3d as o3d
import argparse
import os
from tqdm import tqdm

################################################################################
# For Waymo dataset: original pcd files are in world coordinate 
# -> convert them to local coordinate to use the same pipeline as INF outdoor dataset
################################################################################

"""
Convert point clouds(.pcd) to scans(.npy)
- pcds/%06d.pcd -> scans/%04.npy
- scans: polar coordinate representing the points into ray 
"""

def load_xyz_from_pcd(file_path, l2w):
    """ Load x, y, z data from a PCD file using Open3D. """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    w2l = np.linalg.inv(l2w)
    points_hom = np.ones((points.shape[0], 4)).astype(np.float32)
    points_hom[:, :3] = points
    points_local = points_hom @ w2l.T
    return points_local[:, :3]

def pcd_to_lidarscan(pcd_path, l2w):
    # Extract x, y, z from the PCD file
    points = load_xyz_from_pcd(pcd_path, l2w)

    # Convert Cartesian coordinates (x, y, z) to spherical coordinates (phi, theta, distance)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    distance = np.linalg.norm(points, axis=1)
    phi = np.arcsin(z / (distance + 1e-6))  # Avoid division by zero
    theta = np.arctan2(y, x)

    # Assume beam origins are at (0,0) for now
    beam_origins_x = np.zeros_like(x)
    beam_origins_y = np.zeros_like(y)

    # Construct the output array in the specified format
    output_data = np.stack([beam_origins_x, beam_origins_y, phi, theta, distance], axis=1)
    return output_data, points


def main(args):
    pcd_path = args.pcd_path
    lidar_pose_path = args.lidar_pose_path
    scans_output_path = os.path.join(args.output_path, 'scans')
    pcd_output_path = os.path.join(args.output_path, 'pcds')
    os.makedirs(scans_output_path, exist_ok=True)
    os.makedirs(pcd_output_path, exist_ok=True)

    l2w = np.loadtxt(lidar_pose_path).reshape(-1, 4, 4)

    print(len(os.listdir(pcd_path)))
    pcd_files = sorted([ f for f in os.listdir(pcd_path) if os.path.isfile(os.path.join(pcd_path, f)) and f.endswith('.pcd') ])
    print(f"{len(pcd_files)} of pcd files found")
    for idx, pcd_filename in enumerate(tqdm(pcd_files)):
        pcd_filepath = os.path.join(pcd_path, pcd_filename)
        lidarscan, points = pcd_to_lidarscan(pcd_filepath, l2w[idx])

        npy_path = f"{idx:04d}.npy"  # Change this to the desired output path
        np.save(os.path.join(scans_output_path, npy_path), lidarscan)

        # also save point clouds on .ply format
        o3d.io.write_point_cloud(os.path.join(pcd_output_path, f"{idx:04d}.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_path', type=str, required=True, help='Parents directory of .pcd files')
    parser.add_argument('--lidar_pose_path', type=str, required=True, help="filepath of lidars.txt")
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Parents directory of scans/%%04d.npy files')
    args = parser.parse_args()

    main(args)