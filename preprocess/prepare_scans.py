import numpy as np
import open3d as o3d
import argparse
import os
from tqdm import tqdm

"""
Convert point clouds(.pcd) to scans(.npy)
"""

def load_xyz_from_pcd(file_path):
    """ Load x, y, z data from a PCD file using Open3D. """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points


def pcd_to_lidarscan(pcd_path):
    # Extract x, y, z from the PCD file
    points = load_xyz_from_pcd(pcd_path)

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
    return output_data


def main(args):
    pcd_path = args.pcd_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    print(len(os.listdir(pcd_path)))
    pcd_files = sorted([ f for f in os.listdir(pcd_path) if os.path.isfile(os.path.join(pcd_path, f)) and f.endswith('.pcd') ])
    print(f"{len(pcd_files)} of pcd files found")
    for idx, pcd_filename in enumerate(tqdm(pcd_files)):
        pcd_filepath = os.path.join(pcd_path, pcd_filename)
        lidarscan = pcd_to_lidarscan(pcd_filepath)

        npy_path = f"{idx:04d}.npy"  # Change this to the desired output path
        np.save(os.path.join(output_path, npy_path), lidarscan)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_path', type=str, required=True, help='Parents directory of .pcd files')
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Parents directory of output .npy files')
    args = parser.parse_args()

    main(args)