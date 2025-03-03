import numpy as np
import open3d as o3d
import argparse
import os
from tqdm import tqdm

def load_xyz_from_pcd(file_path):
    """Loads x, y, z data from a PCD file using Open3D."""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

# for waymo
def assign_channel_from_vfov(scan_data, vfov_min_deg=-17.6, vfov_max_deg=2.4, num_channels=64):
    """
    Converts the VFOV range to radians and assigns a channel to each point based on its φ value.
    Assumes that the central angles of each channel are evenly distributed.
    """
    # Convert VFOV to radians
    vfov_min = np.deg2rad(vfov_min_deg)
    vfov_max = np.deg2rad(vfov_max_deg)
    # Calculate the angular increment between channels (num_channels=64 means 63 intervals)
    angle_increment = (vfov_max - vfov_min) / (num_channels - 1)
    
    # The third column of scan_data is already the φ value (radians)
    phi_values = scan_data[:, 2]
    
    # Compute the closest channel index by rounding (φ - vfov_min) / angle_increment
    channel_indices = np.rint((phi_values - vfov_min) / angle_increment).astype(int)
    
    # Ensure indices remain within the range of 0 to (num_channels - 1)
    channel_indices = np.clip(channel_indices, 0, num_channels - 1)
    
    return channel_indices

def pcd_to_lidarscan(pcd_path, vfov_min_deg=-17.6, vfov_max_deg=2.4, num_channels=64):
    """
    Loads a PCD file and converts it to the format [beam_origin_x, beam_origin_y, φ, θ, distance].
    Assigns a channel to each point based on the VFOV and appends it as the 6th column.
    """
    # Load x, y, z data from the PCD file
    points = load_xyz_from_pcd(pcd_path)
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    distance = np.linalg.norm(points, axis=1)
    # Compute φ values (radians) using the z/distance ratio
    phi = np.arcsin(z / (distance + 1e-6))  # Prevent division by zero errors
    theta = np.arctan2(y, x)
    
    # Assume beam origins are at (0,0)
    beam_origins_x = np.zeros_like(x)
    beam_origins_y = np.zeros_like(y)
    
    # Create the base scan data: [beam_origins_x, beam_origins_y, φ, θ, distance]
    scan_data = np.stack([beam_origins_x, beam_origins_y, phi, theta, distance], axis=1)
    
    # Assign channels based on VFOV
    channels = assign_channel_from_vfov(scan_data, vfov_min_deg, vfov_max_deg, num_channels)
    
    # Append channel information to scan_data: [beam_origin_x, beam_origin_y, φ, θ, distance, channel]
    output_data = np.concatenate([scan_data, channels[:, None]], axis=1)
    return output_data

def main(args):
    pcd_path = args.pcd_path
    output_path = os.path.join(args.output_path, 'scans')
    os.makedirs(output_path, exist_ok=True)

    vfov_min_deg = args.vfov_min_deg
    vfov_max_deg = args.vfov_max_deg
    num_channels = args.num_channels
    
    pcd_files = sorted([f for f in os.listdir(pcd_path) if os.path.isfile(os.path.join(pcd_path, f)) and f.endswith('.pcd')])
    print(f"{len(pcd_files)} PCD files found.")
    for idx, pcd_filename in enumerate(tqdm(pcd_files)):
        pcd_filepath = os.path.join(pcd_path, pcd_filename)
        lidarscan = pcd_to_lidarscan(pcd_filepath, 
                                     vfov_min_deg=vfov_min_deg, 
                                     vfov_max_deg=vfov_max_deg, 
                                     num_channels=num_channels)
        
        npy_filename = f"{idx:04d}.npy"
        np.save(os.path.join(output_path, npy_filename), lidarscan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_path', type=str, required=True, help='Parent directory containing PCD files')
    parser.add_argument('--vfov_min_deg', type=float, required=True, help='Waymo: -17.6 | KITTI: -24.9 | nuScenes: -30.67')
    parser.add_argument('--vfov_max_deg', type=float, required=True, help='Waymo: +2.4 | KITTI: +2 | nuScenes: 10.67')
    parser.add_argument('--num_channels', type=int, required=True, help='Waymo & KITTI: 64 | nuScenes: 32')
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Parent directory where scans/%%04d.npy files will be saved')
    args = parser.parse_args()
    
    main(args)
