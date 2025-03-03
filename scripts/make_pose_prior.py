import os
import glob
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def euler_and_translation_to_matrix(euler_angles, translation):
    # Convert Euler angles (in degrees, order: xyz) and translation vector to a 4x4 transformation matrix
    rotation_matrix = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = translation
    return extrinsic

def process_test(test_dir, output_filename):
    # Process each camera folder in the test directory:
    # Read the res.json file, convert the parsed Euler angles and translation into a 4x4 extrinsic matrix,
    # then flatten the matrix and write it to the pose_prior file with the camera ID.
    lines = []
    for cam_id in range(4):
        # Search for res.json in folders matching calib_cam{cam_id}_*
        pattern = os.path.join(test_dir, f"calib_cam{cam_id}_*", "res.json")
        res_files = glob.glob(pattern)
        if not res_files:
            print(f"Warning: No res.json found for camera {cam_id} in {test_dir}")
            continue
        res_file = res_files[0]  # If multiple files are found, use the first one
        with open(res_file, 'r') as f:
            data = json.load(f)
        euler_rotation = data["rotation"]
        translation = data["translation"]
        extrinsic = euler_and_translation_to_matrix(euler_rotation, translation) # l2c -> c2l
        # Flatten the 4x4 extrinsic matrix into a 1x16 vector (row-major order)
        extrinsic_flat = extrinsic.flatten()
        extrinsic_str = " ".join(f"{num:.8f}" for num in extrinsic_flat)
        line = f"{cam_id} {extrinsic_str}"
        lines.append(line)
    
    # Write the collected lines into the pose_prior file for the current test
    output_path = os.path.join(test_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Wrote {output_filename} in {test_dir}")

def main():
    # Parse command-line arguments for group and output directory
    parser = argparse.ArgumentParser(description="Generate pose_prior files from res.json files.")
    parser.add_argument("--group", required=True, help="Group name (e.g., groupA)")
    parser.add_argument("--output", default="output", help="Output directory root (default: output)")
    parser.add_argument("--from_lidar", action='store_true', help="from lidar test")
    args = parser.parse_args()

    # Construct the batch directory path based on the provided group and output arguments
    batch_dir = os.path.join(args.output, args.group, "batch")
    if args.from_lidar:
        batch_dir = batch_dir + '_from_lidar'
    # Iterate over test0 to test9 directories
    for i in range(10):
        test_dir = os.path.join(batch_dir, f"test{i}")
        if not os.path.exists(test_dir):
            print(f"Test directory {test_dir} does not exist. Skipping...")
            continue
        output_filename = f"pose_prior{i}.txt"
        process_test(test_dir, output_filename)

if __name__ == "__main__":
    main()
