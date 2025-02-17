"""
params/cams_to_lidar -> ref_ext.json
"""
import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation as R

def load_camera_matrices(input_path):
    """Load camera transformation matrices from a file."""
    camera_matrices = {}
    with open(input_path, 'r') as f:
        lines = f.readlines()
        cam_name = None
        matrix = []
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                if cam_name and matrix:
                    camera_matrices[cam_name] = np.array(matrix)
                cam_name = line[2:]
                matrix = []
            else:
                matrix.append(list(map(float, line.split())))
        if cam_name and matrix:
            camera_matrices[cam_name] = np.array(matrix)
    return camera_matrices

def main():
    parser = argparse.ArgumentParser(description="Convert cams_to_lidar_gt.txt to ref_ext.json")
    parser.add_argument("--input_path", type=str, help="Path to cams_to_lidar_gt.txt")
    parser.add_argument("--output_path", type=str, help="Path to save ref_ext.json")
    args = parser.parse_args()

    camera_matrices = load_camera_matrices(args.input_path)
    
    ref_ext = []
    for cam, mat in camera_matrices.items():
        inv_mat = np.linalg.inv(mat)
        rotation_l2c, translation_l2c = inv_mat[:3, :3], inv_mat[:3, 3]

        # cam_to_INFcam = np.array([
        #     [0, -1, 0],
        #     [0, 0, -1],
        #     [1, 0, 0]
        # ], dtype=np.float32)

        # cam_to_INFcam = np.linalg.inv(cam_to_INFcam)

        rotation_lidar_to_INFcam, translation_lidar_to_INFcam = rotation_l2c, translation_l2c

        euler_rotation = R.from_matrix(rotation_lidar_to_INFcam).as_euler('xyz', degrees=True)
        ref_ext.append({
            "rotation": euler_rotation.tolist(),
            "translation": translation_lidar_to_INFcam.tolist()
            })

    with open(args.output_path, "w") as f:
        json.dump(ref_ext, f, indent=4)

    print(f"Conversion completed. ref_ext.json has been saved at {args.output_path}.")

if __name__ == "__main__":
    main()
