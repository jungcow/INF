import numpy as np
import argparse
import os

def save_poses(l2w_path, output_path):
    # Load the pose data from the file
    poses = np.loadtxt(l2w_path).reshape(-1, 4, 4)
    
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save the poses as a NumPy binary file
    output_file = os.path.join(output_path, "poses.npy")
    np.save(output_file, poses)
    print(f"Saved poses to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save pose data as a NumPy .npy file")
    parser.add_argument("--l2w_path", type=str, help="Path to the input pose file (including filename)")
    parser.add_argument("--output_path", type=str, help="Directory to save the output .npy file")
    
    args = parser.parse_args()
    
    save_poses(args.l2w_path, args.output_path)
