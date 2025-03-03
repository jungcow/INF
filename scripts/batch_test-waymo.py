import os
import json
import shutil
import argparse
import subprocess
import numpy as np
import glob

def copy_density_to_batch(group, iter_num, from_lidar):
    """Copy {group}/density/ directory to {group}/batch/test{iter}"""
    src_density = os.path.join(group, "density")
    if from_lidar:
        dst_batch = os.path.join(group, "batch_from_lidar", f"test{iter_num}")
    else:
        dst_batch = os.path.join(group, "batch", f"test{iter_num}")
    dst_density = os.path.join(dst_batch, "density")
    
    if not os.path.exists(dst_batch):
        os.makedirs(dst_batch)
    
    if os.path.exists(src_density):
        shutil.copytree(src_density, dst_density, dirs_exist_ok=True)
    else:
        print(f"[Warning] {src_density} does not exist!")

def run_batch(args, iter_num, from_lidar):
    """Execute main.py"""
    if from_lidar:
        group_path = os.path.join(args.group, "batch_from_lidar", f"test{iter_num}")
    else:
        group_path = os.path.join(args.group, "batch", f"test{iter_num}")
    
    cmd = [
        "python", "main.py",
        "--model=multicolor",
        "--yaml=color_waymo_cam0,color_waymo_cam1,color_waymo_cam2,color_waymo_cam3,color_waymo_cam4",
        "--density_name=density",
        f"--group={group_path}",
        f"--data.scene={args.scene}"
    ]
    
    if not from_lidar:
        cmd.extend([f"--train.rot_noise={args.rot_noise}", f"--train.trans_noise={args.trans_noise}"])

    subprocess.run(cmd)

def aggregate_results(group, num_batches, from_lidar):
    """Aggregate batch results and generate res_batch.json"""
    if from_lidar:
        batch_path = os.path.join('output', group, "batch_from_lidar")
    else:
        batch_path = os.path.join('output', group, "batch")
    
    all_results = {}
    all_results["total"] = []
    
    for i in range(num_batches):
        test_path = os.path.join(batch_path, f"test{i}")
        cam_folders = glob.glob(os.path.join(test_path, "calib_cam*"))
        
        for cam_folder in cam_folders:
            cam_name = os.path.basename(cam_folder).rsplit("_", 1)[0]  # Extract base cam name
            res_path = os.path.join(cam_folder, "res.json")
            
            if cam_name not in all_results:
                all_results[cam_name] = []
            
            if os.path.exists(res_path):
                with open(res_path, 'r') as f:
                    all_results[cam_name].append(json.load(f))
                    all_results["total"].append(all_results[cam_name][-1])
    
    aggregated_data = {}
    for key in all_results:
        if all_results[key]:
            aggregated_data[key] = compute_statistics(all_results[key])
    
    with open(os.path.join(batch_path, "res_batch.json"), 'w') as f:
        json.dump(aggregated_data, f, indent=4)

def compute_statistics(results):
    """Compute mean and standard deviation from results list"""
    keys = results[0].keys()
    aggregated = {}
    
    for key in keys:
        values = [res[key] for res in results]
        if isinstance(values[0], list):  # Vector data
            values = np.array(values)
            aggregated[key] = {
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist()
            }
        else:  # Scalar data
            values = np.array(values)
            aggregated[key] = {
                "mean": float(values.mean()),
                "std": float(values.std())
            }
    
    return aggregated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True, help="Group path")
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument("--rot_noise", type=float, default=0.0, help="Rotation noise")
    parser.add_argument("--trans_noise", type=float, default=0.0, help="Translation noise")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches")
    args = parser.parse_args()
    
    group = os.path.join('output', args.group)
    for i in range(0, args.num_batches + 0):
        copy_density_to_batch(group, i, from_lidar=False)
        run_batch(args, i, from_lidar=False)
    aggregate_results(args.group, args.num_batches, from_lidar=False)
    print("Batch (GT) processing complete. Results saved to res_batch.json")

    for i in range(args.num_batches):
        copy_density_to_batch(group, i, from_lidar=True)
        run_batch(args, i, from_lidar=True)
    aggregate_results(args.group, args.num_batches, from_lidar=True)
    print("Batch (from_lidar) processing complete. Results saved to res_batch.json")

if __name__ == "__main__":
    main()
