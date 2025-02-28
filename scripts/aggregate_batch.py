import os
import json
import argparse
import numpy as np
import glob

def aggregate_results(group, num_batches, from_lidar=False):
    """Aggregate batch results and generate res_batch.json"""
    if from_lidar:
        batch_path = os.path.join(group, "batch_from_lidar")
    else:
        batch_path = os.path.join(group, "batch")
    
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
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches")
    parser.add_argument("--from_lidar", action='store_true', help="from lidar test")
    args = parser.parse_args()
    
    aggregate_results(args.group, args.num_batches, args.from_lidar)
    print("Aggregation complete. Results saved to res_batch.json")

if __name__ == "__main__":
    main()
