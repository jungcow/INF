#!/bin/bash

set -xe

source=$1
output=$2

python preprocess/prepare_ext_json.py --input_path ${source}/params/cams_to_lidar_gt.txt  --output_path ${output}

python preprocess/generate_poses_npy.py --l2w_path ${source}/params/lidars.txt --output_path ${output}

python preprocess/prepare_scans.py --pcd_path ${source}/pcds/ --lidar_pose_path ${source}/params/lidars.txt --output_path ${output}
# python preprocess/prepare_scans_w_channels.py --pcd_path ${source}/pcds/ --output_path ${output} --vfov_min_deg=-17.6 --vfov_max_deg=2.4 --num_channels=64

python preprocess/prepare_images.py --input_path ${source} --output_path ${output}