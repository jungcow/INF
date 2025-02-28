#!/bin/bash

set -xe

source=$1
output=$2

python preprocess/prepare_ext_json.py --input_path ${source}/params/cams_to_lidar_gt.txt  --output_path ${output}

python preprocess/generate_poses_npy.py --l2w_path ${source}/params/lidars.txt --output_path ${output}

python preprocess/prepare_scans.py --pcd_path ${source}/pcds/ --output_path ${output}

python preprocess/prepare_images.py --input_path ${source} --output_path ${output}