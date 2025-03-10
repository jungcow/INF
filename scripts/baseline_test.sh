#!/bin/bash
 
set -xe
 
source_root=$1
# scene=$2
root=$2

# scene_list=("small_rot" "large_zigzag")  # Add more scenes as needed
scene_list=("large_zigzag")  # Add more scenes as needed

for scene in "${scene_list[@]}"; do
    # source /opt/miniconda3/etc/profile.d/conda.sh
    # conda activate inf


    # bash ${root}/scripts/preprocess.sh ${source_root}/${scene} ${root}/data/kitti-360_${scene}

    # python3 ${root}/main.py --model=density  --yaml="density_kitti-360_${scene}"

    # python ${root}/scripts/batch_test-kitti.py --group="kitti-360_${scene}" --scene="kitti-360_${scene}" --num_batches=10



    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate gaussian_splatting

    # # from GT
    # python ${root}/scripts/make_pose_prior.py --group="kitti-360_${scene}" --output="output"
    # seq 0 9 | xargs -I{} mkdir -p ${root}/../vanilla-3dgs/output/INF/kitti-360/${scene}/batch_gt/test{}/point_cloud/iteration_30000

    # seq 0 9 | xargs -I{} cp -p ${root}/output/kitti-360_${scene}/batch/test{}/pose_prior{}.txt ${root}/../vanilla-3dgs/output/INF/kitti-360/${scene}/batch_gt/test{}/point_cloud/iteration_30000/cams_to_lidar.txt

    # python ${root}/../vanilla-3dgs/full_eval_rig.py -s ${source_root}/${scene} -m ${root}/../vanilla-3dgs/output/INF/kitti-360/${scene}/batch_gt -n INF -r ${root}/../vanilla-3dgs

    # from lidar
    python ${root}/scripts/make_pose_prior.py --group="kitti-360_${scene}" --output="output" --from_lidar
    seq 0 9 | xargs -I{} mkdir -p ${root}/../vanilla-3dgs/output/INF/kitti-360/${scene}/batch_from_lidar/test{}/point_cloud/iteration_30000

    seq 0 9 | xargs -I{} cp -p ${root}/output/kitti-360_${scene}/batch_from_lidar/test{}/pose_prior{}.txt ${root}/../vanilla-3dgs/output/INF/kitti-360/${scene}/batch_from_lidar/test{}/point_cloud/iteration_30000/cams_to_lidar.txt

    python ${root}/../vanilla-3dgs/full_eval_rig.py -s ${source_root}/${scene} -m ${root}/../vanilla-3dgs/output/INF/kitti-360/${scene}/batch_from_lidar -n INF -r ${root}/../vanilla-3dgs

done