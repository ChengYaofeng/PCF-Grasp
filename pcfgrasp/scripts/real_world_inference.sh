#!/bin/bash

set -x
set -e

# export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM='osmesa'
# export PYOPENGL_PLATFORM=egl

# python ./run_tools/real_world_inference.py --filter False \
#                     --ckpt_dir /home/franka/cg_ws/src/contact_graspnet_ros/src/completion_method/checkpoints/all_grasp/03-17-05-02_best_ori_69.pth \
#                     --ori_inference True \

python ./run_tools/real_world_inference.py \
                    --pretrain_ckpt /home/cyf/04_grasp/PCF-Grasp/pcfgrasp/checkpoints/best_l1_cd_320.pth \
                    --filter False \
                    --ckpt_dir /home/cyf/04_grasp/PCF-Grasp/pcfgrasp/checkpoints/06-24-10-40_best_train_289.pth \
                    # /home/franka/cg_ws/src/contact_graspnet_ros/src/completion_method/checkpoints/specific_model/best_ori/06-18-10-20_best_ori_277.pth \
                    # --ori_inference True \
                    # --rot_matrix=[0.5316130258461541, -0.8440791731728401, -0.062417800388337, 0.0319680834776275] \
                    # --tra_matrix=[0.5626850876093676, -0.42399062888756656, 0.5810832286350102]
                    # --ckpt_dir /home/bh/bh_ws/src/contact_graspnet_ros/src/completion_method/checkpoints/specific_model/fuse_points_large_perception/06-13-20-44_best_train_195_plan_h_large.pth \