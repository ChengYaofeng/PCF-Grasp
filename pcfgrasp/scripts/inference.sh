#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=osmesa

python ./inference.py --ckpt_dir /home/cyf/6d_grasp/completion_method/checkpoints/specific_model/fuse_points_large_perception/06-13-20-44_best_train_195_plan_h_large.pth \
                    --pretrain_ckpt /home/cyf/6d_grasp/completion_method/checkpoints/best_l1_cd_320.pth 
