#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=osmesa

python3 ./main.py --config ~/6d_grasp/completion_method/ \
                    --data_path ~/6d_grasp/acronym \
                    --pretrain_ckpt /home/cyf/6d_grasp/completion_method/checkpoints/best_l1_cd_320.pth 
