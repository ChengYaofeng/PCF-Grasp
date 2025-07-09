#!/usr/bin/env bash

set -x
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=0

python  main.py --exp_name='pretrain' \
                --config ~/6d_grasp/completion_method/ \
                --data_path ~/6d_grasp/acronym  \
                --pretrain=True \
                --batch_size 4 \
                --ckpt_dir /home/cyf/6d_grasp/completion_method/checkpoints/best_l1_cd_320.pth

