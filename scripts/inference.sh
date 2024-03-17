#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=egl

python ./inference.py --ckpt_dir '/home/cyf/PCF-Grasp/pcfgrasp_method/checkpoints/train/05-09-11_best_ori_42.pth' \
                    --pretrain_ckpt '/home/cyf/PCF-Grasp/pcfgrasp_method/checkpoints/pretrain/05-08-22_best_pre_292.pth' 
