#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=osmesa

python ./tools/inference_ori_train.py --ckpt_dir /home/cyf/6d_grasp/completion_method/checkpoints/specific_model/best_ori/06-07-18-14_best_ori_140.pth \
                                    --ori_inference True