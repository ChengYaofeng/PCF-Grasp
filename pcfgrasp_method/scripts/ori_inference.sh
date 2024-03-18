#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=egl

python ./run_tools/inference_ori_train.py --ckpt_dir /home/cyf/PCF-Grasp/pcfgrasp_method/checkpoints/train/10-03-17-28_best_ori_26.pth \
                                    --ori_inference True