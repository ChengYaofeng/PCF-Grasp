#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=egl

python3 ./main.py --config ~/PCF-Grasp/pcfgrasp_method/ \
                    --data_path ~/PCF-Grasp/acronym \
                    --batch_size 4 \
                    --pretrain_ckpt '/home/cyf/PCF-Grasp/pcfgrasp_method/checkpoints/pretrain/03-03-12_best_pre_4.pth'
