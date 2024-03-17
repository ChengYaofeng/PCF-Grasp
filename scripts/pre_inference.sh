#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=egl
export QT_GRAPHICSSYSTEM=native

python3 ./inference.py --data_path /home/cyf/PCF-Grasp/acronym \
                    --pretrain_vis=True \
                    --exp_name='pretrain_vis' \
                    --pretrain_ckpt '/home/cyf/PCF-Grasp/pcfgrasp_method/checkpoints/pretrain/03-05-14_best_pre_598.pth'
