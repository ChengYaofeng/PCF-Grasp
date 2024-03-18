#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=egl
# export QT_GRAPHICSSYSTEM=native


python3 ./main.py --config ~/PCF-Grasp/pcfgrasp_method/ \
                    --data_path ~/PCF-Grasp/acronym \
                    --batch_size 4 \
                    --pretrain_ckpt '/home/cyf/PCF-Grasp/pcfgrasp_method/checkpoints/pretrain/05-08-22_best_pre_292.pth' \
                    --exp_name 'train_1024_nofil'
