#!/bin/bash

set -x
set -e

# export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=egl
export QT_GRAPHICSSYSTEM=native


python3 ./inference.py --data_path ~/6d_grasp/acronym \
                    --pretrain_vis=True \
                    --exp_name='pretrain_vis' \
                    --pretrain_ckpt 'path to pcn pth file'
