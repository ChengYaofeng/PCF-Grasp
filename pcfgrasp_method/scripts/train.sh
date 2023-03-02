#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=osmesa

python3 ./main.py --config ~/6d_grasp/completion_method/ \
                    --data_path ~/6d_grasp/acronym \
                    --pretrain_ckpt 'path to pcn pth model'
