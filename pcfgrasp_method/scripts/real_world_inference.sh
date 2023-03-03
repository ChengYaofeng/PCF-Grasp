#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=osmesa

python ./run_tools/real_world_inference.py --pretrain_ckpt 'path to pcn pth file' \
                    --ckpt_dir 'path to pcfgrasp pth file' \