#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=osmesa

python ./inference.py --ckpt_dir 'pcfgrasp model path' \
                    --pretrain_ckpt 'pcn model path' 
