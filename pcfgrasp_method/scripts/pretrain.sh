#!/usr/bin/env bash

set -x
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=0

python  main.py --exp_name='pretrain' \
                --config ~/pcf_grasp/pcfgrasp_method/ \
                --data_path ~/pcf_grasp/acronym  \
                --pretrain=True \
                --batch_size 4 \
                --ckpt_dir 'path of pcn, if you want to train it based on pretrained pcn model'

