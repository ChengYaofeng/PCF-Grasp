#!/usr/bin/env bash

set -x
set -e

export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0

python  main.py --exp_name='pretrain' \
                --config ~/PCF-Grasp/pcfgrasp_method/ \
                --data_path ~/PCF-Grasp/acronym  \
                --pretrain=True \
                --batch_size 32 \
                # --ckpt_dir 'path of pcn, if you want to train it based on pretrained pcn model'

