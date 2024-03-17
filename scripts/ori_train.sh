#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=osmesa

python3 ./run_tools/runner_old_train.py --config ~/PCF-Grasp/pcfgrasp_method/ \
                    --data_path ~/PCF-Grasp/acronym \
                    --log_name ori_train \
                    --exp_name ori_train\
                    --batch_size 4 \