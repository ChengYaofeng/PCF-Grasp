#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=osmesa

python3 ./tools/runner_ori_train.py --config ~/6d_grasp/completion_method/ \
                    --data_path ~/6d_grasp/acronym \
                    --log_name ori_train \
                    --exp_name ori_train