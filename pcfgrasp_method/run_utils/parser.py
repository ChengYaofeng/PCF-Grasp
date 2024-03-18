import os
from pathlib import Path
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='~/PCF-Grasp/pcfgrasp_method/', help='Checkpoint dir')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='existing model check')
    parser.add_argument('--pretrain_ckpt', type=str)
    parser.add_argument('--data_path', type=str, default=None, help='Grasp data root dir')
    parser.add_argument('--output_path', type=str, default='/home/cyf/PCF-Grasp/pcfgrasp_method/checkpoints', help='expariment name')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')

    parser.add_argument('--max_epoch', type=int, default=None, help='Epochs to run')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=None, help='Batch Size during training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='cuda accelerate')
    parser.add_argument('--device', type=str, default='cuda:0', help='logger name')
    
    parser.add_argument('--log_name', type=str, default='train_log', help='logger name')
    parser.add_argument('--exp_name', type=str, default='train', help='expariment name')

    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain points encoder')

    args = parser.parse_args()
    
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args
    # args.experiment_path

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
