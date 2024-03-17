import torch
import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #~/pcf_grasp/pcfgrasp_method

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'run_tools'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'run_utils'))

from dataloader.train_dataloader import ContactDataset
from dataloader.pretrain_dataloader import CompleteDataset
from utils.data import PointCloudReader
from utils.load_data import load_scene_contacts
from run_utils.logger import get_root_logger, print_log, get_logger
from run_tools.inference_pretrain import inference_completion
from run_tools.inference_train import train_inference
from run_utils.config import load_config


def main(args):

    global_config = load_config(args.config_dir, batch_size=1, max_epoch=1, data_path= args.data_path, arg_configs=args.arg_configs, save=True)
    
    # args.use_gpu = torch.cuda.is_available()
    args.use_gpu = False
    
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    #dataset
    contact_infos = load_scene_contacts(global_config['DATA']['data_path'], scene_contacts_path = global_config['DATA']['scene_contacts_path'])

    num_train_samples = len(contact_infos) - global_config['DATA']['num_test_scenes']
    num_test_samples = global_config['DATA']['num_test_scenes']
    print('using {} meshes'.format(num_train_samples + num_test_samples))

    if 'train_and_test' in global_config['DATA'] and global_config['DATA']['train_and_test']:
        num_train_samples = num_train_samples + num_test_samples
        num_test_samples = 0
        print('using train and test data')
    
    pcreader = PointCloudReader(
        root_folder=global_config['DATA']['data_path'],
        batch_size=global_config['OPTIMIZER']['batch_size'],
        estimate_normals=global_config['DATA']['input_normals'],
        raw_num_points=global_config['DATA']['raw_num_points'],
        use_uniform_quaternions=global_config['DATA']['use_uniform_quaternions'],
        scene_obj_scales=[c['obj_scales'] for c in contact_infos],
        scene_obj_paths=[c['obj_paths'] for c in contact_infos],
        scene_obj_transforms=[c['obj_transforms'] for c in contact_infos],
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        use_farthest_point=global_config['DATA']['use_farthest_point'],
        intrinsics=global_config['DATA']['intrinsics'],
        elevation=global_config['DATA']['view_sphere']['elevation'],
        distance_range=global_config['DATA']['view_sphere']['distance_range'],
        pc_augm_config=global_config['DATA']['pc_augm'],
        depth_augm_config=global_config['DATA']['depth_augm']
    )

    #############set mode################ 
    if args.pretrain_vis is True:
        print('=========================start pre_inference========================')
        data_set = CompleteDataset(global_config, pcreader, contact_infos, split='eval')
        inference_completion(args, data_set)

    else:
        print('========================start train_inference=======================')
        data_set = ContactDataset(global_config, pcreader, contact_infos, split='eval')
        train_inference(args, global_config, data_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_vis', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--pretrain_ckpt', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--input_path', default=None, help='train_inference picture pcd scene or object waiting for grasp generation')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    parser.add_argument('--ori_inference', type=bool, default=False)

    parser.add_argument('--log_name', type=str, default='inference_log', help='logger name')
    parser.add_argument('--exp_name', type=str, default='vis', help='expariment name')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--config_dir', type=str, default='/home/cyf/PCF-Grasp/pcfgrasp_method/')

    #mask-rcnn
    parser.add_argument('--config_file', metavar="FILE", default='~/pcf_grasp/pcfgrasp_method/mask_rcnn/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl'])
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help="Minimum score for instance predictions to be shown",)

    args = parser.parse_args()

    main(args)