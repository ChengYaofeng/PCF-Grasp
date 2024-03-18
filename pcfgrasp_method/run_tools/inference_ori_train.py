import torch
import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #/home/cyf/6d_grasp/completion_method
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tools'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'run_utils'))

from dataloader.train_dataloader import ContactDataset
from torch.utils.data import DataLoader
from utils.data import PointCloudReader
from utils.load_data import load_scene_contacts
from utils.grasp_estimator import GraspEstimatior
from run_utils.config import load_config
from utils.visual_grasp import visualize_grasps

def main(args):
    global_config = load_config(args.config_dir, batch_size=1, max_epoch=1, 
                                          data_path= args.data_path, arg_configs=args.arg_configs, save=True)

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

    data_set = ContactDataset(global_config, pcreader, contact_infos, split='eval')
    ori_train_inference(args, global_config, data_set)


def ori_train_inference(args, global_config, data_set=None, K=None, z_range=[0.2, 1.8] ,forward_passes=1):
    
    grasp_estimatior = GraspEstimatior(global_config)

    eval_dataloader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0)

    for _, pc in enumerate(eval_dataloader):
        
        pc = pc.squeeze().numpy()
        # print(pc.shape)
        
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimatior.predict_scene_grasps(pc, args, forward_passes=forward_passes)
        # visualize_grasps(pc, pred_grasps_cam, scores, plot_opencv_cam=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_vis', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrain_ckpt', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--input_path', type=str, default=None, help='train_inference picture pcd scene or object waiting for grasp generation')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    parser.add_argument('--ori_inference', type=bool, default=False)
    
    parser.add_argument('--log_name', type=str, default='inference_log', help='logger name')
    parser.add_argument('--exp_name', type=str, default='vis', help='expariment name')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--config_dir', type=str, default='/home/cyf/6d_grasp/completion_method/')

    args = parser.parse_args()

    main(args)