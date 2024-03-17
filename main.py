import os
import sys
import torch
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #~/pcf_grasp/pcfgrasp_method
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'run_tools'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'run_utils'))

from run_utils.config import load_config
from run_utils.parser import get_args
from run_utils.logger import get_root_logger, print_log
from utils.data import PointCloudReader
from utils.load_data import load_scene_contacts
from dataloader.train_dataloader import ContactDataset
from dataloader.pretrain_dataloader import CompleteDataset

from run_tools.runner_train import train
from run_tools.runner_pretrain import pretrain

#gpu-'egl' & cpu-'osmesa' 
# os.environ["PYOPENGL_PLATFORM"] = "egl"

def main():
    args = get_args()

    config_dir = args.config

    global_config = load_config(config_dir, batch_size=args.batch_size, max_epoch=args.max_epoch, 
                                          data_path= args.data_path, arg_configs=args.arg_configs, save=True)
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    #logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    #dataset
    contact_infos = load_scene_contacts(global_config['DATA']['data_path'], scene_contacts_path = global_config['DATA']['scene_contacts_path'])

    num_train_samples = len(contact_infos) - global_config['DATA']['num_test_scenes']
    num_test_samples = global_config['DATA']['num_test_scenes']
    print_log('using {} meshes'.format(num_train_samples + num_test_samples), logger=logger)

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
    if args.pretrain is not True:
        train_dataset = ContactDataset(global_config, pcreader, contact_infos, split='train', logger=logger)
        test_dataset = ContactDataset(global_config, pcreader, contact_infos, split='test', logger=logger)

        print_log('========================Start Train======================', logger=logger)
        train(args, global_config, train_dataset, test_dataset)
    else:
        train_dataset = CompleteDataset(global_config, pcreader, contact_infos, split='train')
        test_dataset = CompleteDataset(global_config, pcreader, contact_infos, split='test')
        print_log('======================Start Pretrain=====================', logger=logger)
        pretrain(args, global_config, train_dataset, train_dataset)

if __name__ == '__main__':
    main()