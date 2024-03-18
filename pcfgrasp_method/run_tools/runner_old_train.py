import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tools'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'run_utils'))

from run_utils.parser import get_args
from tqdm import tqdm
import time
import numpy as np
from run_utils.logger import get_logger, print_log
from model.cgnet import CGNet, CGLoss
from torch.utils.data import DataLoader
from dataloader.train_dataloader import ContactDataset
from utils.grasp_utils import compute_labels
from utils.data import PointCloudReader
from run_utils.logger import get_root_logger
from run_utils.config import load_config
from utils.load_data import load_scene_contacts

def ori_train(args, global_config, train_dataset, test_dataset):
    """
    args: argparser {object}
    global_config: global_parameter{dict}
    train_dataset: {object subclass of torch.utils.data.dataset}
    test_dataset: {object subclass of torch.utils.data.dataset}
    """
    logger = get_logger(args.log_name)
    #dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=global_config['OPTIMIZER']['batch_size'], shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=global_config['OPTIMIZER']['batch_size'], shuffle=False, num_workers=0, pin_memory=True, drop_last=True)                               
    #gpu指定

    model = CGNet(global_config).cuda()#.to(device)
    loss_cal = CGLoss(global_config).cuda()#.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=global_config['OPTIMIZER']['decay_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    start_epoch = 0
    max_epoch = global_config['OPTIMIZER']['max_epoch']
   
    ######loss_log
    loss_log = np.zeros((10,7))
    # metrics = None
    tmp_loss = 100000


    ##############train
    for epoch in range(start_epoch, max_epoch):
        model.train()

        print_log(f'-----------------Epoch {epoch} Training-----------------', logger=logger)
        for batch_idx, (points, cam_poses, labels_dict) in enumerate(tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9)):
            points, cam_poses = points.cuda(), cam_poses.cuda()
            
            srart_time = time.time()

            optimizer.zero_grad()

            end_points = model(points)

            tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx= \
                labels_dict['tf_pos_contact_points_idx'], labels_dict['tf_pos_contact_dirs_idx'], labels_dict['tf_pos_contact_approaches_idx'], labels_dict['tf_pos_finger_diffs_idx']

            tf_pos_contact_points_idx = tf_pos_contact_points_idx.cuda()
            tf_pos_contact_dirs_idx = tf_pos_contact_dirs_idx.cuda()
            tf_pos_contact_approaches_idx = tf_pos_contact_approaches_idx.cuda()
            tf_pos_finger_diffs_idx = tf_pos_finger_diffs_idx.cuda()

            dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc = \
                    compute_labels(args, tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx, end_points['pred_points'], cam_poses, global_config)

            total_loss, loss_dict = loss_cal(end_points, dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc)

            total_loss.backward()
            optimizer.step()

            end_time = time.time()

            dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss = \
                loss_dict['dir_cosine_loss'], loss_dict['score_loss'], loss_dict['offset_loss'], loss_dict['app_cosine_loss'], loss_dict['adds_loss'], loss_dict['adds_loss_gt2pred']

            total_loss = total_loss.detach().cpu().numpy()
            dir_loss = dir_loss.detach().cpu().numpy()
            bin_ce_loss = bin_ce_loss.detach().cpu().numpy()
            offset_loss = offset_loss.detach().cpu().numpy()
            approach_loss = approach_loss.detach().cpu().numpy()
            adds_loss = adds_loss.detach().cpu().numpy()
            adds_gt2pred_loss = adds_gt2pred_loss.detach().cpu().numpy()
            # print(batch_idx, offset_loss)
            loss_log[batch_idx%10,:] = total_loss, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss
            #total loss: 6.330505     dir loss: 1.000000      ce loss: 1.320836       off loss: 0.574544      app loss: 1.000000 adds loss: 0.443512          adds_gt2pred loss: 0.000000

            if (batch_idx+1)%10 == 0:
                f = tuple(np.mean(loss_log, axis=0)) + ((end_time - srart_time) / 1., )
                print_log('total loss: %f \t dir loss: %f \t ce loss: %f \t off loss: %f \t app loss: %f adds loss: %f \t adds_gt2pred loss: %f \t batch time: %f' % f, logger=logger)

        lr_scheduler.step()

        eval_time = time.time()
        model.eval()
        eval_total_loss = 0.
        with torch.no_grad():
            print_log(f'-----------------Epoch {epoch} Evaluation-----------------', logger=logger)
            for batch_idx, (points, cam_poses, labels_dict) in enumerate(tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9)):
                points, cam_poses = points.cuda(), cam_poses.cuda()

                end_points = model(points)

                tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx= \
                        labels_dict['tf_pos_contact_points_idx'], labels_dict['tf_pos_contact_dirs_idx'], labels_dict['tf_pos_contact_approaches_idx'], labels_dict['tf_pos_finger_diffs_idx']

                tf_pos_contact_points_idx = tf_pos_contact_points_idx.cuda()
                tf_pos_contact_dirs_idx = tf_pos_contact_dirs_idx.cuda()
                tf_pos_contact_approaches_idx = tf_pos_contact_approaches_idx.cuda()
                tf_pos_finger_diffs_idx = tf_pos_finger_diffs_idx.cuda()

                dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc = \
                    compute_labels(args, tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx, end_points['pred_points'], cam_poses, global_config)

                total_loss, loss_dict = loss_cal(end_points, dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc)


                dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss = \
                    loss_dict['dir_cosine_loss'], loss_dict['score_loss'], loss_dict['offset_loss'], loss_dict['app_cosine_loss'], loss_dict['adds_loss'], loss_dict['adds_loss_gt2pred']

                total_loss = total_loss.detach().cpu().numpy()
                dir_loss = dir_loss.detach().cpu().numpy()
                bin_ce_loss = bin_ce_loss.detach().cpu().numpy()
                offset_loss = offset_loss.detach().cpu().numpy()
                approach_loss = approach_loss.detach().cpu().numpy()
                adds_loss = adds_loss.detach().cpu().numpy()
                adds_gt2pred_loss = adds_gt2pred_loss.detach().cpu().numpy()
                # print(batch_idx, offset_loss)
                eval_total_loss += total_loss
                loss_log[batch_idx%10,:] = total_loss, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss
            f = tuple(np.mean(loss_log, axis=0))+ ((time.time() - eval_time)/ 1.,)
            print_log('total loss: %f \t dir loss: %f \t ce loss: %f \t off loss: %f \t app loss: %f adds loss: %f \t adds_gt2pred loss: %f \t eval time: %f' % f, logger=logger)
        
        save_path = os.path.join(str(args.output_path), 'train', '{0}_best_ori_{1}.pth'.format(time.strftime("%m-%d-%H-%M"), epoch))
        avg_eval_loss = eval_total_loss / len(train_dataloader)
        print_log("current_loss:{:.6f}".format(avg_eval_loss), logger=logger)

        if avg_eval_loss<tmp_loss :
                        # 更新损失   
            tmp_loss = avg_eval_loss
            print('Saving at %s' % save_path)
            torch.save(model.state_dict(), save_path)
            print_log('Model Saved in file: %s' % save_path, logger=logger)

        print_log("tmp_loss:{:.6f}".format(tmp_loss), logger=logger)


def main():

    args = get_args()

    config_dir = args.config

    global_config = load_config(config_dir, batch_size=args.batch_size, max_epoch=args.max_epoch, 
                                          data_path= args.data_path, arg_configs=args.arg_configs, save=True)
    # print(global_config)
    # args.use_gpu = torch.cuda.is_available()
    # if args.use_gpu:
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

    train_dataset = ContactDataset(global_config, pcreader, contact_infos, split='train', logger=logger)
    test_dataset = ContactDataset(global_config, pcreader, contact_infos, split='test', logger=logger)
    print_log('=======================start_ori_train=============================')
    ori_train(args, global_config, train_dataset, test_dataset)

if __name__ == '__main__':

    main()