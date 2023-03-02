import torch
import os
import time
from tqdm import tqdm
import numpy as np
from run_utils.logger import get_logger, print_log
from model.pcfnet import PCFNet, PCFLoss
from torch.utils.data import DataLoader
from utils.grasp_utils import compute_labels

def train(args, global_config, train_dataset, test_dataset):
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
    if args.use_gpu:
        model = PCFNet(args, global_config).cuda()
        loss_cal = PCFLoss(global_config, gpu=True).cuda()
    else:
        model = PCFNet(args, global_config)
        loss_cal = PCFLoss(global_config, gpu=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    max_epoch = global_config['OPTIMIZER']['max_epoch']
    start_epoch = global_config['OPTIMIZER']['start_epoch']
  
    ######loss_log
    loss_log = np.zeros((10,7))
    tmp_loss = 1e6

    ##############train
    for epoch in range(start_epoch, max_epoch):
        model.train()
        print_log(f'-----------------Epoch {epoch} Training-----------------', logger=logger)
        for batch_idx, (points, cam_poses, labels_dict) in enumerate(tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9)):
            
            tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx= \
                labels_dict['tf_pos_contact_points_idx'], labels_dict['tf_pos_contact_dirs_idx'], labels_dict['tf_pos_contact_approaches_idx'], labels_dict['tf_pos_finger_diffs_idx']

            start_time = time.time()

            if args.use_gpu:
                points, cam_poses = points.cuda(), cam_poses.cuda()

                tf_pos_contact_approaches_idx = tf_pos_contact_approaches_idx.cuda()
                tf_pos_contact_dirs_idx = tf_pos_contact_dirs_idx.cuda()
                tf_pos_contact_approaches_idx = tf_pos_contact_approaches_idx.cuda()
                tf_pos_finger_diffs_idx = tf_pos_finger_diffs_idx.cuda()

            optimizer.zero_grad()

            end_points = model(points)

            dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc = \
                    compute_labels(tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx, end_points['pred_points'], cam_poses, global_config)

            total_loss, loss_dict = loss_cal(end_points, dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc)

            total_loss.backward()
            optimizer.step()

            dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss = \
                loss_dict['dir_cosine_loss'], loss_dict['score_loss'], loss_dict['offset_loss'], loss_dict['app_cosine_loss'], loss_dict['adds_loss'], loss_dict['adds_loss_gt2pred']

            total_loss = total_loss.detach().cpu().numpy()
            dir_loss = dir_loss.detach().cpu().numpy()
            bin_ce_loss = bin_ce_loss.detach().cpu().numpy()
            offset_loss = offset_loss.detach().cpu().numpy()
            approach_loss = approach_loss.detach().cpu().numpy()
            adds_loss = adds_loss.detach().cpu().numpy()
            adds_gt2pred_loss = adds_gt2pred_loss.detach().numpy()

            loss_log[batch_idx%10,:] = total_loss, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss

            if (batch_idx+1)%10 == 0:
                f = tuple(np.mean(loss_log, axis=0)) + ((time.time() - start_time) / 1., )
                print_log('total loss: %f \t dir loss: %f \t ce loss: %f \t off loss: %f \t app loss: %f adds loss: %f \t adds_gt2pred loss: %f \t batch time: %f' % f, logger=logger)

        lr_scheduler.step()
        eval_time = time.time()
        model.eval()
        eval_total_loss = 0.
        with torch.no_grad():
            print_log(f'-----------------Epoch {epoch} Evaluation-----------------', logger=logger)
            for batch_idx, (points, cam_poses, labels_dict) in enumerate(tqdm(test_dataloader, total=len(test_dataloader), smoothing=0.9)):

                tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx= \
                        labels_dict['tf_pos_contact_points_idx'], labels_dict['tf_pos_contact_dirs_idx'], labels_dict['tf_pos_contact_approaches_idx'], labels_dict['tf_pos_finger_diffs_idx']

                if args.use_gpu:
                    points, cam_poses = points.cuda(), cam_poses.cuda()

                    tf_pos_contact_approaches_idx = tf_pos_contact_approaches_idx.cuda()
                    tf_pos_contact_dirs_idx = tf_pos_contact_dirs_idx.cuda()
                    tf_pos_contact_approaches_idx = tf_pos_contact_approaches_idx.cuda()
                    tf_pos_finger_diffs_idx = tf_pos_finger_diffs_idx.cuda()  

                end_points = model(points)
                
                dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc = \
                    compute_labels(tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx, end_points['pred_points'], cam_poses, global_config)

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

                eval_total_loss += total_loss
                loss_log[batch_idx%10,:] = eval_total_loss, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss
            f = tuple(np.mean(loss_log, axis=0))+ ((time.time() - eval_time)/ 1.,)
            print_log('total loss: %f \t dir loss: %f \t ce loss: %f \t off loss: %f \t app loss: %f adds loss: %f \t adds_gt2pred loss: %f \t eval time: %f' % f, logger=logger)
        
        save_path = os.path.join(str(args.output_path), 'train', '{0}_best_ori_{1}.pth'.format(time.strftime("%m-%d-%H"), epoch))
        avg_eval_loss = eval_total_loss / len(train_dataloader)
        print_log("current_loss:{:.6f}".format(avg_eval_loss), logger=logger)
        if avg_eval_loss<tmp_loss :
            print('Saving at %s' % save_path)
            torch.save(model.state_dict(), save_path)
            print_log('Model Saved in file: %s' % save_path, logger=logger)
            tmp_loss = avg_eval_loss
        print_log("tmp_loss:{:.6f}".format(tmp_loss), logger=logger)
