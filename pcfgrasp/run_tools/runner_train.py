import torch
import os
import time
from tqdm import tqdm
import numpy as np
from run_utils.logger import get_logger, print_log
from model.pcn_cgnet import PCGNet, PCGLoss
from torch.utils.data import DataLoader
from utils.grasp_utils import compute_labels
from model.pcn import cd_loss, l1_cd_metric
# from model.pointr import fps

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
    #gpu指定

    model = PCGNet(args, global_config)#.to(device)
    loss_cal = PCGLoss(global_config)#.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=global_config['OPTIMIZER']['decay_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    start_epoch = 0
  
    ######loss_log
    loss_log = np.zeros((10,7))
    # metrics = None
    tmp_loss = 1e6


    ##############train
    for epoch in range(start_epoch, 200):
        model.train()
        # model.zero_grad()
        print_log(f'-----------------Epoch {epoch} Training-----------------', logger=logger)
        for batch_idx, (points, cam_poses, labels_dict) in enumerate(tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9)):
            
            start_time = time.time()

            tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx= \
                labels_dict['tf_pos_contact_points_idx'], labels_dict['tf_pos_contact_dirs_idx'], labels_dict['tf_pos_contact_approaches_idx'], labels_dict['tf_pos_finger_diffs_idx']

            optimizer.zero_grad()

            end_points = model(points)

            dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc = \
                    compute_labels(tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx, end_points['pred_points'], cam_poses, global_config)

            total_loss, loss_dict = loss_cal(end_points, dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc)

            # coarse_loss = cd_loss(end_points['coarse'].cuda(), labels_dict['target'].cuda())
            # refine_loss = cd_loss(end_points['refine'].cuda(), labels_dict['target'].cuda())
            # total_loss = total_loss + coarse_loss.cpu() + refine_loss.cpu()

            total_loss.backward()
            optimizer.step()

            end_time = time.time()

            dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss = \
                loss_dict['dir_cosine_loss'], loss_dict['score_loss'], loss_dict['offset_loss'], loss_dict['app_cosine_loss'], loss_dict['adds_loss'], loss_dict['adds_loss_gt2pred']

            total_loss = total_loss.detach().numpy()
            dir_loss = dir_loss.detach().numpy()
            bin_ce_loss = bin_ce_loss.detach().numpy()
            offset_loss = offset_loss.detach().numpy()
            approach_loss = approach_loss.detach().numpy()
            adds_loss = adds_loss.detach().numpy()
            adds_gt2pred_loss = adds_gt2pred_loss.detach().numpy()

            loss_log[batch_idx%10,:] = total_loss, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss
            #total loss: 6.330505     dir loss: 1.000000      ce loss: 1.320836       off loss: 0.574544      app loss: 1.000000 adds loss: 0.443512          adds_gt2pred loss: 0.000000

            if (batch_idx+1)%10 == 0:
                f = tuple(np.mean(loss_log, axis=0)) + ((end_time - start_time) / 1., )
                print_log('total loss: %f \t dir loss: %f \t ce loss: %f \t off loss: %f \t app loss: %f adds loss: %f \t adds_gt2pred loss: %f \t batch time: %f' % f, logger=logger)
                # print_log(f'c_loss:{coarse_loss}, r_loss:{refine_loss}', logger=logger)

        lr_scheduler.step()
        eval_time = time.time()
        model.eval()
        eval_total_loss = 0.
        with torch.no_grad():
            print_log(f'-----------------Epoch {epoch} Evaluation-----------------', logger=logger)
            for batch_idx, (points, cam_poses, labels_dict) in enumerate(tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9)):

                tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx= \
                        labels_dict['tf_pos_contact_points_idx'], labels_dict['tf_pos_contact_dirs_idx'], labels_dict['tf_pos_contact_approaches_idx'], labels_dict['tf_pos_finger_diffs_idx']

                end_points = model(points)
                
                dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc = \
                    compute_labels(tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx, end_points['pred_points'], cam_poses, global_config)

                total_loss, loss_dict = loss_cal(end_points, dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc)


                dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss = \
                    loss_dict['dir_cosine_loss'], loss_dict['score_loss'], loss_dict['offset_loss'], loss_dict['app_cosine_loss'], loss_dict['adds_loss'], loss_dict['adds_loss_gt2pred']

                # coarse_loss = l1_cd_metric(end_points['coarse'].cuda(), labels_dict['target'].cuda()).detach()
                # refine_loss = l1_cd_metric(end_points['refine'].cuda(), labels_dict['target'].cuda()).detach()
                # total_loss = total_loss + coarse_loss.cpu() + refine_loss.cpu()

                total_loss = total_loss.detach().numpy()
                dir_loss = dir_loss.detach().numpy()
                bin_ce_loss = bin_ce_loss.detach().numpy()
                offset_loss = offset_loss.detach().numpy()
                approach_loss = approach_loss.detach().numpy()
                adds_loss = adds_loss.detach().numpy()
                adds_gt2pred_loss = adds_gt2pred_loss.detach().numpy()

                eval_total_loss += total_loss
                loss_log[batch_idx%10,:] = eval_total_loss, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss
            f = tuple(np.mean(loss_log, axis=0))+ ((time.time() - eval_time)/ 1.,)
            print_log('total loss: %f \t dir loss: %f \t ce loss: %f \t off loss: %f \t app loss: %f adds loss: %f \t adds_gt2pred loss: %f \t eval time: %f' % f, logger=logger)
            # print_log(f'c_loss:{coarse_loss}, r_loss:{refine_loss}', logger=logger)
        
        save_path = os.path.join(str(args.output_path), 'train', '{0}_best_ori_{1}.pth'.format(time.strftime("%m-%d-%H"), epoch))
        avg_eval_loss = eval_total_loss / len(train_dataloader)
        print_log("current_loss:{:.6f}".format(avg_eval_loss), logger=logger)
        if avg_eval_loss<tmp_loss :
            
            print('Saving at %s' % save_path)
            torch.save(model.state_dict(), save_path)
            print_log('Model Saved in file: %s' % save_path, logger=logger)
            #更新损失   
            tmp_loss = avg_eval_loss
        print_log("tmp_loss:{:.6f}".format(tmp_loss), logger=logger)
