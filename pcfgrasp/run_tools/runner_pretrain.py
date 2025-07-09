from xml.dom.minicompat import NodeList
import torch
import os
from tqdm import tqdm
import time
from run_utils.logger import get_logger, print_log
from model.pcn import PCN, cd_loss, l1_cd_metric
from torch.utils.data import DataLoader

# EMD = EarthMoverDistance()

def pretrain(args, global_config, train_dataset, test_dataset):
    """
    args: argparser {object}
    global_config: global_parameter{dict}
    train_dataset: {object subclass of torch.utils.data.dataset}
    test_dataset: {object subclass of torch.utils.data.dataset}
    """
    logger = get_logger(args.log_name)
    #dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=global_config['OPTIMIZER']['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=global_config['OPTIMIZER']['batch_size'], shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    #gpu指定
    device = args.device

    model = PCN(num_dense=16384, split='pretrain').cuda()
    # loss_cal = PCGLoss(global_config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)#, betas=(0.9, 0.999)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=global_config['OPTIMIZER']['decay_rate'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    start_epoch = 0
    max_epoch = global_config['OPTIMIZER']['max_epoch']
    # print(max_epoch)

    ######model_load
    if args.ckpt_dir is not None:
        # checkpoint = torch.load(args.ckpt_dir)
        model.load_state_dict(torch.load(args.ckpt_dir))
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch']
  
    ######loss_log
    tmp_loss = 1e6
    step = len(train_dataloader) // 1

    train_step, val_step = 0, 0
        # hyperparameter alpha

    model.zero_grad()
    ##############train
    for epoch in range(start_epoch, max_epoch):
        print_log(f'-----------------Epoch {epoch} Training-----------------', logger=logger)
        if train_step < 10000:
            alpha = 0.01
        elif train_step < 20000:
            alpha = 0.1
        elif epoch < 50000:
            alpha = 0.5
        else:
            alpha = 1.0

        model.train()

        for batch_idx, (points, target) in enumerate(train_dataloader):
            # print(points.shape)
            start_time = time.time()
            # print(points.shape, gt_points.shape)

            points = points.cuda()
            target = target.cuda()

            optimizer.zero_grad()

            coarse, _ = model(points)
            # print(coarse.shape, refine.shape)

            coarse_loss = cd_loss(coarse, target)
            # fine_loss = cd_loss(refine, target)

            total_loss = coarse_loss #+ alpha * fine_loss

            total_loss.backward()
            optimizer.step()

            batch_time = time.time()-start_time

            # if (batch_idx+1) % step == 0:
            #     print_log('Epoch [{}] [Batch {}/{} BatchTime:{:.4f} Coarse_loss:{:.4f} Fine_loss:{:.4f}'.format(
            #                 epoch, batch_idx+1, len(train_dataloader), batch_time, coarse_loss.detach()*1000, fine_loss.detach()*1000), logger=logger)
            if (batch_idx+1) % step == 0:
                print_log('Epoch [{}] [Batch {}/{} BatchTime:{:.4f} Coarse_loss:{:.4f}'.format(
                            epoch, batch_idx+1, len(train_dataloader), batch_time, coarse_loss.detach()*1000), logger=logger)
            train_step += 1
            del points, target, coarse, coarse_loss, total_loss

        lr_scheduler.step()
        total_coarse = 0.
        total_fine = 0.

        eval_time = time.time()
        model.eval()

        with torch.no_grad():
            print_log(f'-----------------Epoch {epoch} Evaluation-----------------', logger=logger)
            for batch_idx, (points, target) in enumerate(test_dataloader):
                points = points.cuda()
                target = target.cuda()

                coarse, _ = model(points)

                total_coarse += l1_cd_metric(coarse, target).detach()
                # total_fine += l1_cd_metric(refine, target).detach()

        
            average_coarese = total_coarse / len(test_dataset)
            # average_fine = total_fine / len(test_dataset)

            eval_time_total = time.time() - eval_time
            # print_log("Epoch [{}] Evaluation EvalTime:{:.4f} Average_Coarse_Loss:{:.4f}  Average_Fine_Loss:{:.4f}".format(
            #             epoch, eval_time_total, 1e3 * average_coarese, 1e3 * average_fine), logger=logger)
            print_log("Epoch [{}] Evaluation EvalTime:{:.4f} Average_Coarse_Loss:{:.4f}".format(
                        epoch, eval_time_total, 1e3 * average_coarese), logger=logger)
            total_eval = average_coarese# + average_fine
            print_log("current_loss:{:.6f}".format(1e3 * total_eval), logger=logger)

            save_path = os.path.join(args.output_path, 'pretrain', '{0}_best_pre_{1}.pth'.format(time.strftime("%m-%d-%H"), epoch))

            if total_eval<tmp_loss :

                # print('Saving at %s' % save_path)
                torch.save(model.state_dict(), save_path)
                print_log('Model Saved in file: %s' % save_path, logger=logger)
                #更新损失   
                tmp_loss = total_eval

            print_log("best_loss:{:.6f}".format(1e3 * tmp_loss), logger=logger)
            del total_eval, coarse, points, target
