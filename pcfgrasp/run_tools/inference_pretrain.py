import torch
import numpy as np
from model.pcn import PCN, l1_cd_metric
from utils.data import vis_pc
from torch.utils.data import DataLoader
from run_utils.logger import print_log, get_logger

def resume_model(base_model, logger = None):

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    state_dict = torch.load('/home/cyf/6d_grasp/completion_method/checkpoints/model2022-05-10-15:34:03-0.pth', map_location=map_location)
    print_log(state_dict.keys(), logger=logger)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    base_model.load_state_dict(base_ckpt, False)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    # best_metrics = state_dict['best_metrics']
    # if not isinstance(best_metrics, dict):
    #     best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    # print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch


def inference_completion(args, data_set):
    """
    data: pc {1 N 3}
    """
    print(len(data_set))
    train_dataloader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    # logger = get_logger('test')

    #gpu
    device = args.device

    model = PCN(num_dense=16384, split='pretrain').to(device)
    model.load_state_dict(torch.load(args.pretrain_ckpt), strict=False)

    model.eval()
    metrics = 0.
    with torch.no_grad():
        
        for _ ,( batch_data, gt_data) in enumerate(train_dataloader):
            # print(gt_data.shape)

            # vis_pc(torch.cat([batch_data[..., :3].squeeze(),gt_data[..., :3].squeeze()], dim=0).numpy())
            vis_pc(batch_data.squeeze().numpy())
            # vis_pc(gt_data.squeeze().numpy())

            # vis_pc(gt_data[:, :3].squeeze().numpy())

            pc = batch_data.cuda().float()
            # print(pc.shape)

            coarse, dense = model(pc)

            metrics += l1_cd_metric(coarse, gt_data)
            # print(coarse.shape)
            # vis_pc(coarse.cpu().numpy().squeeze())
            # vis_pc(np.concatenate([coarse[0, :, :].cpu().squeeze().numpy(), batch_data.cpu().squeeze().numpy()], axis=0))
        # print(metrics)

if __name__ == '__main__':

    inference_completion()
