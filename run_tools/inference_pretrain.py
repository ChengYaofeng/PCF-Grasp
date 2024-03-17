import torch
from model.pcn import PCN, l1_cd_metric
from utils.data import vis_pc
from torch.utils.data import DataLoader


def inference_completion(args, data_set):
    """
    data: pc {1 N 3}
    """
    train_dataloader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    #gpu
    if args.use_gpu:
        model = PCN(num_dense=16384, split='pretrain').cuda()
    else:
        model = PCN(num_dense=16384, split='pretrain')

    model.load_state_dict(torch.load(args.pretrain_ckpt), strict=False)

    model.eval()
    metrics = 0.
    with torch.no_grad():
        
        for _ ,(batch_data, gt_data) in enumerate(train_dataloader):

            # vis_pc(batch_data.squeeze().numpy())
            if args.use_gpu:
                batch_data = batch_data.cuda().float()

            coarse = model(batch_data)

            metrics += l1_cd_metric(coarse, gt_data)
            # vis_pc(coarse.cpu().numpy().squeeze())

if __name__ == '__main__':

    inference_completion()
