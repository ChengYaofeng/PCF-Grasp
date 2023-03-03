import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mesh_utils
from utils.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from utils.grasp_utils import build_6d_grasp, get_bin_vals
from model.pcn import PCN
from utils.pointnet2_utils import index_points, farthest_point_sample, query_ball_point

class PointSplit(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointSplit, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, ori_xyz, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = ori_xyz

        new_points_list = []

        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)

            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            """
            torch.Size([1, 64, 32, 1024])
            torch.Size([1, 128, 64, 1024])
            torch.Size([1, 128, 128, 1024])"""

            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]

            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PCFNet(nn.Module):
    def __init__(self, args, global_config):
        super(PCFNet,self).__init__()

        self.global_config = global_config
        self.model_config = global_config['MODEL']

        self.prtrain_ckpt = args.pretrain_ckpt

        if args.use_gpu is True:
            self.pcn = PCN(split='train').cuda()
        else:
            self.pcn = PCN(split='train')

        self._pretrain_pcn()

        self.split = PointSplit(npoint=1024, radius_list=[0.04, 0.08, 0.16], nsample_list=[64, 64, 128], in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.o_sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.04, 0.08, 0.16], nsample_list=[64, 64, 128], in_channel=64+128+128, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.o_sa3 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.08, 0.16, 0.32], nsample_list=[64, 64, 128], in_channel=128+256+256, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        self.o_sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128+256+256+3, mlp=self.model_config['pointnet_sa_module']['mlp'], group_all=self.model_config['pointnet_sa_module']['group_all'])
        self.o_fp3 = PointNetFeaturePropagation(1664, [256, 256]) 
        self.o_fp2 = PointNetFeaturePropagation(896, [256, 128]) 
        self.o_fp1 = PointNetFeaturePropagation(448, [128, 128, 128]) 

        self.layer1 = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Conv1d(128,3,1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Conv1d(128,3,1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128,1,1)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,10,1)
        )


    def _pretrain_pcn(self):
        self.pcn.load_state_dict(torch.load(self.prtrain_ckpt, map_location='cpu'), strict=True)
        for param in self.pcn.parameters():
            param.requires_grad = False

    def forward(self, points):
        """
        Input:
            points: [b,n,c] {default: b 1024 3} in our default, n=1024, N=2048
        
        Output:
            end_points: {dict}
        """

        data_config = self.global_config['DATA']
        input_normals = data_config['input_normals']
        
        l0_points = None

        self.pcn.eval()
        coarse = self.pcn(points) #B n 3 {B 3 1024}

        fuse = torch.cat((coarse, points), dim=1)  #B N 3 {B 2048 3}

        l0_xyz = fuse[:, :, :3]
        l0_xyz = l0_xyz.permute(0,2,1) # B C N
        ori_xyz = points # B 1024 3
 
        l1_xyz, l1_points = self.split(ori_xyz, l0_xyz, l0_points)
        l2_xyz, l2_points = self.o_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.o_sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.o_sa4(l3_xyz, l3_points)

        l3_points = self.o_fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.o_fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.o_fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        pred_points = l1_xyz.permute(0,2,1) # B n C {B 1024 3}

        grasp_dir_head = self.layer1(l1_points)
        grasp_dir_head_normed = F.normalize(grasp_dir_head, dim=1)

        approach_dir_head = self.layer2(l1_points)
        approach_dir_orthog = approach_dir_head - torch.sum(torch.mul(grasp_dir_head_normed, approach_dir_head), dim=1, keepdim=True) * grasp_dir_head_normed
        approach_dir_head_orthog = F.normalize(approach_dir_orthog, dim=1)

        binary_score_head = self.layer3(l1_points)

        grasp_offset_head = self.layer4(l1_points)

        grasp_dir_head_normed = grasp_dir_head_normed.permute(0,2,1) # B n 3
        approach_dir_head_orthog = approach_dir_head_orthog.permute(0,2,1) # B n 3
        binary_score_head = binary_score_head.permute(0,2,1) # B n 1
        grasp_offset_head = grasp_offset_head.permute(0,2,1) # B n 10

        m = nn.Sigmoid()

        grasp_dir_head = grasp_dir_head_normed
        approach_dir_head = approach_dir_head_orthog

        end_points = {}

        end_points['grasp_dir_head'] = grasp_dir_head_normed
        end_points['approach_dir_head'] = approach_dir_head_orthog
        end_points['binary_score_head'] = binary_score_head
        end_points['binary_score_pred'] = m(binary_score_head)
        end_points['grasp_offset_head'] = grasp_offset_head
        end_points['grasp_offset_pred'] = m(grasp_offset_head) if self.model_config['bin_offsets'] else grasp_offset_head
        end_points['pred_points'] = pred_points
        end_points['coarse'] = coarse.permute(0,2,1)

        return end_points

class PCFLoss(nn.Module):
    def __init__(self, global_config, gpu=True):
        super(PCFLoss, self).__init__()
        self.global_config = global_config
        self.gpu = gpu

    def forward(self, end_points, dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam):
        """
        Input:
            end_points: {dict} predict parameters
            end_points['points']: [B,N,C]
            end_points['grasp_offset_head']: [B, N, C] 
        Output:
            total_loss
        """

        grasp_dir_head = end_points['grasp_dir_head']
        approach_dir_head = end_points['approach_dir_head']
        grasp_offset_head = end_points['grasp_offset_head']
        bin_weights = self.global_config['DATA']['labels']['bin_weights']
        torch_bin_weights = torch.tensor(bin_weights)

        global_config = self.global_config
        min_geom_loss_divisor = torch.tensor(float(global_config['LOSS']['min_geom_loss_divisor'])) if 'min_geom_loss_divisor' in global_config['LOSS'] else torch.tensor(1.)
        pos_grasps_in_view = torch.maximum(torch.sum(grasp_success_labels_pc, dim=1), min_geom_loss_divisor)

        pointclouds_pl = end_points['pred_points']

        ### ADS Gripper PC Loss
        if global_config['MODEL']['bin_offsets']:
            thickness_pred = get_bin_vals(global_config)[torch.argmax(grasp_offset_head, dim=2)]
            thickness_gt = get_bin_vals(global_config)[torch.argmax(offset_labels_pc, dim=2)]
        else:
            thickness_pred = grasp_offset_head[:, :, 0]
            thickness_gt = offset_labels_pc[:, :, 0]

        if self.gpu:
            thickness_pred = thickness_pred.cuda()
            thickness_gt = thickness_gt.cuda()
            torch_bin_weights = torch_bin_weights.cuda()


        pred_grasps = build_6d_grasp(approach_dir_head, grasp_dir_head, pointclouds_pl, thickness_pred, self.gpu)  # b x num_point x 4 x 4
        gt_grasps_proj = build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pointclouds_pl, thickness_gt, self.gpu)  # b x num_point x 4 x 4

        pos_gt_grasps_proj = torch.where(torch.broadcast_to(torch.unsqueeze(torch.unsqueeze(grasp_success_labels_pc.type(torch.bool), dim=2), dim=3),
                            gt_grasps_proj.shape), gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000)

        gripper = mesh_utils.create_gripper('panda')
        gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size'])  # b x 5 x 3
        sym_gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size'],symmetric=True)

        gripper_control_points_homog = torch.cat([gripper_control_points, torch.ones((global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4
        sym_gripper_control_points_homog = torch.cat([sym_gripper_control_points, torch.ones((global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4

        # only use per point pred grasps but not per point gt grasps
        control_points = torch.unsqueeze(gripper_control_points_homog, dim=1).repeat(1, gt_grasps_proj.shape[1], 1, 1)  # b x num_point x 5 x 4
        sym_control_points = torch.unsqueeze(sym_gripper_control_points_homog, dim=1).repeat(1, gt_grasps_proj.shape[1], 1, 1)  # b x num_point x 5 x 4
        if self.gpu:
            control_points = control_points.cuda()
            sym_control_points = sym_control_points.cuda()
        pred_control_points = torch.matmul(control_points, torch.transpose(pred_grasps, dim0=2, dim1=3))[:, :, :, :3]  # b x num_point x 5 x 3
        # pred_points = torch.matmul(control_points, torch.transpose(pred_grasps, dim0=2, dim1=3))

        ### Pred Grasp to GT Grasp ADD-S Loss
        gt_control_points = torch.matmul(control_points, torch.transpose(pos_gt_grasps_proj, dim0=2, dim1=3))[:, :, :, :3]  # b x num_pos_grasp_point x 5 x 3
        sym_gt_control_points = torch.matmul(sym_control_points, torch.transpose(pos_gt_grasps_proj, dim0=2, dim1=3))[:, :, :, :3] # b x num_pos_grasp_point x 5 x 3

        squared_add = torch.sum((torch.unsqueeze(pred_control_points, dim=2) - torch.unsqueeze(gt_control_points, dim=1)) ** 2, dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)
        sym_squared_add = torch.sum((torch.unsqueeze(pred_control_points, dim=2) - torch.unsqueeze(sym_gt_control_points, dim=1)) ** 2, dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)

        # symmetric ADD-S
        neg_squared_adds = -torch.cat([squared_add, sym_squared_add], dim=2)  # b x num_point x 2num_pos_grasp_point
        neg_squared_adds_k = torch.topk(neg_squared_adds, k=1, sorted=False)[0]  # b x num_point
        # If any pos grasp exists
        min_adds = torch.minimum(torch.sum(grasp_success_labels_pc, dim=1, keepdims=True),torch.ones_like(neg_squared_adds_k[:, :, 0])) * torch.sqrt(-neg_squared_adds_k[:, :, 0])  # tf.minimum(tf.sqrt(-neg_squared_adds_k), tf.ones_like(neg_squared_adds_k)) # b x num_point
        adds_loss = torch.mean(end_points['binary_score_pred'][:, :, 0] * min_adds)

        ### GT Grasp to pred Grasp ADD-S Loss
        gt_control_points = torch.matmul(control_points, torch.transpose(gt_grasps_proj, dim0=2, dim1=3))[:, :, :, :3]  # b x num_pos_grasp_point x 5 x 3
        sym_gt_control_points = torch.matmul(sym_control_points, torch.transpose(gt_grasps_proj, dim0=2, dim1=3))[:, :, :, :3]  # b x num_pos_grasp_point x 5 x 3

        neg_squared_adds = -torch.sum((torch.unsqueeze(pred_control_points, dim=1) - torch.unsqueeze(gt_control_points, dim=2)) ** 2, dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)
        neg_squared_adds_sym = -torch.sum((torch.unsqueeze(pred_control_points, dim=1) - torch.unsqueeze(sym_gt_control_points, dim=2)) ** 2,dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)

        neg_squared_adds_k_gt2pred, pred_grasp_idcs = torch.topk(neg_squared_adds, k=1, sorted=False)  # b x num_pos_grasp_point
        neg_squared_adds_k_sym_gt2pred, pred_grasp_sym_idcs = torch.topk(neg_squared_adds_sym, k=1, sorted=False)  # b x num_pos_grasp_point
        pred_grasp_idcs_joined = torch.where(neg_squared_adds_k_gt2pred < neg_squared_adds_k_sym_gt2pred, pred_grasp_sym_idcs, pred_grasp_idcs)


        min_adds_gt2pred = torch.minimum(-neg_squared_adds_k_gt2pred, -neg_squared_adds_k_sym_gt2pred)  # b x num_pos_grasp_point x 1
        masked_min_adds_gt2pred = torch.multiply(min_adds_gt2pred[:, :, 0], grasp_success_labels_pc)
        batch_idcs_x, _ = torch.meshgrid(torch.arange(pred_grasp_idcs_joined.shape[0]), torch.arange(pred_grasp_idcs_joined.shape[1]))
        if self.gpu:
            batch_idcs_x = batch_idcs_x.cuda()
        gather_idcs = torch.stack((batch_idcs_x, pred_grasp_idcs_joined[:, :, 0]), dim=2)
        nearest_pred_grasp_confidence = end_points['binary_score_pred'][:, :, 0][gather_idcs[:, :, 0], gather_idcs[:, :, 1]]

        adds_loss_gt2pred = torch.mean(torch.sum(nearest_pred_grasp_confidence * masked_min_adds_gt2pred, dim=1) / pos_grasps_in_view)

        ### Grasp baseline Loss
        cosine_distance = torch.tensor(1.) - torch.sum(torch.multiply(dir_labels_pc_cam, grasp_dir_head), dim=2)

        # only pass loss where we have labeled contacts near pc points
        masked_cosine_loss = torch.multiply(cosine_distance, grasp_success_labels_pc)
        dir_cosine_loss = torch.mean(torch.sum(masked_cosine_loss, dim=1) / pos_grasps_in_view)

        ### Grasp Approach Loss
        approach_labels_orthog = F.normalize(approach_labels_pc_cam - torch.sum(torch.multiply(grasp_dir_head, approach_labels_pc_cam), dim=2,keepdims=True) * grasp_dir_head, dim=2)
        cosine_distance_approach = torch.tensor(1.) - torch.sum(torch.multiply(approach_labels_orthog, approach_dir_head),dim=2)

        masked_approach_loss = torch.multiply(cosine_distance_approach, grasp_success_labels_pc)
        approach_cosine_loss = torch.mean(torch.sum(masked_approach_loss, dim=1) / pos_grasps_in_view)

        ### Grasp Offset/Thickness Loss
        if global_config['MODEL']['bin_offsets']:
            if global_config['LOSS']['offset_loss_type'] == 'softmax_cross_entropy':
                offset_loss = torch.zeros(grasp_offset_head.shape[0], grasp_offset_head.shape[1])
                for batch in range(offset_loss.shape[0]):
                    offset_loss[batch] = F.cross_entropy(grasp_offset_head[batch],
                                                        torch.argmax(offset_labels_pc[batch], dim=1), reduction='none')
                offset_loss = torch.mean(offset_loss)
            else:
                offset_loss = offset_labels_pc * -torch.log(torch.sigmoid(grasp_offset_head)) + (
                            1 - offset_labels_pc) * -torch.log(1 - torch.sigmoid(grasp_offset_head))

                if 'too_small_offset_pred_bin_factor' in global_config['LOSS'] and global_config['LOSS']['too_small_offset_pred_bin_factor']:
                    too_small_offset_pred_bin_factor = torch.tensor(global_config['LOSS']['too_small_offset_pred_bin_factor'], torch.float32)
                    collision_weight = (offset_labels_pc + torch.sum(offset_labels_pc, dim=2, keepdims=True) - torch.cumsum(offset_labels_pc, dim=2)) * too_small_offset_pred_bin_factor + torch.constant(1.)
                    offset_loss = torch.multiply(collision_weight, offset_loss)

                offset_loss = torch.mean(torch.multiply(torch.reshape(torch_bin_weights, (1, 1, -1)), offset_loss), axis=2)
        else:
            offset_loss = (grasp_offset_head[:, :, 0] - offset_labels_pc[:, :, 0]) ** 2
        masked_offset_loss = torch.multiply(offset_loss, grasp_success_labels_pc)
        offset_loss = torch.mean(torch.sum(masked_offset_loss, dim=1) / pos_grasps_in_view)

        ### Grasp Confidence Loss
        bin_ce_loss = torch.unsqueeze(grasp_success_labels_pc, dim=2) * -torch.log(
            torch.sigmoid(end_points['binary_score_head'])) + (
                                1 - torch.unsqueeze(grasp_success_labels_pc, dim=2)) * -torch.log(
            1 - torch.sigmoid(end_points['binary_score_head']))
        if 'topk_confidence' in global_config['LOSS'] and global_config['LOSS']['topk_confidence']:
            bin_ce_loss, _ = torch.topk(torch.squeeze(bin_ce_loss), k=global_config['LOSS']['topk_confidence'])
        bin_ce_loss = torch.mean(bin_ce_loss)

        loss_dict = {}
        loss_dict['dir_cosine_loss'] = dir_cosine_loss
        loss_dict['app_cosine_loss'] = approach_cosine_loss
        loss_dict['offset_loss'] = offset_loss
        loss_dict['score_loss'] = bin_ce_loss
        loss_dict['adds_loss'] = adds_loss
        loss_dict['adds_loss_gt2pred'] = adds_loss_gt2pred

        total_loss = 0
        if self.global_config['MODEL']['pred_contact_base']:  #false
            total_loss += self.global_config['OPTIMIZER']['dir_cosine_loss_weight'] * dir_cosine_loss
        if self.global_config['MODEL']['pred_contact_success']:
            total_loss += self.global_config['OPTIMIZER']['score_ce_loss_weight'] * bin_ce_loss
        if self.global_config['MODEL']['pred_contact_offset']:
            total_loss += self.global_config['OPTIMIZER']['offset_loss_weight'] * offset_loss
        if self.global_config['MODEL']['pred_contact_approach']: #false
            total_loss += self.global_config['OPTIMIZER']['approach_cosine_loss_weight'] * approach_cosine_loss
        if self.global_config['MODEL']['pred_grasps_adds']:
            total_loss += self.global_config['OPTIMIZER']['adds_loss_weight'] * adds_loss
        if self.global_config['MODEL']['pred_grasps_adds_gt2pred']: #false
            total_loss += self.global_config['OPTIMIZER']['adds_gt2pred_loss_weight'] * adds_loss_gt2pred
        return total_loss, loss_dict
